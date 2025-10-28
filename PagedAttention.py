# PagedAttention Naive Implementation
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

class Page:
  """
  A single page storing key and value tensors.
  """
  def __init__(self, page_size: int):
    # store tensors of shape (page_size, num_heads, head_dim)
    self.page_size = page_size
    self.key = None # torch.Tensor or None
    self.value = None
    self.filled = 0
  
  def init_tensors(self, num_heads: int, head_dim: int, device: torch.device):
    self.key = torch.zeros(self.page_size, num_heads, head_dim, device=device)
    self.value = torch.zeros(self.page_size, num_heads, head_dim, device=device)
    self.filled = 0
  
  def append(self, key: torch.Tensor, value: torch.Tensor):
    # key, value: (num_heads, head_dim)
    assert self.filled < self.page_size
    self.key[self.filled] = key
    self.value[self.filled] = value
    self.filled += 1

class PagedKVSequence:
  """
  Manage page tables for a single sequence.
  page_table: List[Page] (maybe shared across sequences)
  """
  def __init__(self, page_size: int, num_heads: int, head_dim: int, device: torch.device):
    self.page_size = page_size
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.device = device
    self.pages: List[Page] = []
    self.total_tokens = 0

  def append_kv(self, key: torch.Tensor, value: torch.Tensor):
    # key, value: (num_head, head_dim)
    if not self.pages or self.pages[-1].filled >= self.page_size:
      p = Page(self.page_size)
      p.init_tensors(self.num_heads, self.head_dim, self.device)
      self.pages.append(p)
    self.pages[-1].append(key, value)
    self.total_tokens += 1

  def gather_kv_till(self, upto_index_exclusive: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Gather all k / v tokens [0, upto_index_exclusive]
    assert upto_index_exclusive <= self.total_tokens
    if upto_index_exclusive == 0:
      # Return empty tensors
      return (torch.empty(0, self.num_heads, self.head_dim, device=self.device),
              torch.empty(0, self.num_heads, self.head_dim, device=self.device))

    # Collect parts from pages
    parts_k = []
    parts_v = []
    remaining = upto_index_exclusive
    for page in self.pages:
      take = min(page.filled, remaining)
      if take > 0:
        parts_k.append(page.key[:take]) # (take, num_heads, head_dim)
        parts_v.append(page.value[:take]) # (take, num_heads, head_dim)
        remaining -= take
      if remaining == 0:
        break

    # Concatenate parts
    full_k = torch.cat(parts_k, dim=0) # (upto_index_exclusive, num_heads, head_dim)
    full_v = torch.cat(parts_v, dim=0) # (upto_index_exclusive, num_heads, head_dim)
    return full_k, full_v
        
class DecoderLayer:
    def __init__(self, model_dim: int, num_heads: int, head_dim: int, page_size: int, device: torch.device):
      self.Wq = torch.nn.Linear(model_dim, num_heads * head_dim).to(device)
      self.Wk = torch.nn.Linear(model_dim, num_heads * head_dim).to(device)
      self.Wv = torch.nn.Linear(model_dim, num_heads * head_dim).to(device)
      self.out_proj = torch.nn.Linear(num_heads * head_dim, model_dim).to(device)
      self.model_dim = model_dim
      self.num_heads = num_heads
      self.head_dim = head_dim
      self.page_size = page_size

    def step(self, x_t: torch.Tensor, seq_caches: List[PagedKVSequence]) -> torch.Tensor:
      # x_t: (batch_size, model_dim) BatchSize batches of independent sequences
      batch_size = x_t.size(0)
      
      Q = self.Wq(x_t) # (batch_size, num_heads * head_dim)
      K = self.Wk(x_t) # (batch_size, num_heads * head_dim)
      V = self.Wv(x_t) # (batch_size, num_heads * head_dim)
      
      Qh = split_heads(Q.view(batch_size, 1, -1), self.num_heads).squeeze(2) # (batch_size, num_heads, head_dim)
      Kh = split_heads(K.view(batch_size, 1, -1), self.num_heads).squeeze(2) # (batch_size, num_heads, head_dim)
      Vh = split_heads(V.view(batch_size, 1, -1), self.num_heads).squeeze(2) # (batch_size, num_heads, head_dim)

      # append K/V  per sequence in cache
      for i in range(batch_size):
        seq_caches[i].append_kv(Kh[i].detach(), Vh[i].detach())
      # --- Batched inference path for LLM decoding ---
      # Gather K/V for all sequences in the batch, pad to the same length T_max,
      # build a padding mask and compute attention for the whole batch in one go.
      # This avoids a Python loop over batch elements and enables efficient
      # batch matrix multiplications on GPU.
      #
      # Shapes summary used below (symbols):
      # B = batch_size, H = num_heads, Dh = head_dim, T = total_kv_tokens (per sequence, variable)

      # Gather and pad K/V across the batch:
      # k_padded, v_padded: (B, T_max, H, Dh)
      # lengths: list[int] per sequence
      k_padded, v_padded, lengths = gather_kv_batch(seq_caches, device=Qh.device)

      # Prepare queries for batched attention: Qh is (B, H, Dh).
      # We want q shape (B, H, 1, Dh) for single-step query per sequence.
      q = Qh.unsqueeze(2)  # -> (B, H, 1, Dh)

      # Build attention mask from lengths: mask has shape (B, 1, 1, T_max)
      # Mask positions that are padding with a large negative so softmax -> 0.
      if k_padded.size(1) == 0:
        # No KV in batch yet (should be rare in real decoding), return zeros.
        return torch.zeros(batch_size, self.model_dim, device=Qh.device, dtype=Qh.dtype)

      device = Qh.device
      T_max = k_padded.size(1)
      # lengths -> mask: True where valid
      lens = torch.tensor(lengths, device=device, dtype=torch.long)
      # Create mask (B, T_max) where True => valid key, False => pad
      arange = torch.arange(T_max, device=device).unsqueeze(0)
      valid_mask = (arange < lens.unsqueeze(1))  # (B, T_max)
      # Expand to (B, 1, 1, T_max) to add to qk logits
      attn_mask = (~valid_mask).unsqueeze(1).unsqueeze(1)  # pad positions are True
      # Convert boolean mask to additive mask (large negative)
      neg_inf = -1e9
      additive_mask = attn_mask.to(dtype=Qh.dtype) * neg_inf  # (B,1,1,T_max)

      # Compute batched attention: out -> (B, H, 1, Dh)
      out = batched_paged_scaled_dot_product_attention(q, k_padded, v_padded, mask=additive_mask)
      out = out.squeeze(2)  # -> (B, H, Dh)

      # Merge heads and project back to model dim.
      # merge_heads accepts (B, H, Dh) -> returns (B, H*Dh)
      merged = merge_heads(out)  # -> (B, hidden_size)
      proj = self.out_proj(merged)  # -> (B, model_dim)
      return proj

def paged_scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
  # q: [batch_size, num_heads, 1, head_dim] (single query token per batch member)
  # k, v: [total_kv_tokens, num_heads, head_dim]
  # produce out: [batch_size, num_heads, 1, head_dim]
  # convert k, v to [1, num_heads, total_kv_tokens, head_dim]
  kt = k.permute(1, 0, 2).unsqueeze(0) # [1, num_heads, total_kv_tokens, head_dim]
  vt = v.permute(1, 0, 2).unsqueeze(0) # [1, num_heads, total_kv_tokens, head_dim]
  qk = torch.matmul(q, kt.transpose(-1, -2)) # [batch_size, num_heads, 1, totoal_kv_tokens]
  qk = qk / (q.shape[-1] ** 0.5)
  if mask is not None:
    qk = qk + mask # mask should be broadcastable
  attn = F.softmax(qk, dim=-1) # [batch_size, num_heads, 1, total_kv_tokens]
  out = torch.matmul(attn, vt) # [batch_size, num_heads, 1, head_dim]
  return out


def gather_kv_batch(seq_caches: List[PagedKVSequence], device: torch.device):
  """
  Gather K/V tensors for all sequences in the batch and pad them to the same length.

  Returns:
    k_padded: (B, T_max, H, Dh)
    v_padded: (B, T_max, H, Dh)
    lengths: list of lengths per sequence [L0, L1, ...] (ints)
  """
  batch = len(seq_caches)
  kv_list_k = []
  kv_list_v = []
  lengths = []
  for seq in seq_caches:
    L = seq.total_tokens
    lengths.append(L)
    if L == 0:
      # empty tensor with shape (0, H, Dh)
      kv_list_k.append(torch.empty(0, seq.num_heads, seq.head_dim, device=device))
      kv_list_v.append(torch.empty(0, seq.num_heads, seq.head_dim, device=device))
    else:
      k_full, v_full = seq.gather_kv_till(L)  # (L, H, Dh)
      kv_list_k.append(k_full)
      kv_list_v.append(v_full)

  T_max = max(lengths) if lengths else 0
  H = seq_caches[0].num_heads if batch > 0 else 0
  Dh = seq_caches[0].head_dim if batch > 0 else 0

  if T_max == 0:
    # Return empty padded tensors with second dim = 0
    return (torch.empty(batch, 0, H, Dh, device=device),
            torch.empty(batch, 0, H, Dh, device=device),
            lengths)

  k_padded = torch.zeros(batch, T_max, H, Dh, device=device)
  v_padded = torch.zeros(batch, T_max, H, Dh, device=device)
  for i in range(batch):
    Li = lengths[i]
    if Li > 0:
      k_padded[i, :Li] = kv_list_k[i]
      v_padded[i, :Li] = kv_list_v[i]

  return k_padded, v_padded, lengths


def batched_paged_scaled_dot_product_attention(q: torch.Tensor, k_padded: torch.Tensor, v_padded: torch.Tensor, mask=None) -> torch.Tensor:
  """
  Batched version of paged_scaled_dot_product_attention for inference.

  q: (B, H, 1, Dh)
  k_padded: (B, T, H, Dh)
  v_padded: (B, T, H, Dh)
  mask: (B, 1, 1, T) additive mask (large negative on pad positions)

  returns: out (B, H, 1, Dh)
  """
  # permute k/v to (B, H, T, Dh)
  kt = k_padded.permute(0, 2, 1, 3)  # (B, H, T, Dh)
  vt = v_padded.permute(0, 2, 1, 3)  # (B, H, T, Dh)

  # q (B, H, 1, Dh) @ kt.transpose -> (B, H, 1, T)
  qk = torch.matmul(q, kt.transpose(-1, -2))  # (B, H, 1, T)
  qk = qk / (q.shape[-1] ** 0.5)
  if mask is not None:
    qk = qk + mask
  attn = F.softmax(qk, dim=-1)  # (B, H, 1, T)
  out = torch.matmul(attn, vt)  # (B, H, 1, Dh)
  return out

# Utils for splitting and merging heads
def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
  # x: (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, T, head_dim)
  batch_size, seq_len, hidden_size = x.shape
  head_dim = hidden_size // num_heads
  x = x.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
  return x

def merge_heads(x: torch.Tensor) -> torch.Tensor:
  # Accept either:
  # - x: (batch_size, num_heads, seq_len, head_dim) -> returns (batch_size, seq_len, hidden_size)
  # - x: (batch_size, num_heads, head_dim)         -> returns (batch_size, hidden_size)
  if x.dim() == 3:
    # single-token sequence: treat seq_len = 1
    batch_size, num_heads, head_dim = x.shape
    x = x.unsqueeze(2)  # -> (batch_size, num_heads, 1, head_dim)
    squeezed = True
  else:
    squeezed = False

  batch_size, num_heads, seq_len, head_dim = x.shape
  hidden_size = num_heads * head_dim
  x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
  if squeezed:
    # return (batch_size, hidden_size)
    return x.squeeze(1)
  return x

def main():
  Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  NumHeads = 4
  HeadDim = 16
  PagedSize = 8
  model_dim = NumHeads * HeadDim
  BatchSize = 2 # 2 sequences in batch

  layer = DecoderLayer(model_dim, NumHeads, HeadDim, PagedSize, Device)
  seq_caches = [PagedKVSequence(PagedSize, NumHeads, HeadDim, Device) for _ in range(BatchSize)]
  
  # run 10 steps of incremental decoding
  x = torch.randn(BatchSize, model_dim, device=Device)
  for t in range(10):
    x = torch.randn(BatchSize, model_dim, device=Device)
    out = layer.step(x, seq_caches)
    if t % 2 == 0:
      print(f"Step {t}: produced shape {out.shape}") 
  
  # inspect pages
  for i, cache in enumerate(seq_caches):
    print(f"seq {i} pages: {len(cache.pages)}, total tokens: {cache.total_tokens}")

  print("done.")

if __name__ == "__main__":
  main()
  