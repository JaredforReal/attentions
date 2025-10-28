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

    def step(self, x_t: torch.Tensor, seq_cache: PagedKVSequence) -> torch.Tensor:
      # x_t: (batch_size, model_dim) BatchSize batches of independent sequences
      batch_size = x_t.size(0)
      
      Q = self.Wq(x_t) # (batch_size, num_heads * head_dim)
      K = self.Wk(x_t) # (batch_size, num_heads * head_dim)
      V = self.Wv(x_t) # (batch_size, num_heads * head_dim)
      
      Qh = split_heads(Q, self.num_heads, self.head_dim) # (batch_size, num_heads, head_dim)
      Kh = split_heads(K, self.num_heads, self.head_dim) # (batch_size, num_heads, head_dim)
      Vh = split_heads(V, self.num_heads, self.head_dim) # (batch_size, num_heads, head_dim)


# Utils for splitting and merging heads
def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
  # x: (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, T, head_dim)
  batch_size, seq_len, hidden_size = x.shape
  head_dim = hidden_size // num_heads
  x = x.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
  return x

def merge_heads(x: torch.Tensor) -> torch.Tensor:
  # x: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, hidden_size)
  batch_size, num_headers, seq_len, head_dim = x.shape
  hidden_size = num_headers * head_dim
  x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
  return x
  


def main():
  Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  NumHeads = 4
  HeadDim = 16
  PagedSize = 8
  model_dim = NumHeads * HeadDim
  BatchSize = 2

  layer = DecoderLayer(model_dim, NumHeads, HeadDim, PagedSize, Device)
  