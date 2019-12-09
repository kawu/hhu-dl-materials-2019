from typing import List, Tuple
import torch

# Type alias for Tensor type
TT = torch.TensorType

# Type aliases for language and person name
Lang = str
Name = str

# Dataset: a list of (name, language) pairs
DataSet = List[Tuple[Name, Lang]]
