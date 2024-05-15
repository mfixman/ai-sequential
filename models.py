import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F
from torch.nn import Parameter

class SelfAttention(nn.Module):
	def __init__(self, input_size, out_size):
		super().__init__()
		self.dk_size = out_size
		self.query_linear = nn.Linear(in_features=input_size, out_features=out_size)
		self.key_linear = nn.Linear(in_features=input_size, out_features=out_size)
		self.value_linear = nn.Linear(in_features=input_size, out_features=out_size)
		self.softmax = nn.Softmax(dim=-1)  # Specify the dimension

	def forward(self, input_vector):
		query_out = F.relu(self.query_linear(input_vector))
		key_out = F.relu(self.key_linear(input_vector))
		value_out = F.relu(self.value_linear(input_vector))
		out_q_k = torch.bmm(query_out, key_out.transpose(1, 2)) / math.sqrt(self.dk_size)
		softmax_q_k = self.softmax(out_q_k)
		out_combine = torch.bmm(softmax_q_k, value_out)
		return out_combine

