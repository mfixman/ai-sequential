import torch
import wandb
import logging

from scores import rouge_scores
from collections import defaultdict
from dataset import NewsDataset
from torch import nn
from torch import tensor, FloatTensor, LongTensor
from torch.utils.data import DataLoader
from utils import collate_fn, collate_fn_v2, CrossSimilarityLoss
from wandb import Artifact

class SuperTransformer(nn.Module):
	def __init__(self):
		super().__init__()

	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0,1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def create_mask(self, src, tgt):
		src_seq_len = src.shape[1]
		tgt_seq_len = tgt.shape[1]

		tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
		src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool) # No mask for the source

		src_padding_mask = (src == self.pad_idx)
		tgt_padding_mask = (tgt == self.pad_idx)
		return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

	def get_positional_encoding(self, seq_len, emb_size):
		"""Generate positional encodings for a given sequence length and embedding size."""
		position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1).to(self.device)
		div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(torch.log(tensor(10000.)) / emb_size)).to(self.device)

		positional_encoding = torch.zeros(seq_len, emb_size).to(self.device)
		positional_encoding[:, 0::2] = torch.sin(position * div_term)
		positional_encoding[:, 1::2] = torch.cos(position * div_term)

		return positional_encoding.unsqueeze(0)
