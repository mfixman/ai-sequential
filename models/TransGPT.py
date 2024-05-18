import torch
import wandb
import logging

from scores import rouge_scores
from collections import defaultdict
from dataset import NewsDataset
from torch import nn
from torch.nn import functional as F
from torch import tensor, FloatTensor, LongTensor
from torch.utils.data import DataLoader
from utils import collate_fn, collate_fn_v2, CrossSimilarityLoss
from wandb import Artifact
from transformers import GPT2Model

from models.SuperTransformer import SuperTransformer

class TransGPT(SuperTransformer):
	def __init__(self, input_dim, output_dim, pad_idx, model_settings):
		super().__init__()
		self.emb_size = 768

		self.gpt = GPT2Model.from_pretrained("openai-community/gpt2")
		for param in self.gpt.parameters():
			param.requires_grad = False

		self.src_word_embedding = nn.Embedding(output_dim, self.emb_size, padding_idx=pad_idx)
		self.trg_word_embedding = nn.Embedding(output_dim, self.emb_size, padding_idx=pad_idx)

		self.tf_embedding = nn.Embedding(output_dim, self.emb_size, padding_idx=pad_idx)
		self.idf_embedding = nn.Embedding(output_dim, self.emb_size, padding_idx=pad_idx)

		self.transformer = nn.Transformer(
			d_model=self.emb_size,
			nhead=8,
			num_encoder_layers=model_settings['num_layers'],
			num_decoder_layers=model_settings['num_layers'],
			dim_feedforward= 4 * self.emb_size,
			dropout=model_settings['dropout'],
			batch_first = True,
		)
		self.fc_out = nn.Linear(self.emb_size, output_dim)
		self.pad_idx = pad_idx

	def forward(self, src, trg, *rest):
		self.device = src.device

		tf_src, tf_trg, idf_src, idf_trg = rest

		src = src[:, :1024]
		trg = trg[:, :1024]
		tf_src = tf_src[:, :1024]
		tf_trg = tf_trg[:, :1024]
		idf_src = idf_src[:, :1024]
		idf_trg = idf_trg[:, :1024]

		# Remove <EOS> token from targets.
		trg = trg[:, :-1]
		tf_trg = tf_trg[:, :-1]
		idf_trg = idf_trg[:, :-1]

		# input shape: [batch_size, seq_len]
		N, src_seq_length = src.shape
		N, trg_seq_length = trg.shape

		# embed shape: [batch_size, seq_len, emd_dim]
		embed_src = self.src_word_embedding(src)
		embed_trg = self.trg_word_embedding(trg)

		# Embed tf and idf metrics
		embed_tf_src = self.tf_embedding(tf_src)
		embed_tf_trg = self.tf_embedding(tf_trg)
		embed_idf_src = self.idf_embedding(idf_src)
		embed_idf_trg = self.idf_embedding(idf_trg)

		# src_positions, trg_positions shape: [1, seq_len, emb_dim]
		trg_positions = self.get_positional_encoding(trg_seq_length, self.emb_size)
		src_positions = self.get_positional_encoding(src_seq_length, self.emb_size)

		# Add positional encoding
		embed_src += src_positions
		embed_trg += trg_positions
		# Add tf metrics
		embed_src += embed_tf_src
		embed_trg += embed_tf_trg
		# Add idf metrics
		embed_src += embed_idf_src
		embed_trg += embed_idf_trg

		# embed_src, embed_trg shape: [batch_size, seq_len, emb_dim]

		# Create masks
		src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_mask(src, trg)

		memory = self.transformer.encoder(
			embed_src,
			mask=src_mask, 
			src_key_padding_mask=src_padding_mask, 
			is_causal=False,
		)
		
		token_type_ids = torch.zeros(N, src_seq_length, dtype=torch.long).to(self.device)
		gpt_output = self.gpt(
			inputs_embeds=memory,
			token_type_ids = token_type_ids
		)
		
		output = gpt_output[:, :trg_seq_length, :]
		
		# output shape [batch_size, seq_len, vocab_size]
		output = self.fc_out(output)
		return output, None, None
