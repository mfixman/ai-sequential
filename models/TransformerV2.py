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

from models.SuperTransformer import SuperTransformer

class TransformerV2(SuperTransformer):
	def __init__(self, input_dim, output_dim, pad_idx, model_settings):
		super().__init__()
		emb_size = model_settings['encoder_embedding_dim']

		self.src_word_embedding = nn.Embedding(output_dim, emb_size, padding_idx=pad_idx)
		self.trg_word_embedding = nn.Embedding(output_dim, emb_size, padding_idx=pad_idx)

		self.tf_embedding = nn.Embedding(output_dim, emb_size, padding_idx=pad_idx)
		self.idf_embedding = nn.Embedding(output_dim, emb_size, padding_idx=pad_idx)

		self.emb_size = emb_size
		self.transformer = nn.Transformer(
			d_model=emb_size,
			nhead=8,
			num_encoder_layers=model_settings['num_layers'],
			num_decoder_layers=model_settings['num_layers'],
			dim_feedforward= 4 * emb_size,
			dropout=model_settings['dropout']
		)
		self.fc_out = nn.Linear(emb_size, output_dim)
		self.pad_idx = pad_idx

	def forward(self, src, trg, *rest):
		self.device = src.device

		tf_src, tf_trg, idf_src, idf_trg = rest

		trg = trg[:, :-1]
		tf_trg = tf_trg[:, :-1]
		idf_trg = idf_trg[:, :-1]

		N, src_seq_length = src.shape
		N, trg_seq_length = trg.shape
		# input shape: [batch_size, seq_len]

		embed_src = self.src_word_embedding(src)
		embed_trg = self.trg_word_embedding(trg)
		# embed shape: [batch_size, seq_len, emd_dim]

		# Embed tf and idf metrics
		embed_tf_src = self.tf_embedding(tf_src)
		embed_tf_trg = self.tf_embedding(tf_trg)
		embed_idf_src = self.idf_embedding(idf_src)
		embed_idf_trg = self.idf_embedding(idf_trg)

		trg_positions = self.get_positional_encoding(trg_seq_length, self.emb_size)
		src_positions = self.get_positional_encoding(src_seq_length, self.emb_size)
		# src_positions, trg_positions shape: [1, seq_len, emb_dim]

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
		# src_mask, trg_mask shape: [seq_len, seq_len]	---  src_padding_mask, trg_padding_mask shape: [batch_size, seq_len]

		# Reshape for nn.Transformer
		embed_src = embed_src.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]
		embed_trg = embed_trg.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]

		decoder_out = self.transformer(embed_src, embed_trg, src_mask=src_mask, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask)
		# decoder_out shape: [seq_len, batch_size, emb_dim]

		out = self.fc_out(decoder_out)
		# out shape [seq_len, batch_size, vocab_size]

		output = out.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
		return output
