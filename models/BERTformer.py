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
from transformers import BertModel

from models.SuperTransformer import SuperTransformer

class BERTformer(SuperTransformer):
	def __init__(self, input_dim, output_dim, pad_idx, model_settings):
		super().__init__()
		self.emb_size = 768  # bert hidden dim

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		for param in self.bert.parameters():
			param.requires_grad = False

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
		src = src[:, :512] # Trim src if exceed 512 (bert restriction)

		self.device = src.device

		tf_src, tf_trg, idf_src, idf_trg = rest

		# Remove <EOS> token from targets.
		trg = trg[:, :-1]
		tf_trg = tf_trg[:, :-1]
		idf_trg = idf_trg[:, :-1]

		# input shape: [batch_size, seq_len]
		N, src_seq_length = src.shape
		N, trg_seq_length = trg.shape

		# embed shape: [batch_size, seq_len, emd_dim]
		embed_trg = self.trg_word_embedding(trg)

		# Embed tf and idf metrics
		embed_tf_trg = self.tf_embedding(tf_trg)
		embed_idf_trg = self.idf_embedding(idf_trg)

		# src_positions, trg_positions shape: [1, seq_len, emb_dim]
		trg_positions = self.get_positional_encoding(trg_seq_length, self.emb_size)

		# Add positional encoding
		embed_trg += trg_positions
		# Add tf metrics
		embed_trg += embed_tf_trg
		# Add idf metrics
		embed_trg += embed_idf_trg

		# embed_trg shape: [batch_size, seq_len, emb_dim]

		# Create masks
		# src_mask, trg_mask shape: [seq_len, seq_len]	---  src_padding_mask, trg_padding_mask shape: [batch_size, seq_len]
		src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_mask(src, trg)

		# memoryshape: [batch_size, seq_len, emb_dim=512]
		token_type_ids = torch.zeros(N, src_seq_length, dtype=torch.long)
		
		bert_output = self.bert(
			input_ids = src,
			attention_mask = src_padding_mask,
			token_type_ids = token_type_ids
		)
		memory = bert_output.last_hidden_state
		
		pred_decoder_out = self.transformer.decoder(
			embed_trg, 
			memory, 
			tgt_mask=trg_mask, 
			tgt_key_padding_mask=trg_padding_mask,
			tgt_is_causal=False, 
			memory_is_causal=False
		)
		
		trg_decoder_out = self.transformer.decoder(
			embed_trg,
			memory,
			tgt_mask = None,
			tgt_key_padding_mask = trg_padding_mask,
			tgt_is_causal = False,
			memory_is_causal = False,
		)
		
		# output shape [batch_size, seq_len, vocab_size]
		output = self.fc_out(pred_decoder_out)
		return output, pred_decoder_out, trg_decoder_out
