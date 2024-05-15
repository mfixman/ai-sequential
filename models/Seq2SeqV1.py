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

import torch.nn.functional as F

class Seq2SeqV1(nn.Module):
	def __init__(self, input_dim, output_dim, pad_idx, model_settings):
		super().__init__()

		if model_settings['num_layers'] != 3:
			logging.warn('Setting number of layers to 3 for Seq2Seq models.')

		num_layers = 3

		self.encoder = EncoderLSTM(input_dim, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], num_layers, model_settings['dropout'])
		self.decoder = AttDecoderLSTM(output_dim, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], num_layers, model_settings['dropout'])
		self.model = AttSeq2Seq(self.encoder, self.decoder)

		print('Using model Seq2Seq', flush = True)

	def forward(self, src, trg, *rest):
		self.device = src.device
		trg = trg[:, :-1]

		batch_size, trg_len = trg.shape
		trg_vocab_size = self.decoder.output_dim

		outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
		out_seq = torch.zeros(batch_size, trg_len, dtype = torch.long).to(self.device)

		encoder_outputs, hidden, cell = self.encoder(src)

		hidden = hidden[-1,:,:].unsqueeze(0)
		cell = cell[-1,:,:].unsqueeze(0)

		# Create a [num_layers, batch_size, hid_dim] -> We pass the context to all the "blocks" of the LSTM in the decoder
		hidden = torch.cat((hidden, hidden, hidden), dim=0)
		cell = torch.cat((cell, cell, cell), dim=0)

		# First input to the decoder is the <sos> tokens
		input = trg[:, 0]
		for t in range(1, trg_len):
			output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
			outputs[:, t, :] = output.squeeze(1)

			# Top-k sampling
			_, topi = output.squeeze(1).topk(1)
			input = topi.squeeze(-1).detach()

			out_seq[:, t] = input

		return outputs

class EncoderLSTM(nn.Module):
	def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2, dropout=0.2):
		"""
		:param input_dim: Size of the input.
		:param emb_dim: Dimensionality of the embedding layer.
		:param enc_hid_dim: Dimensionality of the encoder's hidden state.
		:param dec_hid_dim: Dimensionality of the decoder's hidden state.
		:param num_layers: Number of LSTM blocks.
		:param dropout: Dropout rate.
		"""
		super().__init__()
		self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
		self.rnn = nn.LSTM(emb_dim, enc_hid_dim, batch_first=True, num_layers=num_layers)
		self.dropout = nn.Dropout(dropout)

	def forward(self, src):
		"""
		:param src: Input tensor containing a batch of sequences.
		"""
		# src shape: [batch_size, src_len]

		embedded = self.dropout(self.embedding(src))
		# embedded shape: [batch_size, src_len, emb_dim]

		outputs, (hidden, cell) = self.rnn(embedded)
		# outputs shape: [batch_size, src_len, enc_hid_dim]
		# hidden, cell shapes: [num_layers, batch_size, enc_hid_dim]

		return outputs, hidden, cell

class DecoderLSTM(nn.Module):
	def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2):
		"""
		:param output_dim: Size of the output vocabulary.
		:param emb_dim: Dimensionality of the embedding layer.
		:param enc_hid_dim: Dimensionality of the encoder's hidden state.
		:param dec_hid_dim: Dimensionality of the decoder's hidden state.
		:param dropout: Dropout rate.
		"""
		super().__init__()
		self.embedding = nn.Embedding(output_dim, dec_hid_dim)
		self.rnn = nn.LSTM(dec_hid_dim, dec_hid_dim, num_layers=num_layers, batch_first=True)
		self.fc_out = nn.Linear(dec_hid_dim, output_dim)

		self.output_dim = output_dim

	def forward(self, input, hidden, cell, encoder_outputs):
		"""
		:param input: Input tensor containing a batch of tokens.
		:param hidden: Last hidden state.
		:param cell: Last cell state.
		:param encoder_outputs: All encoder outputs.
		"""
		#print(f"DEC INPUT SHAPE\ninput: {input.shape}\nhidden: {hidden.shape}\ncell:{cell.shape}\nenc_out: {encoder_outputs.shape}")
		# input shape: [batch_size]
		# hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]
		# encoder_outputs shape: [batch_size, src_len, enc_hid_dim]

		input = input.unsqueeze(-1)
		# input shape: [batch_size, trg_len=1]

		embedded = self.embedding(input)
		# embedded shape: [batch_size, trg_len=1, dec_hid_dim]

		# Combine embedded input word and encoder outputs
		rnn_input = F.relu(embedded)
		# rnn_input shape: [batch_size, trg_len=1, dec_hid_dim]

		output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
		# output shape: [batch_size, trg_len=1, dec_hid_dim]
		# hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]

		prediction = self.fc_out(output)
		# prediction shape: [batch_size, trg_len=1, output_dim==voc_dim]

		return prediction, hidden, cell  # Return the last layer states

class BahdanauAttention(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.Wa = nn.Linear(hidden_size, hidden_size)
		self.Ua = nn.Linear(hidden_size, hidden_size)
		self.Va = nn.Linear(hidden_size, 1)

	def forward(self, query, keys):
		scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
		scores = scores.squeeze(2).unsqueeze(1)
		# scores shape: [batch_size, trg_len=1, src_len]

		weights = F.softmax(scores, dim=-1)
		context = torch.bmm(weights, keys)

		return context, weights

class AttDecoderLSTM(nn.Module):
	def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2, dropout=0.2):
		"""
		:param output_dim: Size of the output vocabulary.
		:param emb_dim: Dimensionality of the embedding layer.
		:param enc_hid_dim: Dimensionality of the encoder's hidden state.
		:param dec_hid_dim: Dimensionality of the decoder's hidden state.
		:param num_layers: Number of blocks in the LSTM layer.
		:param dropout: Dropout rate.
		"""
		super().__init__()
		self.embedding = nn.Embedding(output_dim, dec_hid_dim)
		self.attention = BahdanauAttention(dec_hid_dim)
		self.rnn = nn.LSTM(2*dec_hid_dim, dec_hid_dim, num_layers=num_layers, batch_first=True)
		self.fc_out = nn.Linear(dec_hid_dim, output_dim)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(dec_hid_dim)

		self.output_dim = output_dim

	def forward(self, input, hidden, cell, encoder_outputs):
		"""
		:param input: Input tensor containing a batch of tokens.
		:param hidden: Last hidden state.
		:param cell: Last cell state.
		:param encoder_outputs: All encoder outputs.
		"""
		# input shape: [batch_size]
		# hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]
		# encoder_outputs shape: [batch_size, src_len, enc_hid_dim]

		input = input.unsqueeze(-1)
		# input shape: [batch_size, trg_len=1]

		embedded = self.dropout(self.embedding(input))
		# embedded shape: [batch_size, trg_len=1, dec_hid_dim]

		query = hidden[-1,:,:].unsqueeze(0).permute(1, 0, 2) # Pass the last hidden layer as query for attention
		# query shape: [batch_size, 1 (last layer), dec_hid_dim]

		context, att_weights = self.attention(query, encoder_outputs)
		# context shape: [batch_size, trg_len=1, dec_hid_dim]
		# att_weights shape: [batch_size, trg_len=1, src_len]
		context = self.layer_norm(context)
		# Combine embedded input word and encoder outputs
		rnn_input = torch.cat((embedded, context), dim=2)
		# rnn_input shape: [batch_size, trg_len=1, dec_hid_dim]

		output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
		# output shape: [batch_size, trg_len=1, dec_hid_dim]
		# hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]

		prediction = self.fc_out(output)
		# prediction shape: [batch_size, trg_len=1, output_dim==voc_dim]

		return prediction, hidden, cell, att_weights  # Return the last layer states

class AttSeq2Seq(nn.Module):
	def __init__(self, encoder, decoder):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.teacher_forcing_ratio = 0.5  # Initial teacher forcing ratio

	def forward(self, src, trg):
		self.device = src.device

		batch_size, trg_len = trg.shape
		_, src_len = src.shape
		trg_vocab_size = self.decoder.output_dim

		outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device) # Tensor to store decoder outputs
		out_seq = torch.zeros(batch_size, trg_len).to(self.device) # Tensor to store the output sequence
		attentions = torch.zeros(batch_size, trg_len, src_len).to(self.device) # Tensor to store attention matrices

		encoder_outputs, hidden, cell = self.encoder(src) # Encode the source sequence

		hidden = hidden[-1,:,:].unsqueeze(0) # Get the last hidden state of the encoder and unsqueeze to [1, batch, hid_dim]
		cell = cell[-1,:,:].unsqueeze(0) # Get the last cell state of the encoder and unsqueeze to [1, batch, hid_dim]

		# Create a [num_layers, batch_size, hid_dim] -> We pass the context to all the "blocks" of the LSTM in the decoder
		hidden = torch.cat((hidden, hidden, hidden), dim=0)
		cell = torch.cat((cell, cell, cell), dim=0)

		input = trg[:, 0] # First input to the decoder is the <sos> tokens
		out_seq[:, 0] = input

		print(f"Input: {input.shape}\n{input}")

		for t in range(1, trg_len):
			output, hidden, cell, att_weights = self.decoder(input, hidden, cell, encoder_outputs)
			# output shape: [batch_size, trg_len, voc_dim]
			outputs[:, t, :] = output.squeeze(1)
			attentions[:, t, :] = att_weights.squeeze(1)

			teacher_force = random.random() < self.teacher_forcing_ratio

			# Top-k sampling
			probs = torch.softmax(output, dim=-1)
			k = 10
			top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
			out = torch.multinomial(top_k_probs.view(-1, k), 1).view(-1, output.shape[1]) # Sampling from the top k probabilities to get the indices
			out = torch.gather(top_k_indices, 2, out.unsqueeze(-1)).squeeze(-1) # Map back the indices to vocabulary index
			out_seq[:,t] = out.squeeze(1)

			input = (trg[:, t] if teacher_force else out.squeeze(0)).detach()

			self.teacher_forcing_ratio -= (0.5 / 1000)	# Decrease by 0.0005 each step
			self.teacher_forcing_ratio = max(self.teacher_forcing_ratio, 0)  # Ensure it doesn't go below 0

		return outputs, out_seq.type(torch.LongTensor), attentions
