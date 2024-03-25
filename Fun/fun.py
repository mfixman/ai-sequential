import torch
import torch.nn as nn
import wandb
from wandb import Artifact

from torch import tensor
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer

import sys
import pickle

from typing import List, Dict, Tuple

sos = '<SOS>'
eos = '<EOS>'
pad = '<PAD>'
unk = '<UNK>'
special = {sos, eos, pad, unk}
device = 'cuda'

class Encoder(nn.Module):
	def __init__(self, len_vocab : int):
		super().__init__()
		self.embedding = nn.Embedding(
				num_embeddings = len_vocab,
				embedding_dim = 256,
		)
		self.gru = nn.GRU(
				input_size = 256,
				hidden_size = 128,
				num_layers = 1,
				batch_first = True,
		)

	def forward(self, input : tensor) -> Tuple[tensor, tensor]:
		x = self.embedding(input)
		output, hidden = self.gru(x)
		return output, hidden

class Decoder(nn.Module):
	def __init__(self, len_vocab : int):
		super().__init__()
		self.embedding = nn.Embedding(num_embeddings = len_vocab, embedding_dim = 256)
		self.relu = nn.ReLU()
		self.gru = nn.GRU(input_size = 256, hidden_size = 128, batch_first = True)
		self.out = nn.Linear(128, 128)
		self.out2 = nn.Linear(128, len_vocab)

	def forward(self, input : tensor, hidden : tensor):
		x = self.embedding(input)
		x, h = self.gru(x, hidden)
		x = self.out(x)
		x = self.relu(x)
		x = self.out2(x)
		return x, h

class Runner:
	loader : DataLoader
	encoder : Encoder
	decoder : Decoder
	optimiser : Optimizer
	criterion : nn.CrossEntropyLoss

	def __init__(self):
		self.load_data(n = 30)
		self.encoder = Encoder(len(self.vocab)).to(device)
		self.decoder = Decoder(len(self.vocab)).to(device)
		self.optimiser = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr = 1e-3, weight_decay = 1e-3)
		self.criterion = nn.CrossEntropyLoss().to(device)

		wandb.watch(self.encoder, log = 'all', log_freq = 100)
		wandb.watch(self.decoder, log = 'all', log_freq = 100)

	def load_data(self, n = None, batch_size = 256) -> DataLoader:
		print('Loading data', file = sys.stderr)
		self.vocab = pickle.load(open('vocab.pickle', 'rb'))

		train_text = pickle.load(open('train_text.pickle', 'rb'))[:n]
		train_high = pickle.load(open('train_high.pickle', 'rb'))[:n]

		val_text = pickle.load(open('val_text.pickle', 'rb'))[:n]
		val_high = pickle.load(open('val_high.pickle', 'rb'))[:n]

		self.text_len = max(max(len(x) for x in train_text), max(len(x) for x in val_text))
		self.high_len = max(max(len(x) for x in train_high), max(len(x) for x in val_high))

		tn = [tensor(y + [self.vocab[pad]] * (self.text_len - len(y))) for y in train_text]
		th = [tensor(y + [self.vocab[pad]] * (self.high_len - len(y))) for y in train_high]

		self.loader = DataLoader(list(zip(tn, th)), batch_size = batch_size, shuffle = True)

		self.val_text = tensor([(float(y) + [float(self.vocab[pad])] * (self.text_len - len(y))) for y in val_text], requires_grad = True)
		self.val_high = tensor([(float(y) + [float(self.vocab[pad])] * (self.high_len - len(y))) for y in val_high], requires_grad = True)
		print('Loaded data', file = sys.stderr)

	def run_part(self, text : tensor, high : tensor):
		self.optimiser.zero_grad()
		encoder_output, encoder_hidden = self.encoder(text)

		assert all(high[:, 0] == self.vocab[sos])
		decoder_input = high[:, 0]
		decoder_hidden = encoder_hidden

		loss = tensor(0.).to(device)
		text_len = text.size(1)
		high_len = high.size(1)
		for t in range(1, high_len):
			di1 = decoder_input.unsqueeze(1)
			decoder_output, decoder_hidden = self.decoder(di1, decoder_hidden)
			loss += self.criterion(decoder_output.squeeze(1), high[:, t])
			decoder_input = high[:, t]

		loss.backward()
		self.optimiser.step()

		return loss

	def log_models(self):
		self.log(self.encoder, 'encoder')
		self.log(self.decoder, 'decoder')

	@staticmethod
	def log(model, name):
		torch.save(model.state_dict(), f'{name}.pth')

		artifact = Artifact(f'{name}-weights', type = 'model')
		artifact.add_file(f'{name}.pth')
		wandb.log_artifact(artifact)

	def run_epoch(self):
		loss = 0
		for e, (text, high) in enumerate(self.loader):
			loss += self.run_part(text.to(device), high.to(device))

		return loss
	
	def val_loss(self):
		with torch.no_grad():
			val_loss = self.run_part(self.val_text.to(device), self.val_high.to(device))

		return val_loss


def main():
	torch.autograd.set_detect_anomaly(True)
	wandb.init(project = 'fun')
	runner = Runner()

	epochs = 1001
	last_val_loss = float('inf')
	for e in range(epochs):
		train_loss = runner.run_epoch()
		val_loss = runner.val_loss()

		print(f'Epoch {e}: train loss = {train_loss}, val loss = {val_loss}', file = sys.stderr)
		wandb.log({'Epoch': e, 'Train Loss': train_loss, 'Val Loss': val_loss})

		if e % 10 == 0 and val_loss < last_val_loss:
			print('Best loss found!', file = sys.stderr)
			runner.log_models()

if __name__ == '__main__':
	main()
