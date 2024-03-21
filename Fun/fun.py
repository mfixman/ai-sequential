import numpy
import matplotlib

import torch
import torch.nn as nn
import datasets
import wandb

from torch import tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pickle

sos = '<SOS>'
eos = '<EOS>'
pad = '<PAD>'
unk = '<UNK>'
device = 'cuda'

class Encoder(nn.Module):
    def __init__(self, vocab : dict[str, int]):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = len(vocab),
            embedding_dim = 256,
        )
        self.gru = nn.GRU(
            input_size = 256,
            hidden_size = 128,
            num_layers = 1,
            batch_first = True,
        )

    def forward(self, input : tensor) -> tuple[tensor, tensor]:
        x = self.embedding(input)
        output, hidden = self.gru(x)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, vocab : dict[str, int]):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 256)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size = 256, hidden_size = 128, batch_first = True)
        self.out = nn.Linear(128, 128)
        self.out2 = nn.Linear(128, len(vocab))

    def forward(self, input : tensor, hidden : tensor):
        x = self.embedding(input)
        print(f'{x.shape} {hidden.shape} {torch.cuda.memory_allocated("cuda")}')
        x, h = self.gru(x, hidden)
        x = self.out(x)
        x = self.relu(x)
        x = self.out2(x)
        return x, h

class Runner:
    loader : DataLoader
    def __init__(self):
        self.loader = self.load_data()
        self.encoder = Encoder(self.vocab).to(device)
        self.decoder = Decoder(self.vocab).to(device)
        self.optimiser = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr = 1e-3, weight_decay = 1e-3)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def load_data(self, batch_size = 1) -> DataLoader:
        train_text = pickle.load(open('train_text.pickle', 'rb'))
        train_high = pickle.load(open('train_high.pickle', 'rb'))
        self.vocab = {k: e for e, k, in enumerate(pickle.load(open('vocab.pickle', 'rb')))}

        tn = [tensor([self.vocab[x] for x in y]) for y in train_text]
        tn = pad_sequence(tn, batch_first = True, padding_value = self.vocab[pad])

        th = [tensor([self.vocab[x] for x in y]) for y in train_high]
        th = pad_sequence(th, batch_first = True, padding_value = self.vocab[pad])

        return DataLoader(list(zip(tn, th)), batch_size = batch_size, shuffle = True)

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

    def run_epoch(self):
        for text, high in self.loader:
            self.run_part(text.to(device), high.to(device))

def main():
    torch.autograd.set_detect_anomaly(True)
    runner = Runner()

    epochs = 1
    for e in range(epochs):
        runner.run_epoch()
        print(f'Epoch {e}: loss = {loss.item()}')

if __name__ == '__main__':
    main()
