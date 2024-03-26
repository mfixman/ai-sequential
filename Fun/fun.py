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
    def __init__(self, len_vocab : int, quotient = 1):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = len_vocab,
            embedding_dim = 256 // quotient,
        )
        self.lstm = nn.LSTM(
            input_size = 256 // quotient,
            hidden_size = 128 // quotient,
            num_layers = 1,
            batch_first = True,
        )

    def forward(self, input : tensor) -> tuple[tensor, tensor]:
        x = self.embedding(input)
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, len_vocab : int, quotient = 1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = len_vocab, embedding_dim = 256 // quotient)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 256 // quotient, hidden_size = 128 // quotient, batch_first = True)
        self.out = nn.Linear(128 // quotient, 128 // quotient)
        self.out2 = nn.Linear(128 // quotient, len_vocab)

    def forward(self, input : tensor, hidden : tensor):
        x = self.embedding(input)
        x, (h, c) = self.lstm(x, hidden)
        x = self.out(x)
        x = self.relu(x)
        x = self.out2(x)
        return x, (h, c)

class Runner:
    loader : DataLoader
    encoder : Encoder
    decoder : Decoder
    optimiser : Optimizer
    criterion : nn.CrossEntropyLoss
    val_loader : DataLoader

    def __init__(self, n, batch_size, quotient):
        self.load_data(n = n, batch_size = batch_size)
        self.encoder = Encoder(len(self.vocab), quotient).to(device)
        self.decoder = Decoder(len(self.vocab), quotient).to(device)
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

        vn = [tensor(y + [self.vocab[pad]] * (self.text_len - len(y))) for y in val_text]
        vh = [tensor(y + [self.vocab[pad]] * (self.high_len - len(y))) for y in val_high]
        self.val_loader = DataLoader(list(zip(vn, vh))[:batch_size], batch_size = batch_size, shuffle = True)

        print('Loaded data', file = sys.stderr)

    def run_part(self, text : tensor, high : tensor):
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(text)

        assert all(high[:, 0] == self.vocab[sos])
        decoder_input = high[:, 0].unsqueeze(1)
        decoder_hidden = (encoder_hidden, encoder_cell)

        loss = tensor(0.).to(device)
        text_len = text.size(1)
        high_len = high.size(1)
        for t in range(1, high_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()

            loss += self.criterion(decoder_output.squeeze(1), high[:, t])

        return loss

    def train(self, text : tensor, high : tensor):
        self.optimiser.zero_grad()
        loss = self.run_part(text, high)
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

    def run_epoch(self, loader, training):
        loss = 0
        for e, (text, high) in enumerate(loader):
            if e % 50 == 0:
                print(f'Running part {e}', file = sys.stderr)

            if training:
                loss += self.train(text.to(device), high.to(device))
            else:
                loss += self.run_part(text.to(device), high.to(device))

        return loss

def main():
    torch.autograd.set_detect_anomaly(True)

    config = dict(
        n = 10000,
        batch_size = 32,
        learner = 'lstm',
        quotient = 4,
        epochs = 101,
    )
    wandb.init(project = 'fun', config = config)
    runner = Runner(n = config['n'], batch_size = config['batch_size'], quotient = config['quotient'])

    epochs = config['epochs']
    last_val_loss = float('inf')
    for e in range(1, epochs):
        print(f'Training epoch {e}', file = sys.stderr)
        train_loss = runner.run_epoch(runner.loader, training = True)

        print(f'Validation epoch {e}', file = sys.stderr)
        with torch.no_grad():
            val_loss = runner.run_epoch(runner.val_loader, training = False)

        print(f'Epoch {e}: train loss = {train_loss}, val loss = {val_loss}', file = sys.stderr)
        wandb.log({'Epoch': e, 'Train Loss': train_loss, 'Val Loss': val_loss})

        if e % 10 == 0 and val_loss < last_val_loss:
            print('Best loss found!', file = sys.stderr)
            runner.log_models()

if __name__ == '__main__':
    main()
