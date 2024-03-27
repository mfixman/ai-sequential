import logging
import pickle
import random
import sys
import torch
import wandb
import wandb

from torch import nn, tensor, Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from wandb import Artifact
from wandb import Artifact

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
            embedding_dim = 256 // 2 // quotient,
        )
        self.dropout = nn.Dropout(p = 0.3)
        self.lstm = nn.LSTM(
            input_size = 256 // 2 // quotient,
            hidden_size = 128 // quotient,
            num_layers = 1,
            batch_first = True,
        )

    def forward(self, input : Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedding(input)
        x = self.dropout(x)
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, len_vocab : int, quotient = 1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = len_vocab, embedding_dim = 256 // quotient)
        self.lstm = nn.LSTM(input_size = 256 // quotient, hidden_size = 128 // quotient, batch_first = True)
        self.out = nn.Linear(128 // quotient, len_vocab)

    def forward(self, input : Tensor, hidden : Tensor):
        x = self.embedding(input)
        x, (h, c) = self.lstm(x, hidden)
        x = self.out(x)
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
        self.quotient = quotient
        self.encoder = Encoder(len(self.vocab), quotient).to(device)
        self.decoder = Decoder(len(self.vocab), quotient).to(device)
        self.optimiser = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr = 1e-3, weight_decay = 1e-3)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = self.vocab[pad]).to(device)

        wandb.watch(self.encoder, log = 'all', log_freq = 100)
        wandb.watch(self.decoder, log = 'all', log_freq = 100)

    def load_data(self, n = None, batch_size = 256) -> DataLoader:
        print('Loading data', file = sys.stderr)
        self.vocab = pickle.load(open('pickles/vocab.pickle', 'rb'))

        train_text = pickle.load(open('pickles/train_text.pickle', 'rb'))[:n]
        train_high = pickle.load(open('pickles/train_high.pickle', 'rb'))[:n]

        val_text = pickle.load(open('pickles/val_text.pickle', 'rb'))[:n]
        val_high = pickle.load(open('pickles/val_high.pickle', 'rb'))[:n]

        self.text_len = max(max(len(x) for x in train_text), max(len(x) for x in val_text))
        self.high_len = max(max(len(x) for x in train_high), max(len(x) for x in val_high))

        tn = [tensor(y + [self.vocab[pad]] * (self.text_len - len(y))) for y in train_text]
        th = [tensor(y + [self.vocab[pad]] * (self.high_len - len(y))) for y in train_high]
        self.loader = DataLoader(list(zip(tn, th)), batch_size = batch_size, shuffle = True)

        vn = [tensor(y + [self.vocab[pad]] * (self.text_len - len(y))) for y in val_text]
        vh = [tensor(y + [self.vocab[pad]] * (self.high_len - len(y))) for y in val_high]
        self.val_loader = DataLoader(list(zip(vn, vh))[:10 * batch_size], batch_size = batch_size, shuffle = True)

        logging.info('Loaded data')

    def run_part(self, text : Tensor, high : Tensor):
        _, hidden = self.encoder(text)

        assert all(high[:, 0] == self.vocab[sos])
        input = high[:, 0].unsqueeze(1)

        loss = tensor(0.).to(device)
        for t in range(1, self.high_len):
            output, hidden = self.decoder(input, hidden)
            _, topi = decoder_output.topk(1)
            input = topi.squeeze(-1).detach()

            loss += self.criterion(decoder_output.squeeze(1), high[:, t])

        return loss

    def train(self, text : Tensor, high : Tensor):
        self.optimiser.zero_grad()
        loss = self.run_part(text, high)
        loss.backward()
        self.optimiser.step()
        return loss

    def run_epoch(self, loader, training):
        loss = 0
        for e, (text, high) in enumerate(loader):
            if e % 50 == 0:
                logging.info(f'Running part {e}')

            if training:
                loss += self.train(text.to(device), high.to(device))
            else:
                loss += self.run_part(text.to(device), high.to(device))

        return loss / len(loader.dataset)

    def log_models(self):
        torch.save(self.encoder.state_dict(), 'encoder.pth')
        torch.save(self.decoder.state_dict(), 'decoder.pth')

        artifact = Artifact(f'weights', type = 'model', metadata = {'quotient': self.quotient})
        artifact.add_file('encoder.pth')
        artifact.add_file('decoder.pth')

        wandb.log_artifact(artifact)

def main():
    config = dict(
        n = 10,
        batch_size = 2,
        learner = 'lstm, half embedding, one linear',
        quotient = 2,
        epochs = 101,
    )
    wandb.init(project = 'fun', config = config)
    runner = Runner(n = config['n'], batch_size = config['batch_size'], quotient = config['quotient'])

    epochs = config['epochs']
    best_val_loss = float('inf')
    for e in range(1, epochs):
        logging.info(f'Training epoch {e}')
        train_loss = runner.run_epoch(runner.loader, training = True)

        logging.info(f'Validation epoch {e}')
        with torch.no_grad():
            val_loss = runner.run_epoch(runner.val_loader, training = False)

        print(f'Epoch {e}: train loss = {train_loss}, val loss = {val_loss}', file = sys.stderr)
        wandb.log({'Epoch': e, 'Train Loss': train_loss, 'Val Loss': val_loss})

        if val_loss < best_val_loss:
            print('Best loss found!', file = sys.stderr)
            runner.log_models()
            best_val_loss = val_loss

if __name__ == '__main__':
    main()
