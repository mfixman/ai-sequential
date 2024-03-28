import logging
import pickle
import sys
import torch
import wandb
import wandb

from numpy import random
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
    def __init__(self, len_vocab : int, quotient = 1, padding_idx = None):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = len_vocab,
            embedding_dim = 512 // quotient,
            padding_idx = padding_idx,
        )
        self.dropout = nn.Dropout(p = 0.3)
        self.lstm = nn.LSTM(
            input_size = 512 // quotient,
            hidden_size = 256 // quotient,
            num_layers = 1,
            batch_first = True,
        )

    def forward(self, input : Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedding(input)
        x = self.dropout(x)
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, len_vocab : int, quotient = 1, padding_idx = None):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = len_vocab,
            embedding_dim = 512 // quotient,
            padding_idx = padding_idx,
        )
        self.lstm = nn.LSTM(input_size = 512 // quotient, hidden_size = 256 // quotient, batch_first = True)
        self.out = nn.Linear(256 // quotient, len_vocab)

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

    teacher_p : float

    def __init__(self, n, config):
        self.config = config

        self.load_data()
        self.encoder = Encoder(len(self.vocab), config['quotient'], self.vocab[pad]).to(device)
        self.decoder = Decoder(len(self.vocab), config['quotient'], self.vocab[pad]).to(device)
        self.optimiser = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr = 1e-3, weight_decay = 1e-3)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = self.vocab[pad]).to(device)
        self.teacher_p = 1

        wandb.watch(self.encoder, log = 'all', log_freq = 100)
        wandb.watch(self.decoder, log = 'all', log_freq = 100)

    def load_data(self) -> DataLoader:
        logging.info('Loading data')
        self.vocab = pickle.load(open('pickles/vocab.pickle', 'rb'))

        n = self.config['n']
        postfix = self.config['postfix']

        train_text = pickle.load(open('pickles/train_text' + postfix + '.pickle', 'rb'))[:n]
        train_high = pickle.load(open('pickles/train_high' + postfix + '.pickle', 'rb'))[:n]
        train_data = list(zip(tensor(train_text), tensor(train_high)))
        self.loader = DataLoader(train_data, batch_size = self.config['batch_size'], shuffle = True)

        val_text = pickle.load(open('pickles/validation_text' + postfix + '.pickle', 'rb'))[:n]
        val_high = pickle.load(open('pickles/validation_high' + postfix + '.pickle', 'rb'))[:n]
        val_data = list(zip(tensor(val_text), tensor(val_high)))[:50 * self.config['batch_size']]
        self.val_loader = DataLoader(val_data, batch_size = self.config['batch_size'], shuffle = True)

        self.text_len = len(train_text[0])
        self.high_len = len(train_high[0])

        logging.info('Loaded data')

    def run_part(self, text : Tensor, high : Tensor):
        _, hidden = self.encoder(text)

        # assert all(high[:, 0] == self.vocab[sos])
        input = high[:, 0].unsqueeze(1)

        student = self.teacher_p < random.random(self.high_len)
        loss = tensor(0.).to(device)
        for t in range(1, self.high_len):
            output, hidden = self.decoder(input, hidden)

            if student[t]:
                topv, topi = [x.squeeze() for x in output.topk(3)]
                choice = torch.multinomial(topv.squeeze(), 1)
                input = topi.gather(1, choice).detach()
            else:
                input = high[:, t - 1].unsqueeze(1)

            loss += self.criterion(output.squeeze(1), high[:, t])

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

        self.teacher_p *= .9
        return loss / len(loader.dataset)

    def log_models(self):
        torch.save(self.encoder.state_dict(), 'encoder.pth')
        torch.save(self.decoder.state_dict(), 'decoder.pth')

        artifact = Artifact(f'fun_weights', type = 'model', metadata = self.config)
        artifact.add_file('encoder.pth')
        artifact.add_file('decoder.pth')

        wandb.log_artifact(artifact)

def main():
    config = dict(
        n = 100000,
        batch_size = 300,
        learner = 'lstm, teacher forcing',
        quotient = 1,
        epochs = 101,
        postfix = '',
        loss = 'Categorical Cross-Entropy',
    )
    wandb.init(project = 'fun', config = config)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    runner = Runner(n = config['n'], config = config)

    epochs = config['epochs']
    best_val_loss = float('inf')
    for e in range(1, epochs):
        logging.info(f'Training epoch {e}')
        train_loss = runner.run_epoch(runner.loader, training = True)

        logging.info(f'Validation epoch {e}')
        with torch.no_grad():
            val_loss = runner.run_epoch(runner.val_loader, training = False)

        logging.info(f'Epoch {e}: train loss = {train_loss:g}, val loss = {val_loss:g}')
        wandb.log({'Epoch': e, 'Train Loss': train_loss, 'Val Loss': val_loss})

        if val_loss < best_val_loss:
            logging.info('Best loss found!')
            runner.log_models()
            best_val_loss = val_loss

if __name__ == '__main__':
    main()
