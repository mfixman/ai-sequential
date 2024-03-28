import datasets
import pickle
import sys
import os

from collections import defaultdict
from itertools import chain
from torchtext.transforms import BERTTokenizer
from nltk.corpus import stopwords, gutenberg

sos = '<SOS>'
eos = '<EOS>'
pad = '<PAD>'
unk = '<UNK>'
sym = '<SYM>'
bols = [sos, eos, pad, unk, sym]

class Tokeniser:
    tokeniser : BERTTokenizer
    stops : set[str]

    vocab_set : set[str]
    vocab : dict[str, int]

    lens : defaultdict
    lens = defaultdict(lambda: 0)

    def __init__(self):
        self.tokenizer = BERTTokenizer(
            vocab_path = 'https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt',
            do_lower_case = True,
            strip_accents = True,
            return_tokens = True,
        )
        self.stops = set(stopwords.words('english'))
        self.vocab_set = set()

    def good(self, word : str):
        if word in self.stops:
            return False

        return any(letter.isalnum() for letter in word)

    def tokenise_raw(self, phrases : list[str], part = str) -> list[list[str]]:
        print('Getting data', file = sys.stderr)
        lists = [self.tokenizer(str(p)) for p in phrases]

        print('Tokenising', file = sys.stderr)
        data = [[word for word in phrase if self.good(word)] for phrase in lists]

        print('Getting word set', file = sys.stderr)
        words = set(word for phrase in data for word in phrase)

        print('Adding to vocab', file = sys.stderr)
        self.vocab_set |= words
        self.lens[part] = max(self.lens[part], max(len(x) for x in data))

        print('Finished!', file = sys.stderr)
        return data

    @staticmethod
    def get_part(name : str) -> str:
        if 'text' in name:
            return 'text'

        if 'high' in name:
            return 'high'

        raise ValueError(f'Name not found: {name}')

    def tokenise(self, name_phrases : tuple[str, list[str]]) -> list[list[str]]:
        name, phrases = name_phrases
        try:
            return self.tokenise_raw(phrases, self.get_part(name))
        except Exception as e:
            print(f'Exception: {e}', file = sys.stderr)
            raise

    def save_vocab(self):
        self.vocab = dict(zip(bols + list(self.vocab_set), range(len(bols) + len(self.vocab_set))))
        print(f'Saved vocab with {len(self.vocab)} words, lens {self.lens}', file = sys.stderr)

    def fix(self, name : str, phrase : list[str]) -> list[str]:
        remaining = self.lens[self.get_part(name)] - len(phrase)
        return [sos] + phrase + [eos] + remaining * [pad]

    def vocabularise(self, name_tokens : tuple[str, list[list[str]]]):
        (name, tokens) = name_tokens

        print('Vocabularising', file = sys.stderr)
        return [[self.vocab[word] for word in self.fix(name, phrase)] for phrase in tokens]

def main():
    dataset = datasets.load_dataset('cnn_dailymail', '3.0.0')
    tokeniser = Tokeniser()

    names = ['train', 'validation', 'test']
    sets = {k: v for name in names for k, v in {f'{name}_text': dataset[name].data[0], f'{name}_high': dataset[name].data[1]}.items()}
    tokens = dict(zip(sets.keys(), map(tokeniser.tokenise, sets.items())))
    tokeniser.save_vocab()
    data = dict(zip(sets.keys(), map(tokeniser.vocabularise, tokens.items())))

    try:
        os.mkdir('pickles')
    except FileExistsError:
        pass

    print('Dumping data', file = sys.stderr)
    for k, v in data.items():
        pickle.dump(v, open(os.path.join('pickles', k + '.pickle'), 'wb'))
        pickle.dump(v[:10], open(os.path.join('pickles', k + '_10.pickle'), 'wb'))

    print('Dumping vocab', file = sys.stderr)
    pickle.dump(tokeniser.vocab, open('pickles/vocab.pickle', 'wb'))

if __name__ == '__main__':
    main()
