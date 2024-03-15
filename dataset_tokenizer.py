import datasets
import pickle
import os
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

class DatasetTokenizer:
    def __init__(self, saving_path='', ngram_size=1):
        """
        Initializes the DatasetTokenizer class.

        :param saving_path: The directory path where the processed files will be saved.
        :param ngram_size: The size of the n-grams to generate (default is 1, which means unigrams).
        """
        self.saving_path = saving_path
        self.ngram_size = ngram_size
        self.punctuation = None
        self.stop_words = stopwords.words('english')
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.vocabulary = set()

    def load_dataset(self):
        """Loads the dataset."""
        return datasets.load_dataset('cnn_dailymail', '3.0.0')

    def tokenize_text(self, texts):
        """
        Tokenizes a list of texts using NLTK's word_tokenize function.
        """
        return [tokenize.word_tokenize(str(text)) for text in texts]

    def clean_text(self, tokenized_texts):
        """
        Cleans tokenized texts by removing stopwords and punctuation.
        """
        self.initialize_punctuation(tokenized_texts)
        return [[word for word in text if word.lower() not in self.stop_words and word not in self.punctuation]
                for text in tokenized_texts]
    
    def generate_ngrams(self, cleaned_texts):
        """
        Generates n-grams from cleaned texts.
        """
        if self.ngram_size == 1:
            return cleaned_texts
        return [[' '.join(ngram) for ngram in ngrams(text, self.ngram_size)] for text in cleaned_texts]

    def initialize_punctuation(self, texts):
        """Initializes punctuation set."""
        words_raw = set.union(*[set(text) for text in texts])
        self.punctuation = set(char for word in words_raw for char in word if not char.isalnum())

    def update_vocabulary(self, texts):
        """Updates the vocabulary set."""
        for text in texts:
            self.vocabulary.update(text)

    def convert_vocabulary(self):
        """Convert set vocabulary in dictionary adding the special tokens at the beginning."""
        vocab_dict = {token: idx for idx, token in enumerate(self.special_tokens)}
        vocab_dict.update({token: idx + len(self.special_tokens) for idx, token in enumerate(sorted(self.vocabulary))})
        return vocab_dict

    def save_data(self, name, data):
        """Saves data to a pickle file."""
        pickle.dump(data, open(os.path.join(self.saving_path, f'{name}.pickle'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def process_splits(self, dataset):
        """Processes each dataset split."""
        for split in dataset:
            print(f"Processing {split}...")
            raw_texts = dataset[split]['article']
            raw_summaries = dataset[split]['highlights']

            tokenized_texts = self.tokenize_text(raw_texts)
            tokenized_summaries = self.tokenize_text(raw_summaries)

            cleaned_texts = self.clean_text(tokenized_texts)
            cleaned_summaries = self.clean_text(tokenized_summaries)

            ngram_texts = self.generate_ngrams(cleaned_texts)
            ngram_summaries = self.generate_ngrams(cleaned_summaries)

            self.update_vocabulary(ngram_texts)
            self.update_vocabulary(ngram_summaries)

            self.save_data(f'{split}_text', ngram_texts)
            self.save_data(f'{split}_high', ngram_summaries)

    def tokenize_dataset(self):
        """Orchestrates the dataset tokenization process."""
        print("Loading dataset...")
        dataset = self.load_dataset()
        print(" done")
        self.process_splits(dataset)
        print(" done")
        print("Convert vocabulary")
        self.vocabulary = self.convert_vocabulary()
        print(" done")
        print("Saving vocabulary...")
        self.save_data('vocabulary', self.vocabulary)
        print(" done")

if __name__ == '__main__':
    tokenizer = DatasetTokenizer('tokenized_data')
    tokenizer.tokenize_dataset()