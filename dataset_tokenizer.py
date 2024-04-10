import datasets
import pickle
import os
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from transformers import AutoTokenizer

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
        ignore = ['.']
        self.punctuation = set(char for word in words_raw for char in word if not char.isalnum() and word not in ignore)

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


class SubwordDatasetTokenizer(DatasetTokenizer):
    def __init__(self, saving_path='', model_name='bert-base-uncased'):
        """
        Initializes the SubwordDatasetTokenizer class.

        :param saving_path: The directory path where the processed files will be saved.
        :param model_name: The name of the pre-trained tokenizer model to use.
        """
        super().__init__(saving_path)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_text(self, texts):
        """
        Tokenizes a list of texts using the pre-trained WordPiece tokenizer.
        """
        return [self.tokenizer.tokenize(text) for text in texts]
    
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

            self.save_data(f'{split}_text', cleaned_texts)
            self.save_data(f'{split}_high', cleaned_summaries)
    
    def tokenize_dataset(self):
        """Orchestrates the dataset tokenization process."""
        print("Loading dataset...")
        dataset = self.load_dataset()
        print(" done")
        self.process_splits(dataset)
        print(" done")
        print("Saving vocabulary...")
        self.save_data('vocabulary', self.tokenizer.get_vocab())
        print(" done")


if __name__ == '__main__':
    subword_tokenizer = SubwordDatasetTokenizer('tokenized_data_subword_v2', model_name='bert-base-uncased')
    
    # Accessing special tokens and their attributes
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Special Tokens Used by the Tokenizer:")
    print("=======================================")
    print(f"Pad Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    print(f"Unknown Token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
    print(f"Start of Sequence Token: {tokenizer.cls_token}, ID: {tokenizer.cls_token_id}")
    print(f"End of Sequence Token: {tokenizer.sep_token}, ID: {tokenizer.sep_token_id}")
    print(f"Mask Token: {tokenizer.mask_token}, ID: {tokenizer.mask_token_id}")
    print("=======================================")

    # Tokenization
    subword_tokenizer.tokenize_dataset()

    #tokenizer = DatasetTokenizer('tokenized_data')
    #tokenizer.tokenize_dataset()
