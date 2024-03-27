import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from utils import collate_fn
from dataset_tokenizer import SubwordDatasetTokenizer

class NewsDataset(Dataset):
    def __init__(self, data_dir, special_tokens, split_type, vocabulary_file):
        """
        Initializes the NewsDataset class.

        :param data_dir: Directory where the tokenized data is stored.
        :param split_type: Type of the dataset split ('train', 'test', or 'validation').
        :param vocabulary_file: Path to the vocabulary pickle file.
        :param special_tokens: Special tokens list
        """
        self.data_dir = data_dir
        self.special_tokens = special_tokens
        self.split_type = split_type

        # Load the vocabulary
        with open(vocabulary_file, 'rb') as vocab_file:
            self.vocabulary = pickle.load(vocab_file)

        # Create a reverse vocabulary for lookup
        self.idx_to_token = {idx: token for token, idx in self.vocabulary.items()}

        # Load the tokenized text and summaries
        self.texts = self.load_data(f'{split_type}_text.pickle')
        self.summaries = self.load_data(f'{split_type}_high.pickle')

    def load_data(self, file_name):
        """
        Loads the tokenized data from a pickle file.

        :param file_name: Name of the pickle file to load.
        """
        with open(os.path.join(self.data_dir, file_name), 'rb') as data_file:
            data = pickle.load(data_file)
        return data

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns the tokenized text and summary at the specified index, along with their numerical representations.

        :param idx: Index of the data point.
        """
        #print(f"\nText: {self.texts[idx]}")
        #print(f"Summary: {self.summaries[idx]}\n")
        text_tokens = [self.special_tokens[2]] + self.texts[idx] + [self.special_tokens[3]]
        summary_tokens = [self.special_tokens[2]] + self.summaries[idx] + [self.special_tokens[3]]

        # Convert tokens to their numerical representations
        text_numerical = [self.vocabulary.get(token, self.vocabulary[self.special_tokens[1]]) for token in text_tokens]
        summary_numerical = [self.vocabulary.get(token, self.vocabulary[self.special_tokens[1]]) for token in summary_tokens]

        # Convert lists to PyTorch tensors
        text_tensor = torch.tensor(text_numerical, dtype=torch.long)
        summary_tensor = torch.tensor(summary_numerical, dtype=torch.long)

        return text_tensor, summary_tensor
    
if __name__ == '__main__':
    dataset = NewsDataset(data_dir='tokenized_data_subword', split_type='validation', special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],  vocabulary_file=os.path.join('tokenized_data_subword', 'vocabulary.pickle'))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    subword_tokenizer = SubwordDatasetTokenizer(model_name='bert-base-uncased')
    for text, summary in dataloader:
        print(f"Text_shape: {text.shape}")
        text_detokenized = subword_tokenizer.tokenizer.decode(text[0].cpu().numpy())
        summary_detokenized = subword_tokenizer.tokenizer.decode(summary[0].cpu().numpy())
        print(f"\nText_loader: {text}\nText: {text_detokenized}")
        print(f"Sum_loader: {summary}\nSum: {summary_detokenized}\n")
        #print(dataset.vocabulary['give'])
        break