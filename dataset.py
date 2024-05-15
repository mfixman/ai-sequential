import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from utils import collate_fn
from dataset_tokenizer import SubwordDatasetTokenizer
import logging

class NewsDataset(Dataset):
	def __init__(self, data_dir, special_tokens, split_type, vocabulary_file, version='1', max_samples = None):
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
		self.version = version

		logging.info(f'Loading {split_type} dataset')

		# Load the vocabulary
		with open(vocabulary_file, 'rb') as vocab_file:
			self.vocabulary = pickle.load(vocab_file)

		# Create a reverse vocabulary for lookup
		self.idx_to_token = {idx: token for token, idx in self.vocabulary.items()}

		# Load the tokenized text and summaries and the correspondive metrics
		self.texts = self.load_data(f'{split_type}_text.pickle')[:max_samples]
		self.summaries = self.load_data(f'{split_type}_high.pickle')[:max_samples]
		if self.version == '2':
			self.tf_texts = self.load_data(f'{split_type}_tf_text.pickle')
			self.tf_summaries = self.load_data(f'{split_type}_tf_high.pickle')
			self.idf_texts = self.load_data(f'{split_type}_idf_text.pickle')
			self.idf_summaries = self.load_data(f'{split_type}_idf_high.pickle')

		#print(f"Init IDF: {self.idf_texts}")

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

		if self.version=='2':
			#print(f"tf: {self.tf_texts[idx]}") # COuunter with index x
			tf_text = [self.tf_texts[idx][word] for word in self.texts[idx]] # Get the values from the Counter for the defined (idx) text
			#print(f"tf text tokens: {tf_text}")
			tf_summary = [self.tf_summaries[idx][word] for word in self.summaries[idx]] # Get the values from the Counter (dictionary) for the selected (idx) summary

			idf_text = [self.idf_texts[word] for word in self.texts[idx]] # Get the values from the idf dictionary for each word in the sequence
			idf_summary = [self.idf_summaries[word] for word in self.summaries[idx]] # Get the values from the idf dictionary for each word in the sequence

			# Adjust shapes of tf/idf lists to include the start and end of sequence tokens -> 
			#-> we insert 0 values because we do not want to add additional information to the special tokens in the encoding!
			tf_text.append(0)
			tf_text.insert(0,0)
			tf_summary.append(0)
			tf_summary.insert(0,0)
			idf_text.append(0)
			idf_text.insert(0,0)
			idf_summary.append(0)
			idf_summary.insert(0,0)

			tf_text_tensor = torch.tensor(tf_text, dtype=torch.long)
			tf_summary_tensor = torch.tensor(tf_summary, dtype=torch.long)
			idf_text_tensor = torch.tensor(idf_text, dtype=torch.long)
			idf_summary_tensor = torch.tensor(idf_summary, dtype=torch.long)

			return text_tensor, summary_tensor, tf_text_tensor, tf_summary_tensor, idf_text_tensor, idf_summary_tensor
	
		return text_tensor, summary_tensor
	
if __name__ == '__main__':
	dataset = NewsDataset(data_dir='tokenized_data_subword_v2', split_type='validation', special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],  vocabulary_file=os.path.join('tokenized_data_subword_v2', 'vocabulary.pickle'))

	dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

	subword_tokenizer = SubwordDatasetTokenizer(model_name='bert-base-uncased')
	for text, summary, tf_text, tf_summary, idf_text, idf_summary in dataloader:
		# Check shapes
		print("\nShapes:")
		print(f"Text_shape: {text.shape}\nSummary_shape: {summary.shape}\ntf_text_shape: {tf_text.shape}")
		print(f"tf_summart_shape: {tf_summary.shape}\nidf_text_shape: {idf_text.shape}\nidf_sum_shape: {idf_summary.shape}\n\n")
		text_detokenized = subword_tokenizer.tokenizer.decode(text[0].cpu().numpy())
		summary_detokenized = subword_tokenizer.tokenizer.decode(summary[0].cpu().numpy())
		#print(f"\nText_loader: {text}\nText: {text_detokenized}")
		#print(f"Sum_loader: {summary}\nSum: {summary_detokenized}\n")
		#print(dataset.vocabulary['give'])
		break
