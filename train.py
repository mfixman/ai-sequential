import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import logging

from collections import defaultdict
from dataset import NewsDataset
from dataset_tokenizer import SubwordDatasetTokenizer
from logger import Logger
from scores import rouge_scores
from torch import tensor, FloatTensor, LongTensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import load_config, collate_fn, collate_fn_v2, CrossSimilarityLoss, select_model
from wandb import Artifact

from torchmetrics.text.rouge import ROUGEScore

torch.manual_seed(42)
# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained(
			'bert-base-uncased'
		)
		self.rougeScore = ROUGEScore()
		self.artifact_to_delete = None

	def train(self, data_settings, model_settings, train_settings, logger):
		# Dataset
		train_dataset = NewsDataset(
			data_dir=data_settings['dataset_path'],
			special_tokens=data_settings['special_tokens'],
			split_type='train',
			vocabulary_file=data_settings['vocabulary_path'],
			version=model_settings['version'],
			max_samples = train_settings['max_samples']
		)
		val_dataset = NewsDataset(
			data_dir=data_settings['dataset_path'],
			special_tokens=data_settings['special_tokens'],
			split_type='validation',
			vocabulary_file=data_settings['vocabulary_path'],
			version=model_settings['version'],
			max_samples = train_settings['max_samples']
		)

		if model_settings['version'] == '1':
			train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
			val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
		elif model_settings['version'] == '2':
			# Change collate function to v2
			train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_v2)
			val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_v2)

		# Model
		INPUT_DIM = len(train_dataset.vocabulary)
		OUTPUT_DIM = len(train_dataset.vocabulary)
		PAD_IDX = train_dataset.vocabulary[data_settings['special_tokens'][0]]
		print(f"\nVocabulary size: {INPUT_DIM}\n")
		model = select_model(INPUT_DIM, OUTPUT_DIM, PAD_IDX, model_settings, device)

		# Optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

		# Loading checkpoint
		epoch_start = 0
		if train_settings['load_checkpoint']:
			ckpt = torch.load(f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_v{model_settings['version']}_ckt.pth", map_location=device)
			model_weights = ckpt['model_weights']
			model.load_state_dict(model_weights)
			optimizer_state = ckpt['optimizer_state']
			optimizer.load_state_dict(optimizer_state)
			epoch_start = ckpt['epoch']
			print("Model's pretrained weights loaded!")

		# Loss
		criterion = CrossSimilarityLoss(
						weight_semantic=0.2,
						weight_ce=0.8,
						pad_idx=train_dataset.vocabulary[data_settings['special_tokens'][0]],
						criterion=train_settings['loss'])

		# Train loop
		min_loss = float('inf')
		for epoch in range(epoch_start, train_settings['epochs']):
			train_loss = self.train_loop(model, train_loader, criterion, optimizer, model_settings)
			val_loss, val_scores = self.validation_loop(model, val_loader, criterion, model_settings)

			# print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Score: {val_score:.3f}')
			info = {'train_loss': train_loss, 'validation_loss': val_loss, **val_scores}
			logging.info('\t'.join(f'{k}: {v:g}' for k, v in info.items()))
			if logger:
				logger.log(info)

			# Save checkpoint if improvement
			if val_loss < min_loss:
				print(f'Loss decreased ({min_loss:.4f} --> {val_loss:.4f}). Saving model ...')
				ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
				# torch.save(ckpt, f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_v{model_settings['version']}_ckt.pth")
				self.save_artifacts(model, optimizer)
				min_loss = val_loss

	def save_artifacts(self, model, optimizer, **metadata):
		artifact = dict(
			model_weights = model.state_dict(),
			optimizer_state = optimizer.state_dict(),
		)
		torch.save(artifact, 'model.pth')

		api = wandb.Api()
		wandb_label = f'{wandb.run.id}_best'
		name = 'model_weights'

		artifact = Artifact(name, type = 'model', metadata = metadata)
		artifact.add_file('model.pth')

		labels = [wandb_label]
		wandb.log_artifact(artifact, aliases = labels)

		if self.artifact_to_delete is not None:
			logging.info(f'Deleting old artifact with ID {self.artifact_to_delete.id}')
			self.artifact_to_delete.delete()
			self.artifact_to_delete = None

		try:
			old_artifact = api.artifact(f'{wandb.run.entity}/{wandb.run.project}/{name}:{wandb_label}')
			old_artifact.aliases = []
			old_artifact.save()

			self.artifact_to_delete = old_artifact
		except wandb.errors.CommError as e:
			logging.info(f'First artifact, not deleting ({e})')

	def train_loop(self, model, train_loader, criterion, optimizer, model_settings, clip=1):
		logging.info('Starting training')
		model.train()
		epoch_loss = 0
		dec_out = None # Placeholder
		emb_trg = None # Placeholder

		for i, (src, trg, *rest) in enumerate(train_loader):
			if i % 10 == 0:
				logging.info(f'Parsing {i}/{len(train_loader)}')

			src, trg = src.to(device), trg.to(device)
			if model_settings['version'] == '1' and model_settings['model_name'] == 'seq2seq':
				output, out_seq, attentions = model(src, trg) # trg shape: [batch_size, trg_len]
				output_dim = output.shape[-1]
				output = output[1:].view(-1, output_dim)
				trg = trg[1:].reshape(-1)
			elif model_settings['version'] == '1' and model_settings['model_name'] == 'transformer':
				trg_input = trg[:, :-1] #remove last token of trg
				output; dec_out, emd_trg = model(src, trg_input)
				output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
				trg = trg[:, 1:].reshape(-1) # Reshape to [batch_size*trg_len]
				output = output.reshape(-1, output.shape[-1])  # Reshape to [batch_size*trg_len, vocab_size]
			elif model_settings['version'] == '2' and model_settings['model_name'] == 'transformer':
				tf_src, tf_trg, idf_src, idf_trg = rest
				tf_src, tf_trg, idf_src, idf_trg = tf_src.to(device), tf_trg.to(device), idf_src.to(device), idf_trg.to(device)

				trg_input = trg[:, :-1] #remove last token of trg
				tf_trg_input = tf_trg[:, :-1]
				idf_trg_input = idf_trg[:, :-1]
				output, dec_out, emb_trg = model(src, trg_input, tf_src, tf_trg_input, idf_src, idf_trg_input)
				output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
				trg = trg[:, 1:].reshape(-1) # Remove first token and reshape to [batch_size*trg_len]
				output = output.reshape(-1, output.shape[-1]) # Reshape to [batch_size*trg_len, vocab_size]
			else:
				raise ValueError(f"Model version {model_settings['version']} with model name {model_settings['model_name']} not valid!")

			optimizer.zero_grad()
			loss = criterion.get_loss(output, trg, dec_out, emb_trg)
			l1_lambda = 0.00001
			l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
			loss += l1_lambda*l1_norm
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
			optimizer.step()
			epoch_loss += loss.item()

		avg_loss = epoch_loss / len(train_loader)
		return avg_loss

	@torch.no_grad()
	def validation_loop(self, model, val_loader, criterion, model_settings) -> tuple[FloatTensor, dict[str, FloatTensor]]:
		logging.info('Starting validation')
		model.eval()

		epoch_loss = tensor(0.).to(device)
		sum_scores = defaultdict(lambda: tensor(0.).to(device))
		dec_out = None # Placeholder
		emb_trg = None # Placeholder

		for i, (src, trg, *rest) in enumerate(val_loader):
			if i % 10 == 0:
				logging.info(f'Parsing {i}/{len(val_loader)}')

			src, trg = src.to(device), trg.to(device)

			if model_settings['version'] == '1' and model_settings['model_name'] == 'seq2seq':
				output, out_seq, attentions = model(src, trg) # trg shape: [batch_size, trg_len]
				output_dim = output.shape[-1]
				output = output[1:].view(-1, output_dim)
				trg = trg[1:].reshape(-1)
			elif model_settings['version'] == '1' and model_settings['model_name'] == 'transformer':
				trg_input = trg[:, :-1] #remove last token of trg
				output, dec_out, emb_trg = model(src, trg_input)
				output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
				trg = trg[:, 1:].reshape(-1)
				output = output.reshape(-1, output.shape[-1])
			elif model_settings['version'] == '2' and model_settings['model_name'] == 'transformer':
				tf_src, tf_trg, idf_src, idf_trg = rest
				tf_src, tf_trg, idf_src, idf_trg = tf_src.to(device), tf_trg.to(device), idf_src.to(device), idf_trg.to(device)

				trg_input = trg[:, :-1] #remove last token of trg
				tf_trg_input = tf_trg[:, :-1]
				idf_trg_input = idf_trg[:, :-1]
				output, dec_out, emb_trg = model(src, trg_input, tf_src, tf_trg_input, idf_src, idf_trg_input)
				output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
				trg = trg[:, 1:] # .reshape(-1)
			else:
				raise ValueError(f"Model version {model_settings['version']} with model name {model_settings['model_name']} not valid!")

			# Apologies for the CPU-bound `for`.
			for o, t in zip(output, trg):
				output_text = self.tokenizer.decode(o.argmax(dim = 1), skip_special_tokens = True).join(' ')
				trg_text = self.tokenizer.decode(t, skip_special_tokens = True).join(' ')
				rouges = self.rougeScore(output_text, trg_text)
				sum_scores = {k: sum_scores[k] + v for k, v in rouges.items()}

			loss = criterion.get_loss(output.reshape(-1, output.shape[2]), trg.reshape(-1), dec_out, emb_trg)
			l1_lambda = 0.00001
			l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
			loss += l1_lambda*l1_norm
			epoch_loss += loss

		avg_loss = epoch_loss / len(val_loader)
		avg_scores = {k: v / len(val_loader.dataset) for k, v in sum_scores.items()}
		return avg_loss, avg_scores

def main():
	config = load_config()

	logging.basicConfig(
		level = logging.INFO,
		format = '[%(asctime)s] %(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)

	data_setting = config['data_settings']
	model_setting = config['model_params']
	train_setting = config['train']

	if train_setting['log']:
		wandb_logger = Logger(
			f"{model_setting['model_name']}_v{model_setting['version']}_lr={train_setting['learning_rate']}_L1",
			project='NewSum',
			config = config,
		)
		logger = wandb_logger.get_logger()
	else:
		logger = None

	trainer = Trainer()
	trainer.train(data_setting, model_setting, train_setting, logger)

if __name__ == '__main__':
	main()
