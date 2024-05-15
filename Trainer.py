import torch
import wandb
import logging

from scores import rouge_scores
from collections import defaultdict
from dataset import NewsDataset
from torch import tensor, FloatTensor, LongTensor
from torch.utils.data import DataLoader
from utils import collate_fn, collate_fn_v2, CrossSimilarityLoss
from wandb import Artifact

from models.Seq2SeqV1 import Seq2SeqV1
from models.TransformerV1 import TransformerV1
from models.TransformerV2 import TransformerV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
	def __init__(self, data_settings, model_settings, train_settings, logger):
		self.artifact_to_delete = None
		self.data_settings = data_settings
		self.model_settings = model_settings
		self.train_settings = train_settings
		self.logger = logger

		self.train_dataset = NewsDataset(
			data_dir=self.data_settings['dataset_path'],
			special_tokens=self.data_settings['special_tokens'],
			split_type='train',
			vocabulary_file=self.data_settings['vocabulary_path'],
			version=self.model_settings['version'],
			max_samples = self.train_settings['max_samples']
		)
		self.val_dataset = NewsDataset(
			data_dir=self.data_settings['dataset_path'],
			special_tokens=self.data_settings['special_tokens'],
			split_type='validation',
			vocabulary_file=self.data_settings['vocabulary_path'],
			version=self.model_settings['version'],
			max_samples = self.train_settings['max_samples']
		)

		if self.model_settings['version'] == '1':
			self.collate_fn = collate_fn
		elif self.model_settings['version'] == '2':
			self.collate_fn = collate_fn_v2
		else:
			raise ValueError(f"Unknown version {self.model_settings['version']}")

		input_dim = len(self.train_dataset.vocabulary)
		output_dim = len(self.train_dataset.vocabulary)
		pad_idx = self.train_dataset.vocabulary[self.data_settings['special_tokens'][0]]
		if self.model_settings['version'] == '1' and self.model_settings['model_name'] == 'seq2seq':
			self.model = Seq2SeqV1(input_dim, output_dim, pad_idx, model_settings).to(device)
		elif self.model_settings['version'] == '1' and self.model_settings['model_name'] == 'transformer':
			self.model = TransformerV1(input_dim, output_dim, pad_idx, model_settings).to(device)
		elif self.model_settings['version'] == '2' and self.model_settings['model_name'] == 'transformer':
			self.model = TransformerV2(input_dim, output_dim, pad_idx, model_settings).to(device)
		else:
			raise ValueError(f"Unknown version and model {self.model_settings['version']} {self.model_settings['model_name']}")

	def train(self):
		train_loader = DataLoader(self.train_dataset, batch_size=self.train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=self.collate_fn)
		val_loader = DataLoader(self.val_dataset, batch_size=self.train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=self.collate_fn)

		# Optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_settings['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

		# Loading checkpoint
		epoch_start = 0
		if self.train_settings['load_checkpoint']:
			ckpt = torch.load(f"{self.train_settings['checkpoint_folder']}/{self.model_settings['model_name']}_v{self.model_settings['version']}_ckt.pth", map_location=device)
			model_weights = ckpt['model_weights']
			self.model.load_state_dict(model_weights)
			optimizer_state = ckpt['optimizer_state']
			self.optimizer.load_state_dict(optimizer_state)
			epoch_start = ckpt['epoch']
			print("Model's pretrained weights loaded!")

		# Loss
		criterion = CrossSimilarityLoss(
			weight_semantic=0.2,
			weight_ce=0.8,
			pad_idx=self.train_dataset.vocabulary[self.data_settings['special_tokens'][0]],
			criterion=self.train_settings['loss']
		)

		# Train loop
		min_loss = float('inf')
		for epoch in range(epoch_start, self.train_settings['epochs']):
			train_loss = self.train_loop(train_loader, criterion)
			val_loss, val_scores = self.validation_loop(val_loader, criterion)

			# print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Score: {val_score:.3f}')
			info = {'train_loss': train_loss, 'validation_loss': val_loss, **val_scores}
			logging.info('\t'.join(f'{k}: {v:g}' for k, v in info.items()))
			if self.logger:
				self.logger.log(info)

			# Save checkpoint if improvement
			if val_loss < min_loss:
				print(f'Loss decreased ({min_loss:.4f} --> {val_loss:.4f}). Saving model ...')
				ckpt = {'epoch': epoch, 'model_weights': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict()}
				self.save_artifacts(self.model, self.optimizer)
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

	def train_loop(self, train_loader, criterion, clip=1):
		logging.info('Starting training')
		epoch_loss = 0.
		dec_out = None # Placeholder
		emb_trg = None # Placeholder

		self.model.train()
		for i, (src, trg, *rest) in enumerate(train_loader):
			if i % 10 == 0:
				logging.info(f'Parsing {i}/{len(train_loader)}')

			src, trg = src.to(device), trg.to(device)
			rest = [r.to(device) for r in rest]
			output = self.model(src, trg, *rest)

			trg = trg[:, 1:].reshape(-1) # Reshape to [batch_size*trg_len]
			output = output.reshape(-1, output.shape[-1]) # Reshape to [batch_size*trg_len, vocab_size]

			self.optimizer.zero_grad()

			loss = criterion.get_loss(output, trg, dec_out, emb_trg)
			l1_lambda = 0.00001
			l1_norm = sum(torch.linalg.norm(p, 1) for p in self.model.parameters())
			loss += l1_lambda*l1_norm
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
			self.optimizer.step()
			epoch_loss += loss.item()

		avg_loss = epoch_loss / len(train_loader)
		return avg_loss

	@torch.no_grad()
	def validation_loop(self, val_loader, criterion) -> tuple[FloatTensor, dict[str, FloatTensor]]:
		logging.info('Starting validation')

		epoch_loss = 0.
		sum_scores : dict[str, float] = defaultdict(lambda: 0.)
		dec_out = None # Placeholder
		emb_trg = None # Placeholder

		self.model.eval()
		for i, (src, trg, *rest) in enumerate(val_loader):
			if i % 10 == 0:
				logging.info(f'Parsing {i}/{len(val_loader)}')

			src, trg = src.to(device), trg.to(device)
			rest = [r.to(device) for r in rest]
			output = self.model(src, trg, *rest)

			trg = trg[:, 1:]

			# Apologies for the CPU-bound `for`.
			for o, t in zip(output, trg):
				rouges = rouge_scores(o, t)
				sum_scores = {k: sum_scores[k] + v.item() for k, v in rouges.items()}

			loss = criterion.get_loss(output.reshape(-1, output.shape[2]), trg.reshape(-1), dec_out, emb_trg)
			l1_lambda = 0.00001
			l1_norm = sum(torch.linalg.norm(p, 1) for p in self.model.parameters())
			loss += l1_lambda*l1_norm
			epoch_loss += loss.item()

		avg_loss = epoch_loss / len(val_loader)
		avg_scores = {k: v / len(val_loader.dataset) for k, v in sum_scores.items()}
		return avg_loss, avg_scores
