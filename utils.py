import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, FloatTensor, LongTensor

def load_config(config_path='config.yaml'):
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config

def collate_fn(batch: list[LongTensor]) -> tuple[LongTensor, LongTensor]:
	text_tensors, summary_tensors = zip(*batch)

	text_tensors_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
	summary_tensors_padded = pad_sequence(summary_tensors, batch_first=True, padding_value=0)
	
	return text_tensors_padded, summary_tensors_padded
	
def collate_fn_v2(batch: list[LongTensor]) -> tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
	text_tensors, summary_tensors, tf_text_tensors, tf_summary_tensors, idf_text_tensors, idf_summary_tensors = zip(*batch)
		
	text_tensors_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
	summary_tensors_padded = pad_sequence(summary_tensors, batch_first=True, padding_value=0)
	tf_text_tensors_padded = pad_sequence(tf_text_tensors, batch_first=True, padding_value=0)
	tf_summary_tensors_padded = pad_sequence(tf_summary_tensors, batch_first=True, padding_value=0)
	idf_text_tensors_padded = pad_sequence(idf_text_tensors, batch_first=True, padding_value=0)
	idf_summary_tensors_padded = pad_sequence(idf_summary_tensors, batch_first=True, padding_value=0)

	return text_tensors_padded, summary_tensors_padded, tf_text_tensors_padded, tf_summary_tensors_padded, idf_text_tensors_padded, idf_summary_tensors_padded

class CrossSimilarityLoss():
	"""
	A custom loss class that computes:
	1. cross-entropy loss
	2. combines cross-entropy loss and semantic similarity (cosine similarity) loss.
	"""
	def __init__(self, varkappa = 0.2, pad_idx=0):
		"""
		Parameters:
			- pad_idx (int): Index used for padding in sequences, which should be ignored in loss calculations.
		"""
		super().__init__()
		self.varkappa = varkappa
		self.include_cs_loss = self.varkappa > 0
	
		self.pad_idx = pad_idx # Padding index to ignore in loss calculations
		self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

	def cosine_similarity_loss(self, target_tokens: LongTensor, embedded_pred: FloatTensor, embedded_target: FloatTensor) -> FloatTensor:
		# Create a mask for padding tokens
		mask = (target_tokens.view(embedded_pred.shape[0], embedded_pred.shape[1]) != self.pad_idx).unsqueeze(-1)  # Shape [batch_size, seq_len, 1]

		# Expand the mask to the embedding dimension
		mask = mask.expand(-1, -1, embedded_pred.size(-1))	# Expand to [batch_size, seq_len, emb_dim]

		# Apply mask to embeddings
		masked_pred = embedded_pred * mask
		masked_target = embedded_target * mask

		# Semantic similarity loss calculation
		cosine_sims = (1 + F.cosine_similarity(masked_pred, masked_target, dim=2)) / 2

		# Calculate the mean only over non-padding elements
		# valid_tokens = mask.sum(dim=[1, 2]) / embedded_pred.size(2)  # Normalize by emb_dim to count tokens, not elements
		semantic_loss = cosine_sims.sum(dim=1).mean() # Normalize by number of valid tokens and average batch

		return semantic_loss

	def get_losses(self, output_logits: FloatTensor, target_tokens: LongTensor, embedded_pred: None | FloatTensor, embedded_target: None | FloatTensor) -> tuple[FloatTensor, float, float]:
		"""
		Parameters:
		- output_logits (Tensor): The logits from the model's output [batch_size*seq_len, vocab_size].
		- target_tokens (Tensor): The ground truth token sequences [batch_size*seq_len].
		- embedded_pred (Tensor): Decoder output for the predicted sequence [seq_len, batch_size, emb_dim].
		- embedded_target (Tensor): Decoder output for the target sequence [seq_len, batch_size, emb_dim].
		
		Returns:
		- loss (Tensor): The calculated loss.
		"""
		cce_loss = self.cross_entropy_loss(output_logits, target_tokens)
		if embedded_pred is None and embedded_target is None and self.include_cs_loss:
			return cce_loss

		semantic_loss = self.cosine_similarity_loss(target_tokens, embedded_pred, embedded_target)
		loss = (1 - self.varkappa) * cce_loss + self.varkappa * semantic_loss 
		return loss, cce_loss.item(), semantic_loss.item()
