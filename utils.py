import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def collate_fn(batch):
    text_tensors, summary_tensors = zip(*batch)

    text_tensors_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    summary_tensors_padded = pad_sequence(summary_tensors, batch_first=True, padding_value=0)
    
    return text_tensors_padded, summary_tensors_padded
    
def collate_fn_v2(batch):
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
    def __init__(self, weight_semantic=0.3, weight_ce=0.7, pad_idx=0, criterion='CrossEntropy'):
        """
        Parameters:
            - weight_semantic (float): Weight of the semantic similarity loss in the combined loss calculation.
            - weight_ce (float): Weight of the cross-entropy loss in the combined loss calculation.
            - pad_idx (int): Index used for padding in sequences, which should be ignored in loss calculations.
            - criterion (str): Choosen loss, 'CrossEntropy' (default) or 'CrossSimilarity'
        """
        super(CrossSimilarityLoss, self).__init__()
        self.weight_semantic = weight_semantic # Semantic loss weighting
        self.weight_ce = weight_ce # Cross-entropy loss weighting
        self.pad_idx = pad_idx # Padding index to ignore in loss calculations
        self.criterion = criterion # Choosen loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def get_loss(self, output_logits, target_tokens, embedded_pred, embedded_target):
        """
        If criterion=='CrossEntropy': calculate cross entropy. In this case embedded sequences will be ignored.
        If criterion=='CrossSimilarity': calculate the combined loss (weighted sum) consisting of cross-entropy and semantic similarity.

        Parameters:
        - output_logits (Tensor): The logits from the model's output [batch_size*seq_len, vocab_size].
        - target_tokens (Tensor): The ground truth token sequences [batch_size*seq_len].
        - embedded_pred (Tensor): Decoder output for the predicted sequence [seq_len, batch_size, emb_dim].
        - embedded_target (Tensor): Decoder output for the target sequence [seq_len, batch_size, emb_dim].
        
        Returns:
        - loss (Tensor): The calculated loss.
        """
        ce_loss = self.cross_entropy_loss(output_logits, target_tokens) # Cross entropy loss calculation
        if self.criterion == 'CrossEntropy':
            return ce_loss

        # Permute dimensions of embedded_pred and embedded_target to [batch_size, seq_len, emb_dim]
        embedded_pred = embedded_pred.permute(1, 0, 2)
        embedded_target = embedded_target.permute(1, 0, 2)

        # Semantic similarity loss calculation
        cosine_sim = F.cosine_similarity(embedded_pred, embedded_target, dim=2)
        semantic_loss = 1 - cosine_sim.mean()  # Average over batch

        loss = self.weight_ce * ce_loss + self.weight_semantic * semantic_loss # Weighted sum of the cross-entropy and semantic similarity losses

        return loss
