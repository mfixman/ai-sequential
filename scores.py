import torch
from torch import nn

from torch import FloatTensor, LongTensor

def rouge_scores(output: FloatTensor, trg: LongTensor) -> dict[str, FloatTensor]:
    inferred = output.argmax(axis = 1)
    return rouge_scores_long(inferred, trg)

def rouge_scores_long(inferred: LongTensor, trg: LongTensor) -> dict[str, FloatTensor]:
    bigrams_left = torch.unique(torch.stack([inferred[:-1], inferred[1:]]), dim = 1)
    bigrams_right = torch.unique(torch.stack([trg[:-1], trg[1:]]), dim = 1)
    combined = torch.cat([bigrams_left, bigrams_right], dim = 1)

    union, counts = torch.unique(combined, return_counts = True, dim = 1)

    intersection_size = torch.sum(counts == 2)
    precision = intersection_size / bigrams_left.shape[1]
    recall = intersection_size / bigrams_right.shape[1]
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), nan = 0)

    return dict(
        precision = precision,
        recall = recall,
        f1 = f1,
    )
