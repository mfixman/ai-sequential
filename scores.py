import torch
from torch import nn

from torch import FloatTensor, LongTensor, Tensor

def rouge_scores(output: FloatTensor, trg: LongTensor) -> dict[str, FloatTensor]:
    inferred = output.argmax(axis = 1)
    return rouge1_scores(inferred, trg) | rouge2_scores(inferred, trg)

# t : b x w
def count_repeated(t : LongTensor) -> Tensor:
    device = t.device

    b = t.shape[0]
    w = t.shape[1]

    numbers = torch.arange(0, b).to(device).unsqueeze(1).expand(b, w)
    flat = torch.stack([numbers.flatten(), t.flatten()], dim = 0)

    uniq, inv = flat.unique(return_inverse = True, dim = 1)
    inv = inv.reshape(b, w)
    print(uniq)
    print(flat)
    return inv

    exp = inv.expand(b, w, w)
    keep = exp == torch.arange(0, w).to(t.device).expand(w).unsqueeze(1)
    q = keep.cumsum(dim = -1)

    rep = q.T[keep.T] - 1
    return torch.stack([t, rep], dim = 1)

# This Rouge1 score takes each token as separate for repetitions.
# Note that this only takes a single element, not a batch size.
def rouge1_scores(inferred: LongTensor, trg: LongTensor) -> dict[str, FloatTensor]:
    infer_rep = count_repeated(inferred)
    trg_rep = count_repeated(trg)

    union = torch.cat([infer_rep, trg_rep], dim = 0)
    _, counts = torch.unique(union, return_counts = True, dim = 0)

    intersection_size = torch.sum(counts == 2)
    precision = intersection_size / inferred.shape[0]
    recall = intersection_size / trg.shape[0]
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), nan = 0)
    return dict(
        rouge1_precision = precision,
        rouge1_recall = recall,
        rouge1_f1 = f1,
    )

# This Rouge2 UNIQUES ALL THE 2-GRAMS.
# Note that this only takes a single element, not a batch size.
def rouge2_scores(inferred: LongTensor, trg: LongTensor) -> dict[str, FloatTensor]:
    bigrams_left = torch.unique(torch.stack([inferred[:-1], inferred[1:]]), dim = 1)
    bigrams_right = torch.unique(torch.stack([trg[:-1], trg[1:]]), dim = 1)
    combined = torch.cat([bigrams_left, bigrams_right], dim = 1)

    union, counts = torch.unique(combined, return_counts = True, dim = 1)

    intersection_size = torch.sum(counts == 2)
    precision = intersection_size / bigrams_left.shape[1]
    recall = intersection_size / bigrams_right.shape[1]
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), nan = 0)

    return dict(
        rouge2_precision = precision,
        rouge2_recall = recall,
        rouge2_f1 = f1,
    )
