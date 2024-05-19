import torch
import wandb
import os 
import logging

from torch.utils.data import DataLoader
from utils import load_config, collate_fn, collate_fn_v2
from dataset import NewsDataset
from dataset_tokenizer import SubwordDatasetTokenizer
from collections import defaultdict
from scores import rouge_scores

from models.Seq2SeqV1 import Seq2SeqV1
from models.TransformerV1 import TransformerV1
from models.TransformerV2 import TransformerV2
from models.BERTformer import BERTformer
from models.TransGPT import TransGPT
from models.BERTGPT import BERTGPT

from typing import Any

torch.manual_seed(42)
#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(data_settings, model_settings, inference_settings):
    # Dataset
    test_dataset = NewsDataset(
			data_dir=data_settings['dataset_path'],
			special_tokens=data_settings['special_tokens'],
			split_type='test',
			vocabulary_file=data_settings['vocabulary_path'],
			version=model_settings['version']
		)
    
    if model_settings['version'] == '1':
        test_loader = DataLoader(test_dataset, batch_size=inference_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
    elif model_settings['version'] == '2':
        # Change collate function to v2
        test_loader = DataLoader(test_dataset, batch_size=inference_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_v2)

    # Model
    input_dim = len(test_dataset.vocabulary)
    output_dim = len(test_dataset.vocabulary)
    pad_idx = test_dataset.vocabulary[data_settings['special_tokens'][0]]
    if model_settings['version'] == '1' and model_settings['model_name'] == 'seq2seq':
        model = Seq2SeqV1(input_dim, output_dim, pad_idx, model_settings).to(device)
    elif model_settings['version'] == '1' and model_settings['model_name'] == 'transformer':
        model = TransformerV1(input_dim, output_dim, pad_idx, model_settings).to(device)
    elif model_settings['version'] == '2' and model_settings['model_name'] == 'transformer':
        model = TransformerV2(input_dim, output_dim, pad_idx, model_settings).to(device)
    elif model_settings['version'] == '2' and model_settings['model_name'] == 'bertformer':
        model = BERTformer(input_dim, output_dim, pad_idx, model_settings).to(device)
    elif model_settings['version'] == '2' and model_settings['model_name'] == 'transgpt':
        model = TransGPT(input_dim, output_dim, pad_idx, model_settings).to(device)
    elif model_settings['version'] == '2' and model_settings['model_name'] == 'bertgpt':
        model = BERTGPT(input_dim, output_dim, pad_idx, model_settings).to(device)
    else:
        raise ValueError(f"Unknown version and model {model_settings['version']} {model_settings['model_name']}")

    # Loading checkpoint
    ckpt = torch.load(f"{inference_settings['checkpoint_folder']}", map_location=device)
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights)
    print(f"{model_settings['model_name']}'s pretrained weights loaded!\n")

    # Start
    model.eval()
    model.to(device)

    # Tokenizer
    subword_tokenizer = SubwordDatasetTokenizer(model_name='bert-base-uncased')

    # Test Loop
    with torch.no_grad():
        logging.info('Starting validation')
        epoch_loss = 0.
        sum_scores : dict[str, float] = defaultdict(lambda: 0.)
        
        model.eval()
        for i, (src, trg, *rest) in enumerate(test_loader):
            pred_dec_out = None
            trg_dec_out = None
            
            if i % 10 == 0:
                logging.info(f'Parsing {i}/{len(test_loader)}')
                
            src, trg, rest = src.to(device), trg.to(device), [r.to(device) for r in rest]
            
            output, *hiddens = model(src, trg, *rest)
            probs = torch.softmax(output, dim=-1) # caluclate probs from logits   shape: [batch_size, seq_len, vocab_size]
            k = 5
            print(f"{probs.shape=}")
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1) # get the k most probable tokens with the probs and the indices!
            print(f"{top_k_probs.shape=}\n{top_k_indices.shape=}")
            out_seq = torch.multinomial(top_k_probs.view(-1, k), 1) # Sampling from the top k probabilities to get the indices
            print(f"{out_seq.shape=}")
            out_seq_indices = torch.gather(top_k_indices, 1, out_seq.unsqueeze(0)) #gather back the vocabulary indices from top_k_indices
            print(f"{out_seq_indices.shape=}")
            out_seq = out_seq_indices.squeeze(-1)
            print(f"{out_seq.shape=}")
                
            for o, t in zip(output, trg):
                rouges = rouge_scores(o, t)
                for k, v in rouges.items():
                    sum_scores[k] += v.item()
            
            #trg = trg[:, 1:].reshape(-1) # Reshape to [batch_size*trg_len]
            #output = output.reshape(-1, output.shape[-1]) # Reshape to [batch_size*trg_len, vocab_size]
            
            avg_scores = {k: v / len(test_loader.dataset) for k, v in sum_scores.items()}
    
            print(f"Source_shape: {src.shape}\n  Source: {src[0]}")
            print(f"Target_shape: {trg.shape}\n  Target: {trg[0]}")
            print(f"Output Shape: {out_seq.shape}\n Output: {out_seq[0]}\n")

            news_text = subword_tokenizer.tokenizer.decode(src[0].cpu().numpy(), skip_special_tokens=True)
            target_text = subword_tokenizer.tokenizer.decode(trg[0].cpu().numpy(), skip_special_tokens=True)
            generated_text = subword_tokenizer.tokenizer.decode(out_seq[0].cpu().numpy(), skip_special_tokens=False)

            print(f"News Text: {news_text}\n")
            print(f'Target Text: {target_text}\n')
            print(f'Generated Text: {generated_text}\n')

            break
        
# Gets a run that matches `tag` as either a tag (ie 'EnhancedSwin2') or
# a model name (ie 'wspszqbr') and its latest artifact weights.
# Project name hardcoded for simplicity. Sorry Greg!
def get_run(tag: str) -> tuple[dict[str, Any], str]:
    api = wandb.Api()
    runs = api.runs('aabati/NewSum', {'$or': [
        {'tags': tag},
        {'name': tag},
    ]})

    if len(runs) == 0:
        raise NameError(f'No run found with either tag or name "{tag}"')
    
    if len(runs) > 1:
        logging.warning(f'WARNING: {len(runs)} runs found with this tag or name! Choosing one of them.')

    run = runs[0]
    artifact = max(run.logged_artifacts(), key = lambda x: x.version)
    artifact_dir = artifact.download()

    return run.config, os.path.join(artifact_dir, 'model.pth')

def main():
    tag = 'transformer_final'
    config, checkpoint = get_run(tag=tag)

    print(checkpoint)
    data_setting = config['data_settings']
    model_setting = config['model_params']
    inference_setting = config['infer']

    inference_setting['checkpoint_folder'] = checkpoint
    
    infer(data_setting, model_setting, inference_setting)

if __name__ == '__main__':
    main()
