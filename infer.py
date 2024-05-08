import torch
from torch.utils.data import DataLoader
from utils import load_config, collate_fn, collate_fn_v2
from models import EncoderLSTM, DecoderLSTM, Seq2Seq, AttDecoderLSTM, AttSeq2Seq, Transformer, TransformerV2
from dataset import NewsDataset
from dataset_tokenizer import SubwordDatasetTokenizer

torch.manual_seed(42)
#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(data_settings, model_settings, inference_settings):
    # Dataset
    test_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='test', vocabulary_file=data_settings['vocabulary_path'], version=model_settings['version'])
    if model_settings['version'] == '1':
        test_loader = DataLoader(test_dataset, batch_size=inference_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
    elif model_settings['version'] == '2':
        # Change collate function to v2
        test_loader = DataLoader(test_dataset, batch_size=inference_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_v2)

    # Model
    INPUT_DIM = len(test_dataset.vocabulary)
    OUTPUT_DIM = len(test_dataset.vocabulary)
    print(f"\nVocabulary size: {INPUT_DIM}\n")
    PAD_IDX = test_dataset.vocabulary[data_settings['special_tokens'][0]]
    if model_settings['model_name'] == 'seq2seq':
        encoder = EncoderLSTM(INPUT_DIM, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'], model_settings['dropout'])
        #decoder = DecoderLSTM(OUTPUT_DIM, model_settings['decoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'])
        #model = Seq2Seq(encoder, decoder, device).to(device)
        decoder = AttDecoderLSTM(OUTPUT_DIM, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'], model_settings['dropout'])
        model = AttSeq2Seq(encoder, decoder, device).to(device)
    elif model_settings['model_name'] == 'transformer':
        if model_settings['version'] == '1':
            model = Transformer(
                vocab_size=OUTPUT_DIM, 
                pad_idx=PAD_IDX, 
                emb_size=model_settings['encoder_embedding_dim'], 
                num_layers=model_settings['num_layers'], 
                forward_expansion=4,
                heads=8,
                dropout=model_settings['dropout'],
                device=device
            ).to(device)
        elif model_settings['version'] == '2':
            model = TransformerV2(
                vocab_size=OUTPUT_DIM, 
                pad_idx=PAD_IDX, 
                emb_size=model_settings['encoder_embedding_dim'], 
                num_layers=model_settings['num_layers'], 
                forward_expansion=4,
                heads=8,
                dropout=model_settings['dropout'],
                device=device
            ).to(device)
        else:
            raise ValueError("Version not supported!")
    else:
        raise ValueError("Selected model not available. Please choose between 'seq2seq' and 'transformer")

    # Loading checkpoint
    if inference_settings['load_checkpoint']:
        ckpt = torch.load(f"{inference_settings['checkpoint_folder']}/{model_settings['model_name']}_v{model_settings['version']}_ckt.pth", map_location=device)
        model_weights = ckpt['model_weights']
        model.load_state_dict(model_weights)
        print(f"{model_settings['model_name']}'s pretrained weights loaded!\n")

    # Start
    model.eval()
    model.to(device)

    # Tokenizer
    subword_tokenizer = SubwordDatasetTokenizer(model_name='bert-base-uncased')

    with torch.no_grad():
        if model_settings['version'] == '1':
            for i, (src, trg) in enumerate(test_loader):
                src, trg = src.to(device), trg.to(device)
                if model_settings['model_name'] == 'seq2seq':
                    output, out_seq, attentions = model(src, trg)
                elif model_settings['model_name'] == 'transformer':
                    trg_input = trg[:, :-1] #remove last token of trg
                    output = model(src, trg_input)
                    probs = torch.softmax(output, dim=-1) # caluclate probs from logits
                    k = 5
                    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1) # get the k most probable tokens with the probs and the indices!
                    out_seq = torch.multinomial(top_k_probs.view(-1, k), 1) # Sampling from the top k probabilities to get the indices
                    out_seq_indices = torch.gather(top_k_indices, 2, out_seq.unsqueeze(-1)) #gather back the vocabulary indices from top_k_indices
                    out_seq = out_seq_indices.squeeze(-1).permute(1, 0) # reshape to [batch_size, trg_len]
                else:
                    raise ValueError('Model not valid!')
                break # End loop after 1 sample
        elif model_settings['version'] == '2':
            for i, (src, trg, tf_src, tf_trg, idf_src, idf_trg) in enumerate(test_loader):
                src, trg = src.to(device), trg.to(device)
                tf_src, tf_trg, idf_src, idf_trg = tf_src.to(device), tf_trg.to(device), idf_src.to(device), idf_trg.to(device)
                if model_settings['model_name'] == 'transformer':
                    trg_input = trg[:, :-1] #remove last token of trg
                    tf_trg_input = tf_trg[:, :-1]
                    idf_trg_input = idf_trg[:, :-1]
                    output = model(src, trg_input, tf_src, tf_trg_input, idf_src, idf_trg_input)
                    probs = torch.softmax(output, dim=-1) # caluclate probs from logits
                    k = 5
                    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1) # get the k most probable tokens with the probs and the indices!
                    out_seq = torch.multinomial(top_k_probs.view(-1, k), 1) # Sampling from the top k probabilities to get the indices
                    out_seq_indices = torch.gather(top_k_indices, 2, out_seq.unsqueeze(-1)) #gather back the vocabulary indices from top_k_indices
                    out_seq = out_seq_indices.squeeze(-1).permute(1, 0) # reshape to [batch_size, trg_len]
                else:
                    raise ValueError("Model not valid! Only 'transformer' is supported for version '2'.")

        print(f"Source_shape: {src.shape}\n  Source: {src[0]}")
        print(f"Target_shape: {trg.shape}\n  Target: {trg[0]}")
        print(f"Output Shape: {out_seq.shape}\n Output: {out_seq[0]}\n")

        news_text = subword_tokenizer.tokenizer.decode(src[0].cpu().numpy(), skip_special_tokens=True)
        target_text = subword_tokenizer.tokenizer.decode(trg[0].cpu().numpy(), skip_special_tokens=True)
        generated_text = subword_tokenizer.tokenizer.decode(out_seq[0].cpu().numpy(), skip_special_tokens=False)

        print(f"News Text: {news_text}\n")
        print(f'Target Text: {target_text}\n')
        print(f'Generated Text: {generated_text}\n')
        

def main():
    config = load_config()

    data_setting = config['data_settings']
    model_setting = config['seq2seq_params']
    inference_setting = config['infer']

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    
    infer(data_setting, model_setting, inference_setting)

if __name__ == '__main__':
    main()
