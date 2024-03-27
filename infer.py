import torch
from torch.utils.data import DataLoader
from utils import load_config, collate_fn, plot_attention
from models import EncoderLSTM, DecoderLSTM, Seq2Seq, AttDecoderLSTM, AttSeq2Seq
from dataset import NewsDataset
from dataset_tokenizer import SubwordDatasetTokenizer

torch.manual_seed(42)
#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(data_settings, model_settings, inference_settings):
    # Dataset
    test_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='test', vocabulary_file=data_settings['vocabulary_path'])
    test_loader = DataLoader(test_dataset, batch_size=inference_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model
    INPUT_DIM = len(test_dataset.vocabulary)
    OUTPUT_DIM = len(test_dataset.vocabulary)
    encoder = EncoderLSTM(INPUT_DIM, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'], model_settings['dropout'])
    decoder = AttDecoderLSTM(OUTPUT_DIM, model_settings['decoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'], model_settings['dropout'])
    model = AttSeq2Seq(encoder, decoder, device)

    # Loading checkpoint
    if inference_settings['load_checkpoint']:
        ckpt = torch.load(f"{inference_settings['checkpoint_folder']}/{model_settings['model_name']}_ckt.pth", map_location=device)
        model_weights = ckpt['model_weights']
        model.load_state_dict(model_weights)
        print("Model's pretrained weights loaded!\n")

    # Start
    model.eval()
    model.to(device)

    # Tokenizer
    subword_tokenizer = SubwordDatasetTokenizer(model_name='bert-base-uncased')

    for i, (src, trg) in enumerate(test_loader):
        src, trg = src.to(device), trg.to(device)

        output, out_seq, attentions = model(src, trg)

        print(f"Source_shape: {src.shape}\n  Source: {src[0]}")
        print(f"Target_shape: {trg.shape}\n  Target: {trg[0]}")
        print(f"Output Shape: {output.shape}\n Output: {output[0]}")

        news_text = subword_tokenizer.tokenizer.decode(src[0].cpu().numpy(), skip_special_tokens=True)
        target_text = subword_tokenizer.tokenizer.decode(trg[0].cpu().numpy(), skip_special_tokens=True)

        generated_text = subword_tokenizer.tokenizer.decode(out_seq[0].cpu().numpy(), skip_special_tokens=True)
        
        print(f"News Text: {news_text}\n")
        print(f'Target Text: {target_text}\n')
        print(f'Generated Text: {generated_text}\n')

        plot_attention(input_tokens=src[0], output_tokens=out_seq[0], attentions=attentions[0])
        break

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
