import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_config, collate_fn
from models import Encoder, Decoder, Seq2Seq
from dataset import NewsDataset

torch.manual_seed(42)
#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_settings, model_settings, train_settings):
    # Dataset
    train_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='validation', vocabulary_file=data_settings['vocabulary_path'])
    train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model
    INPUT_DIM = len(train_dataset.vocabulary)
    OUTPUT_DIM = len(train_dataset.vocabulary)
    print(f"\nVocabulary size: {INPUT_DIM}\n")
    encoder = Encoder(INPUT_DIM, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['dropout'])
    decoder = Decoder(OUTPUT_DIM, model_settings['decoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['dropout'])
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'])

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocabulary[data_settings['special_tokens'][0]])

    # Logger
    # TBD

    # Train loop
    train_loop(model, train_loader, criterion, optimizer, model_settings, train_settings)

def train_loop(model, train_loader, criterion, optimizer, model_settings, train_settings, clip=1):
    min_loss = float('inf')
    for epoch in range(train_settings['epochs']):
        model.train()
        epoch_loss = 0

        for i, (src, trg) in enumerate(train_loader):
            src, trg = src.transpose(0, 1).to(device), trg.transpose(0, 1).to(device)

            optimizer.zero_grad()
            output = model(src, trg)

            # trg shape: [trg_len, batch_size]
            # output shape: [trg_len, batch_size, output_dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch: {epoch+1:02} | Train Loss: {avg_loss:.3f}')

        # Save checkpoint if improvement
        if avg_loss < min_loss:
            print(f'Loss decreased ({min_loss:.4f} --> {avg_loss:.4f}). Saving model ...')
            torch.save(model.state_dict(), f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_ckt.pth")
            min_loss = avg_loss

def main():
    config = load_config()

    data_setting = config['data_settings']
    model_setting = config['seq2seq_params']
    train_setting = config['train']

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    
    train(data_setting, model_setting, train_setting)
    #validate(data_setting, model_setting, train_setting)

if __name__ == '__main__':
    main()
