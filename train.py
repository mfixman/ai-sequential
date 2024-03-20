import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_config, collate_fn
from models import Encoder, Decoder, Seq2Seq
from dataset import NewsDataset
from logger import Logger
import os

torch.manual_seed(42)
os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_settings, model_settings, train_settings, logger):
    # Dataset
    train_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='train', vocabulary_file=data_settings['vocabulary_path'])
    val_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='validation', vocabulary_file=data_settings['vocabulary_path'])
    train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model
    INPUT_DIM = len(train_dataset.vocabulary)
    OUTPUT_DIM = len(train_dataset.vocabulary)
    print(f"\nVocabulary size: {INPUT_DIM}\n")
    encoder = Encoder(INPUT_DIM, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'], model_settings['dropout'])
    decoder = Decoder(OUTPUT_DIM, model_settings['decoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['dropout'])
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'])

    # Loading checkpoint
    epoch_start = 0
    if train_settings['load_checkpoint']:
        ckpt = torch.load(f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_ckt.pth", map_location=device)
        model_weights = ckpt['model_weights']
        model.load_state_dict(model_weights)
        optimizer_state = ckpt['optimizer_state']
        optimizer.load_state_dict(optimizer_state)
        epoch_start = ckpt['epoch']
        print("Model's pretrained weights loaded!")

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocabulary[data_settings['special_tokens'][0]])

    # Train loop
    min_loss = float('inf')
    for epoch in range(epoch_start, train_settings['epochs']):
        train_loss = train_loop(model, train_loader, criterion, optimizer)
        val_loss = validation_loop(model, val_loader, criterion)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
        logger.log({'train_loss': train_loss, 'validation_loss': val_loss})

        # Save checkpoint if improvement
        if val_loss < min_loss:
            print(f'Loss decreased ({min_loss:.4f} --> {val_loss:.4f}). Saving model ...')
            ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
            torch.save(ckpt, f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_ckt.pth")
            min_loss = val_loss

def train_loop(model, train_loader, criterion, optimizer, clip=1):
        model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()
            output = model(src, trg)

            # trg shape: [batch_size, trg_len]
            # output shape: [trg_len, batch_size, voc_size] ??

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

def validation_loop(model, val_loader, criterion, clip=1):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, (src, trg) in enumerate(val_loader):
                src, trg = src.to(device), trg.to(device)

                output = model(src, trg)

                # trg shape: [batch_size, trg_len]
                # output shape: [trg_len, batch_size, voc_size] ??

                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)

                loss = criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(val_loader)
            return avg_loss

def main():
    config = load_config()

    data_setting = config['data_settings']
    model_setting = config['seq2seq_params']
    train_setting = config['train']

    wandb_logger = Logger(
        f"News_Summarization",
        project='news_sum')
    logger = wandb_logger.get_logger()

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    
    train(data_setting, model_setting, train_setting, logger)
    #validate(data_setting, model_setting, train_setting)

if __name__ == '__main__':
    main()
