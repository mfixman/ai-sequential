import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_config, collate_fn
from models import EncoderLSTM, DecoderLSTM, Seq2Seq, AttDecoderLSTM, AttSeq2Seq, Transformer
from dataset import NewsDataset
from logger import Logger
import os

from transformer import Transformer as CustomTransformer

torch.manual_seed(42)
#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_settings, model_settings, train_settings, logger):
    # Dataset
    train_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='validation', vocabulary_file=data_settings['vocabulary_path'])
    val_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='test', vocabulary_file=data_settings['vocabulary_path'])
    train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model
    INPUT_DIM = len(train_dataset.vocabulary)
    OUTPUT_DIM = len(train_dataset.vocabulary)
    print(f"\nVocabulary size: {INPUT_DIM}\n")
    if model_settings['model_name'] == 'seq2seq':
        encoder = EncoderLSTM(INPUT_DIM, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'], model_settings['dropout'])
        #decoder = DecoderLSTM(OUTPUT_DIM, model_settings['decoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'])
        #model = Seq2Seq(encoder, decoder, device).to(device)
        decoder = AttDecoderLSTM(OUTPUT_DIM, model_settings['encoder_embedding_dim'], model_settings['hidden_dim'], model_settings['hidden_dim'], model_settings['num_layers'], model_settings['dropout'])
        model = AttSeq2Seq(encoder, decoder, device).to(device)
    elif model_settings['model_name'] == 'transformer':
        PAD_IDX = train_dataset.vocabulary[data_settings['special_tokens'][0]]
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
    elif model_settings['model_name'] == 'custom_transformer':
        model = CustomTransformer(embed_dim=512,
                            vocab_size=OUTPUT_DIM, 
                            seq_length=2500,
                            num_layers=model_settings['num_layers'],
                            expansion_factor=4,
                            n_heads=8).to(device)
    else:
        raise ValueError("Selected model not available. Please choose between 'seq2seq' and 'transformer")

    # Parameter initialisation
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

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
        check_initial_loss(model, train_loader, criterion, model_settings)
        train_loss = train_loop(model, train_loader, criterion, optimizer, model_settings)
        val_loss = validation_loop(model, val_loader, criterion, model_settings)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
        logger.log({'train_loss': train_loss, 'validation_loss': val_loss})

        # Save checkpoint if improvement
        if val_loss < min_loss:
            print(f'Loss decreased ({min_loss:.4f} --> {val_loss:.4f}). Saving model ...')
            ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
            torch.save(ckpt, f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_ckt.pth")
            min_loss = val_loss

def check_initial_loss(model, data_loader, criterion, model_settings):
    model.eval()
    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            print(f"Src: {src.shape}\n{src}\n\nTrg: {trg.shape}\n{trg}")
            if model_settings['model_name']=='transformer':
                trg_input = trg[:, :-1]  # Exclude the last token for target input
                print(f"Trg Input: {trg_input.shape}\n{trg_input}")
                output = model(src, trg_input)
                print(f"Output: {output.shape}\n{output}")
                output = output.permute(1,0,2)
                output = output.reshape(-1, output.shape[-1])
                trg = trg[:, 1:].reshape(-1)  # Shift target for loss computation
            elif model_settings['model_name']=='custom_transformer':
                trg_input = trg[:, :-1]  # Exclude the last token for target input
                print(f"Trg Input: {trg_input.shape}\n{trg_input}")
                output = model(src, trg_input)
                print(f"Output: {output.shape}\n{output}")
                output = output.reshape(-1, output.shape[-1])
                trg = trg[:, 1:].reshape(-1)  # Shift target for loss computation
            else:
                break

            loss = criterion(output, trg)
            print(f"Initial loss: {loss.item()}")
            break

def train_loop(model, train_loader, criterion, optimizer, model_settings, clip=1):
        model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)

            #print(f"Src shape: {src.shape}\nSrc: {src}")
            #print(f"Src shape: {src.shape}\nSrc: {src}")

            optimizer.zero_grad()
            
            if model_settings['model_name'] == 'seq2seq':
                output, out_seq, attentions = model(src, trg)
                # trg shape: [batch_size, trg_len]
                #print(f"Output shape: {output.shape}\n Out: {output}")
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)
            elif model_settings['model_name'] == 'transformer':
                trg_input = trg[:, :-1] #remove last token of trg
                output = model(src, trg_input)
                # Reshape output to [batch_size, trg_len, vocab_size]
                output = output.permute(1,0,2)
                #print(f"Out BEF: {output.shape}\n{output}")
                #print(f"Trg BEF: {trg.shape}\n{trg}")
                #output_dim = output.shape[-1]
                #output = output.reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                output = output.reshape(-1, output.shape[-1])
                #trg = trg.reshape(-1)
            elif model_settings['model_name'] == 'custom_transformer':
                out = model(src, trg)
                print(f"Custom Transf:\nOut: {out.shape}\n{out}")
            else:
                 raise ValueError("Model not valid!")
            

            #print(f"Out: {output.shape}\n{output}")
            #print(f"Trg: {trg.shape}\n{trg}")

            loss = criterion(output, trg)
            l1_lambda = 0.00001
            l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
            loss += l1_lambda*l1_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

def validation_loop(model, val_loader, criterion, model_settings):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, (src, trg) in enumerate(val_loader):
                src, trg = src.to(device), trg.to(device)

                if model_settings['model_name'] == 'seq2seq':
                    output, out_seq, attentions = model(src, trg)
                    # trg shape: [batch_size, trg_len]
                    #print(f"Output shape: {output.shape}\n Out: {output}")
                    output_dim = output.shape[-1]
                    output = output.squeeze(1)[1:].view(-1, output_dim)
                    trg = trg[1:].reshape(-1)
                elif model_settings['model_name'] == 'transformer':
                    trg_input = trg[:, :-1] #remove last token of trg
                    output = model(src, trg_input)
                    # Reshape output to [batch_size, trg_len, vocab_size]
                    output = output.permute(1,0,2)
                    #print(f"Out BEF: {output.shape}\n{output}")
                    #print(f"Trg BEF: {trg.shape}\n{trg}")
                    #output_dim = output.shape[-1]
                    #output = output.reshape(-1, output_dim)
                    trg = trg[:, 1:].reshape(-1)
                    output = output.reshape(-1, output.shape[-1])
                    #trg = trg.reshape(-1)
                else:
                    raise ValueError("Model not valid!")

                loss = criterion(output, trg)
                l1_lambda = 0.00001
                l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
                loss += l1_lambda*l1_norm
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(val_loader)
            return avg_loss

def main():
    config = load_config()

    data_setting = config['data_settings']
    model_setting = config['seq2seq_params']
    train_setting = config['train']

    wandb_logger = Logger(
        f"{model_setting['model_name']}_lr={train_setting['learning_rate']}_L1",
        project='NewSum')
    logger = wandb_logger.get_logger()

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()

    print("\n############## TRAINING SETTINGS ##############")
    print(train_setting)
    print()
    
    train(data_setting, model_setting, train_setting, logger)
    #validate(data_setting, model_setting, train_setting)

if __name__ == '__main__':
    main()
