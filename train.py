import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_config, collate_fn, collate_fn_v2
from models import EncoderLSTM, DecoderLSTM, Seq2Seq, AttDecoderLSTM, AttSeq2Seq, Transformer, TransformerV2
from dataset import NewsDataset
from logger import Logger
import os

torch.manual_seed(42)
os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_settings, model_settings, train_settings, logger):
    # Dataset
    train_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='validation', vocabulary_file=data_settings['vocabulary_path'], version=model_settings['version'])
    val_dataset = NewsDataset(data_dir=data_settings['dataset_path'], special_tokens=data_settings['special_tokens'], split_type='validation', vocabulary_file=data_settings['vocabulary_path'], version=model_settings['version'])
    if model_settings['version'] == '1':
        train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
    elif model_settings['version'] == '2':
        # Change collate function to v2
        train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_v2)
        val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_v2)

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
        print("Using Transformer\n")
        PAD_IDX = train_dataset.vocabulary[data_settings['special_tokens'][0]]
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
            raise ValueError("Verion not supported!")
    else:
        raise ValueError("Selected model not available. Please choose between 'seq2seq' and 'transformer")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

    # Loading checkpoint
    epoch_start = 0
    if train_settings['load_checkpoint']:
        ckpt = torch.load(f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_v{model_settings['version']}_ckt.pth", map_location=device)
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
        train_loss = train_loop(model, train_loader, criterion, optimizer, model_settings)
        val_loss = validation_loop(model, val_loader, criterion, model_settings)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
        if logger: logger.log({'train_loss': train_loss, 'validation_loss': val_loss})

        # Save checkpoint if improvement
        if val_loss < min_loss:
            print(f'Loss decreased ({min_loss:.4f} --> {val_loss:.4f}). Saving model ...')
            ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
            torch.save(ckpt, f"{train_settings['checkpoint_folder']}/{model_settings['model_name']}_v{model_settings['version']}_ckt.pth")
            min_loss = val_loss

def train_loop(model, train_loader, criterion, optimizer, model_settings, clip=1):
        model.train()
        epoch_loss = 0

        if model_settings['version'] == '1':
            for i, (src, trg) in enumerate(train_loader):
                src, trg = src.to(device), trg.to(device)
                optimizer.zero_grad()
                if model_settings['model_name'] == 'seq2seq':
                    output, out_seq, attentions = model(src, trg) # trg shape: [batch_size, trg_len]
                    output_dim = output.shape[-1]
                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].reshape(-1)
                elif model_settings['model_name'] == 'transformer':
                    trg_input = trg[:, :-1] #remove last token of trg
                    output = model(src, trg_input) 
                    output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
                    trg = trg[:, 1:].reshape(-1)
                    output = output.reshape(-1, output.shape[-1])

                loss = criterion(output, trg)
                l1_lambda = 0.00001
                l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
                loss += l1_lambda*l1_norm
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                epoch_loss += loss.item()
        elif model_settings['version'] == '2':
            for i, (src, trg, tf_src, tf_trg, idf_src, idf_trg) in enumerate(train_loader):
                src, trg = src.to(device), trg.to(device)
                tf_src, tf_trg, idf_src, idf_trg = tf_src.to(device), tf_trg.to(device), idf_src.to(device), idf_trg.to(device)
                optimizer.zero_grad()
                if model_settings['model_name'] == 'transformer':
                    trg_input = trg[:, :-1] #remove last token of trg
                    tf_trg_input = tf_trg[:, :-1]
                    idf_trg_input = idf_trg[:, :-1]
                    output = model(src, trg_input, tf_src, tf_trg_input, idf_src, idf_trg_input)
                    output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
                    trg = trg[:, 1:].reshape(-1)
                    output = output.reshape(-1, output.shape[-1])
                else:
                    raise ValueError("Model not valid! Only 'transformer' is supported for version '2'.")
                
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
            if model_settings['version'] == '1':
                for i, (src, trg) in enumerate(val_loader):
                    src, trg = src.to(device), trg.to(device)
                    if model_settings['model_name'] == 'seq2seq':
                        output, out_seq, attentions = model(src, trg) # trg shape: [batch_size, trg_len]
                        output_dim = output.shape[-1]
                        output = output[1:].view(-1, output_dim)
                        trg = trg[1:].reshape(-1)
                    elif model_settings['model_name'] == 'transformer':
                        trg_input = trg[:, :-1] #remove last token of trg
                        output = model(src, trg_input) 
                        output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
                        trg = trg[:, 1:].reshape(-1)
                        output = output.reshape(-1, output.shape[-1])

                    loss = criterion(output, trg)
                    l1_lambda = 0.00001
                    l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
                    loss += l1_lambda*l1_norm
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    epoch_loss += loss.item()
            elif model_settings['version'] == '2':
                for i, (src, trg, tf_src, tf_trg, idf_src, idf_trg) in enumerate(val_loader):
                    src, trg = src.to(device), trg.to(device)
                    tf_src, tf_trg, idf_src, idf_trg = tf_src.to(device), tf_trg.to(device), idf_src.to(device), idf_trg.to(device)
                    if model_settings['model_name'] == 'transformer':
                        trg_input = trg[:, :-1] #remove last token of trg
                        tf_trg_input = tf_trg[:, :-1]
                        idf_trg_input = idf_trg[:, :-1]
                        output = model(src, trg_input, tf_src, tf_trg_input, idf_src, idf_trg_input)
                        output = output.permute(1,0,2) # Reshape output to [batch_size, trg_len, vocab_size]
                        trg = trg[:, 1:].reshape(-1)
                        output = output.reshape(-1, output.shape[-1])
                    else:
                        raise ValueError("Model not valid! Only 'transformer' is supported for version '2'.")

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

    if train_setting['log']:
        wandb_logger = Logger(
            f"{model_setting['model_name']}_v{model_setting['version']}_lr={train_setting['learning_rate']}_L1",
            project='NewSum')
        logger = wandb_logger.get_logger()
    else:
        logger = None

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()

    print("\n############## TRAINING SETTINGS ##############")
    print(train_setting)
    print()
    
    train(data_setting, model_setting, train_setting, logger)

if __name__ == '__main__':
    main()
