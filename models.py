import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F
from torch.nn import Parameter

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2, dropout=0.2):
        """
        :param input_dim: Size of the input.
        :param emb_dim: Dimensionality of the embedding layer.
        :param enc_hid_dim: Dimensionality of the encoder's hidden state.
        :param dec_hid_dim: Dimensionality of the decoder's hidden state.
        :param num_layers: Number of LSTM blocks. 
        :param dropout: Dropout rate.
        """
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        :param src: Input tensor containing a batch of sequences.
        """
        # src shape: [batch_size, src_len]

        embedded = self.dropout(self.embedding(src))
        # embedded shape: [batch_size, src_len, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs shape: [batch_size, src_len, enc_hid_dim]
        # hidden, cell shapes: [num_layers, batch_size, enc_hid_dim]

        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2):
        """
        :param output_dim: Size of the output vocabulary.
        :param emb_dim: Dimensionality of the embedding layer.
        :param enc_hid_dim: Dimensionality of the encoder's hidden state.
        :param dec_hid_dim: Dimensionality of the decoder's hidden state.
        :param dropout: Dropout rate.
        """
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(output_dim, dec_hid_dim)
        self.rnn = nn.LSTM(dec_hid_dim, dec_hid_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)

        self.output_dim = output_dim

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        :param input: Input tensor containing a batch of tokens.
        :param hidden: Last hidden state.
        :param cell: Last cell state.
        :param encoder_outputs: All encoder outputs.
        """
        #print(f"DEC INPUT SHAPE\ninput: {input.shape}\nhidden: {hidden.shape}\ncell:{cell.shape}\nenc_out: {encoder_outputs.shape}")
        # input shape: [batch_size]
        # hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]
        # encoder_outputs shape: [batch_size, src_len, enc_hid_dim]

        input = input.unsqueeze(-1)
        # input shape: [batch_size, trg_len=1]

        embedded = self.embedding(input)
        # embedded shape: [batch_size, trg_len=1, dec_hid_dim]

        #print(f"INP: {input.shape}\nEmb {embedded.shape}")

        # Combine embedded input word and encoder outputs
        rnn_input = F.relu(embedded)
        # rnn_input shape: [batch_size, trg_len=1, dec_hid_dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output shape: [batch_size, trg_len=1, dec_hid_dim]
        # hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]

        prediction = self.fc_out(output)
        # prediction shape: [batch_size, trg_len=1, output_dim==voc_dim]

        #return prediction, hidden, cell
        return prediction, hidden, cell  # Return the last layer states
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_output_lenght=30):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_output_lenght = max_output_lenght        

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim
        #print(f"\nSrc shape: {src.shape}")
        #print(f"Trg shape: {trg.shape}\n")

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device) # Tensor to store decoder outputs
        out_seq = torch.zeros(batch_size, trg_len).to(self.device) # Tensor to store the output sequence 

        encoder_outputs, hidden, cell = self.encoder(src) # Encode the source sequence

        hidden = hidden[-1,:,:].unsqueeze(0) # Get the last hidden state of the encoder and unsqueeze to [1, batch, hid_dim]
        cell = cell[-1,:,:].unsqueeze(0) # Get the last cell state of the encoder and unsqueeze to [1, batch, hid_dim]

        # Create a [num_layers, batch_size, hid_dim] -> We pass the context to all the "blocks" of the LSTM in the decoder 
        hidden = torch.cat((hidden, hidden, hidden), dim=0)
        cell = torch.cat((cell, cell, cell), dim=0)

        input = trg[:, 0] # First input to the decoder is the <sos> tokens

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            #print(f"\nDecoder output shape: {output.shape}")
            #print(f"Decoder output: {output}\n")
            outputs[:, t, :] = output.squeeze(1)

            #print(f"Output shape: {output.shape}")
            #print(f"Output: {output}")

            # Top-1 sampling
            #top1 = output.argmax(1) # Greedy decoding
            #input = top1

            # Top-k sampling
            _, topi = output.squeeze(1).topk(1)
            input = topi.squeeze(-1).detach()

            #outputs = torch.cat(outputs, dim=0)
            #outputs = F.log_softmax(outputs, dim=-1)

            # Appent to output sequence
            out_seq[:, t] = input

            #print(f"Shape: {out_seq.type(torch.LongTensor).shape}   Dec Input: {out_seq.type(torch.LongTensor)}")

        return outputs, out_seq.type(torch.LongTensor)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        # scores shape: [batch_size, trg_len=1, src_len]

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttDecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2, dropout=0.2):
        """
        :param output_dim: Size of the output vocabulary.
        :param emb_dim: Dimensionality of the embedding layer.
        :param enc_hid_dim: Dimensionality of the encoder's hidden state.
        :param dec_hid_dim: Dimensionality of the decoder's hidden state.
        :param num_layers: Number of blocks in the LSTM layer. 
        :param dropout: Dropout rate.
        """
        super(AttDecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(output_dim, dec_hid_dim)
        self.attention = BahdanauAttention(dec_hid_dim)
        self.rnn = nn.LSTM(2*dec_hid_dim, dec_hid_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dec_hid_dim)

        self.output_dim = output_dim

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        :param input: Input tensor containing a batch of tokens.
        :param hidden: Last hidden state.
        :param cell: Last cell state.
        :param encoder_outputs: All encoder outputs.
        """
        #print(f"DEC INPUT SHAPE\ninput: {input.shape}\nhidden: {hidden.shape}\ncell:{cell.shape}\nenc_out: {encoder_outputs.shape}")
        # input shape: [batch_size, trg_len=1]
        # hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]
        # encoder_outputs shape: [batch_size, src_len, enc_hid_dim]

        embedded = self.dropout(self.embedding(input))
        # embedded shape: [batch_size, trg_len=1, dec_hid_dim]

        query = hidden[-1,:,:].unsqueeze(0).permute(1, 0, 2) # Pass the last hidden layer as query for attention
        # query shape: [batch_size, 1 (last layer), dec_hid_dim]

        context, att_weights = self.attention(query, encoder_outputs)
        # context shape: [batch_size, trg_len=1, dec_hid_dim]
        # att_weights shape: [batch_size, trg_len=1, src_len]
        context = self.layer_norm(context)
        
        rnn_input = torch.cat((embedded, context), dim=2)  # Combine embedded input word and encoder outputs
        # rnn_input shape: [batch_size, trg_len=1, dec_hid_dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output shape: [batch_size, trg_len=1, dec_hid_dim]
        # hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]

        prediction = self.fc_out(output)
        # prediction shape: [batch_size, trg_len=1, output_dim==voc_dim]

        return prediction, hidden, cell, att_weights  # Return the last layer states

class AttSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_output_lenght=30):
        super(AttSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_output_lenght = max_output_lenght   
        self.teacher_forcing_ratio = 0.5  # Initial teacher forcing ratio     

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        _, src_len = src.shape
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device) # Tensor to store decoder outputs
        out_seq = torch.zeros(batch_size, trg_len).to(self.device) # Tensor to store the output sequence
        attentions = torch.zeros(batch_size, trg_len, src_len).to(self.device) # Tensor to store attention matrices

        encoder_outputs, hidden, cell = self.encoder(src) # Encode the source sequence

        hidden = hidden[-1,:,:].unsqueeze(0) # Get the last hidden state of the encoder and unsqueeze to [1, batch, hid_dim]
        cell = cell[-1,:,:].unsqueeze(0) # Get the last cell state of the encoder and unsqueeze to [1, batch, hid_dim]

        # Create a [num_layers, batch_size, hid_dim] -> We pass the context to all the "blocks" of the LSTM in the decoder 
        hidden = torch.cat((hidden, hidden, hidden), dim=0)
        cell = torch.cat((cell, cell, cell), dim=0)

        input = trg[:, 0] # First input to the decoder is the <sos> tokens
        out_seq[:, 0] = input
        input = input.unsqueeze(-1)

        for t in range(1, trg_len):
            output, hidden, cell, att_weights = self.decoder(input, hidden, cell, encoder_outputs)
            # output shape: [batch_size, trg_len, voc_dim]
            outputs[:, t, :] = output.squeeze(1)
            attentions[:, t, :] = att_weights.squeeze(1)

            teacher_force = random.random() < self.teacher_forcing_ratio

            # Top-k sampling
            probs = torch.softmax(output, dim=-1)
            k = 10
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            out = torch.multinomial(top_k_probs.view(-1, k), 1).view(-1, output.shape[1]) # Sampling from the top k probabilities to get the indices
            out = torch.gather(top_k_indices, 2, out.unsqueeze(-1)).squeeze(-1) # Map back the indices to vocabulary index
            out_seq[:,t] = out.squeeze(1)

            input = (trg[:, t].unsqueeze(-1) if teacher_force else out.squeeze(0)).detach()
            
            self.teacher_forcing_ratio -= (0.5 / 1000)  # Decrease by 0.0005 each step
            self.teacher_forcing_ratio = max(self.teacher_forcing_ratio, 0)  # Ensure it doesn't go below 0

        return outputs, out_seq.type(torch.LongTensor), attentions

class SelfAttention(nn.Module):
    def __init__(self, input_size, out_size):
        super(SelfAttention, self).__init__()
        self.dk_size = out_size
        self.query_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.key_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.value_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.softmax = nn.Softmax(dim=-1)  # Specify the dimension

    def forward(self, input_vector):
        query_out = F.relu(self.query_linear(input_vector))
        key_out = F.relu(self.key_linear(input_vector))
        value_out = F.relu(self.value_linear(input_vector))
        out_q_k = torch.bmm(query_out, key_out.transpose(1, 2)) / math.sqrt(self.dk_size)
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, value_out)
        return out_combine

def get_positional_encoding(seq_len, emb_size, device):
    """Generate positional encodings for a given sequence length and embedding size."""
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size)).to(device)

    positional_encoding = torch.zeros(seq_len, emb_size).to(device)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)

    return positional_encoding.unsqueeze(0)

class Transformer(nn.Module):
    def __init__(self, vocab_size, pad_idx, 
                 emb_size=256, num_layers=6, forward_expansion=4, heads=8, max_length=2500, dropout=0.1, device="cuda"):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.trg_word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        self.emb_size = emb_size
        self.device = device
        self.transformer = nn.Transformer(d_model=emb_size, nhead=heads, 
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=emb_size*forward_expansion,
                                          dropout=dropout)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.pad_idx = pad_idx

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool) # No mask for the source

        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, trg):
        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape
        # input shape: [batch_size, seq_len]

        embed_src = self.src_word_embedding(src)
        embed_trg = self.trg_word_embedding(trg)
        # embed shape: [batch_size, seq_len, emd_dim]

        trg_positions = get_positional_encoding(trg_seq_length, self.emb_size, self.device)
        src_positions = get_positional_encoding(src_seq_length, self.emb_size, self.device)
        # src_positions, trg_positions shape: [1, seq_len, emb_dim]

        # Add positional encoding
        embed_src += src_positions
        embed_trg += trg_positions
        # embed_src, embed_trg shape: [batch_size, seq_len, emb_dim]

        # Create masks
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_mask(src, trg)
        # src_mask, trg_mask shape: [seq_len, seq_len]  ---  src_padding_mask, trg_padding_mask shape: [batch_size, seq_len]
        #print(f"Src Mask: {src_mask.shape}\n{src_mask}\nSrc padding mask:{src_padding_mask.shape}\n{src_padding_mask}\n\nTrg mask: {trg_mask.shape}\n{trg_mask}\nTrg padding mask: {trg_padding_mask.shape}\n{trg_padding_mask}")

        # Reshape for nn.Transformer
        embed_src = embed_src.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]
        embed_trg = embed_trg.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]

        decoder_out = self.transformer(embed_src, embed_trg, src_mask=src_mask, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask)
        # decoder_out shape: [seq_len, batch_size, emb_dim]

        out = self.fc_out(decoder_out)
        # out shape [seq_len, batch_size, vocab_size]
        
        return out, decoder_out, embed_trg
    
class TransformerV2(nn.Module):
    def __init__(self, vocab_size, pad_idx, 
                 emb_size=256, num_layers=6, forward_expansion=4, heads=8, max_length=2500, dropout=0.1, device="cuda"):
        super(TransformerV2, self).__init__()
        self.src_word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.trg_word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        
        self.tf_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.idf_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        self.emb_size = emb_size
        self.device = device
        self.transformer = nn.Transformer(d_model=emb_size, nhead=heads, 
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=emb_size*forward_expansion,
                                          dropout=dropout)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.pad_idx = pad_idx

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool) # No mask for the source

        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, trg, tf_src, tf_trg, idf_src, idf_trg):
        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape
        # input shape: [batch_size, seq_len]

        embed_src = self.src_word_embedding(src)
        embed_trg = self.trg_word_embedding(trg)
        # embed shape: [batch_size, seq_len, emd_dim]

        # Embed tf and idf metrics
        embed_tf_src = self.tf_embedding(tf_src)
        embed_tf_trg = self.tf_embedding(tf_trg)
        embed_idf_src = self.idf_embedding(idf_src)
        embed_idf_trg = self.idf_embedding(idf_trg)

        trg_positions = get_positional_encoding(trg_seq_length, self.emb_size, self.device)
        src_positions = get_positional_encoding(src_seq_length, self.emb_size, self.device)
        # src_positions, trg_positions shape: [1, seq_len, emb_dim]

        # Add positional encoding
        embed_src += src_positions
        embed_trg += trg_positions
        # Add tf metrics
        embed_src += embed_tf_src
        embed_trg += embed_tf_trg
        # Add idf metrics
        embed_src += embed_idf_src
        embed_trg += embed_idf_trg
        # embed_src, embed_trg shape: [batch_size, seq_len, emb_dim]

        # Create masks
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_mask(src, trg)
        # src_mask, trg_mask shape: [seq_len, seq_len]  ---  src_padding_mask, trg_padding_mask shape: [batch_size, seq_len]
        #print(f"Src Mask: {src_mask.shape}\n{src_mask}\nSrc padding mask:{src_padding_mask.shape}\n{src_padding_mask}\n\nTrg mask: {trg_mask.shape}\n{trg_mask}\nTrg padding mask: {trg_padding_mask.shape}\n{trg_padding_mask}")

        # Reshape for nn.Transformer
        embed_src = embed_src.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]
        embed_trg = embed_trg.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]

        decoder_out = self.transformer(embed_src, embed_trg, src_mask=src_mask, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask)
        # decoder_out shape: [seq_len, batch_size, emb_dim]

        out = self.fc_out(decoder_out)
        # out shape [seq_len, batch_size, vocab_size]

        return out, decoder_out, embed_trg
