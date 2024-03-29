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
        # input shape: [batch_size]
        # hidden, cell shapes: [num_layers, batch_size, dec_hid_dim]
        # encoder_outputs shape: [batch_size, src_len, enc_hid_dim]

        input = input.unsqueeze(-1)
        # input shape: [batch_size, trg_len=1]

        embedded = self.dropout(self.embedding(input))
        # embedded shape: [batch_size, trg_len=1, dec_hid_dim]

        query = hidden[-1,:,:].unsqueeze(0).permute(1, 0, 2) # Pass the last hidden layer as query for attention
        # query shape: [batch_size, 1 (last layer), dec_hid_dim]

        context, att_weights = self.attention(query, encoder_outputs)
        # context shape: [batch_size, trg_len=1, dec_hid_dim]
        # att_weights shape: [batch_size, trg_len=1, src_len]
        context = self.layer_norm(context)
        # Combine embedded input word and encoder outputs
        rnn_input = torch.cat((embedded, context), dim=2)
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
        #print(f"\nSrc shape: {src.shape}")
        #print(f"Trg shape: {trg.shape}\n")

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

        for t in range(1, trg_len):
            output, hidden, cell, att_weights = self.decoder(input, hidden, cell, encoder_outputs)
            #print(f"\nDecoder output shape: {output.shape}")
            #print(f"Decoder output: {output}\n")
            outputs[:, t, :] = output.squeeze(1)
            attentions[:, t, :] = att_weights.squeeze(1)

            teacher_force = random.random() < self.teacher_forcing_ratio

            #print(f"Output shape: {output.shape}")
            #print(f"Output: {output}")

            # Top-k sampling
            probs = torch.softmax(output, dim=-1)
            k = 10
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            out_seq = torch.multinomial(top_k_probs.view(-1, k), 1).view(-1, output.shape[1]) # Sampling from the top k probabilities to get the indices
            out_seq = torch.gather(top_k_indices, 2, out_seq.unsqueeze(-1)).squeeze(-1) # Map back the indices to vocabulary index
            print(f"Out seq: {out_seq.shape}\n{out_seq}")
            input = (trg[:, t] if teacher_force else out_seq).detach()
            """_, topi = output.squeeze(1).topk(10)
            input = (trg[:, t] if teacher_force else topi.squeeze(-1)).detach()
            #input = topi.squeeze(-1).detach()"""

            self.teacher_forcing_ratio -= (0.5 / 1000)  # Decrease by 0.0005 each step
            self.teacher_forcing_ratio = max(self.teacher_forcing_ratio, 0)  # Ensure it doesn't go below 0

            #print(f"Shape: {out_seq.type(torch.LongTensor).shape}   Dec Input: {out_seq.type(torch.LongTensor)}")
        #print(attentions)
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
        self.src_position_embedding = nn.Embedding(max_length, emb_size)
        self.trg_word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.trg_position_embedding = nn.Embedding(max_length, emb_size)

        self.emb_size = emb_size
        self.device = device
        self.transformer = nn.Transformer(d_model=emb_size, nhead=heads, 
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=emb_size*forward_expansion,
                                          dropout=dropout)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src == self.pad_idx)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape
        # input shape: [batch_size, seq_len]

        embed_src = self.src_word_embedding(src)
        embed_trg = self.trg_word_embedding(trg)
        # embed shape: [batch_size, seq_len, emd_dim]

        #src_positions = (torch.arange(0, src_seq_length).unsqueeze(0).expand(N, src_seq_length).to(self.device))
        #trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(0).expand(N, trg_seq_length).to(self.device))
        trg_positions = get_positional_encoding(trg_seq_length, self.emb_size, self.device)
        src_positions = get_positional_encoding(src_seq_length, self.emb_size, self.device)
        # src_positions, trg_positions shape: [1, seq_len, emb_dim]

        #embed_src = self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        #embed_trg = self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)

        embed_src += src_positions
        embed_trg += trg_positions
        # embed_src, embed_trg shape: [batch_size, seq_len, emb_dim]

        src_padding_mask = self.make_src_mask(src) # shape: [batch_size, src_len]
        trg_mask = torch.triu(torch.ones((trg_seq_length, trg_seq_length), device=self.device), diagonal=1).bool()
        # trg_mask shape: [trg_len, trg_len]

        # Reshape for nn.Transformer
        embed_src = embed_src.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]
        embed_trg = embed_trg.permute(1,0,2) # shape [seq_len, batch_size, emb_dim]

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        out = self.fc_out(out)

        return out
