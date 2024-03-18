import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.2):
        """
        :param input_dim: Size of the input vocabulary.
        :param emb_dim: Dimensionality of the embedding layer.
        :param enc_hid_dim: Dimensionality of the encoder's hidden state.
        :param dec_hid_dim: Dimensionality of the decoder's hidden state.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        :param src: Input tensor containing a batch of sequences.
        """
        # src shape: [src_len, batch_size]

        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src_len, batch_size, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs shape: [src_len, batch_size, enc_hid_dim * num_directions]
        # hidden, cell shapes: [num_layers * num_directions, batch_size, enc_hid_dim]

        # Concatenate the final forward and backward hidden, cell states and pass through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))
        # hidden, cell shape: [batch_size, enc_hid_dim]

        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.2):
        """
        :param output_dim: Size of the output vocabulary.
        :param emb_dim: Dimensionality of the embedding layer.
        :param enc_hid_dim: Dimensionality of the encoder's hidden state.
        :param dec_hid_dim: Dimensionality of the decoder's hidden state.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        #self.rnn = nn.LSTM(emb_dim + enc_hid_dim * 2, dec_hid_dim)
        #self.fc_out = nn.Linear(enc_hid_dim * 2 + dec_hid_dim + emb_dim, output_dim)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        :param input: Input tensor containing a batch of tokens.
        :param hidden: Last hidden state of the encoder.
        :param cell: Last cell state of the encoder.
        :param encoder_outputs: All encoder outputs.
        """
        # input shape: [batch_size]
        # hidden, cell shapes: [batch_size, dec_hid_dim]
        # encoder_outputs shape: [src_len, batch_size, enc_hid_dim * num_directions]

        input = input.unsqueeze(0)
        # input shape: [1, batch_size]

        embedded = self.dropout(self.embedding(input))
        # embedded shape: [1, batch_size, emb_dim]

        # Combine embedded input word and encoder outputs
        rnn_input = torch.cat((embedded, hidden.unsqueeze(0)), dim=2)

        #print("Decoder shapes:")
        #print(f"Input shape: {rnn_input.shape}")
        #print(f"Hidden: {hidden.shape}")
        #print(f"Cell: {cell.shape}")

        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # output shape: [seq_len, batch_size, dec_hid_dim * num_directions]
        # hidden, cell shapes: [num_layers * num_directions, batch_size, dec_hid_dim]

        #print(f"Output shape: {output.shape}")
        #print(f"Hidden: {hidden.shape}")
        #print(f"Cell: {cell.shape}")

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat((output, hidden, embedded), dim=1))
        # prediction shape: [batch_size, output_dim]

        return prediction, hidden.squeeze(0), cell.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_output_lenght=30):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_output_lenght = max_output_lenght

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        #src = src.transpose(0, 1)  # Transpose to [src_len, batch_size]
        #trg = trg.transpose(0, 1)  # Transpose to [trg_len, batch_size]
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim
        #print(f"\nSrc shape: {src.shape}")
        #print(f"Trg shape: {trg.shape}\n")

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) # Tensor to store decoder outputs

        encoder_outputs, hidden, cell = self.encoder(src) # Encode the source sequence

        input = trg[0, :] # First input to the decoder is the <sos> tokens

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            #print(f"\nDecoder output shape: {output.shape}")
            #print(f"Decoder output: {output}\n")
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio # Decide whether to use teacher forcing
            top1 = output.argmax(1) # Greedy decoding
            input = trg[t] if teacher_force else top1

        return outputs
