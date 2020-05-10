import torch
import torch.nn.functional as F
import torch.nn as nn

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, vocab_size, embedding_dim):
        super(RNN_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers)
        self.lstm2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_cell):
        embed = self.embedding(input)
        (hidden, cell) = hidden_cell
        # print(embed.size())
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))
        output = self.lstm2o(output)
        return output, (hidden, cell)

    def initHidden(self, batch_size, device):
        # n_layers x batch_size x hidden_size
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device), torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))
