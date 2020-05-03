import torch
import torch.nn.functional as F
import torch.nn as nn

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.lstm2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_cell):
        (hidden, cell) = hidden_cell
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.lstm2o(output)
        return output, (hidden, cell)

    def initHidden(self, batch_size, device):
        # n_layers x batch_size x hidden_size
        return (torch.zeros(1, batch_size, self.hidden_size).to(device), torch.zeros(1, batch_size, self.hidden_size).to(device))
