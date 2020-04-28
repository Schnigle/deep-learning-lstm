import torch
import torch.nn.functional as F
import torch.nn as nn

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.lstm2o = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.lstm2o(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
