import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0.):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)