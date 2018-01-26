import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class s2l_Net(nn.Module):
    """TODO: NETWORK DESCRIPTION"""

    def __init__(self, h_size=128, v_size=10000, embed_d=300, mlp_d=256, num_classes=3, lstm_layers=1):
        super(s2l_Net, self).__init__()
        self.num_layers = lstm_layers
        self.hidden_size = h_size
        self.embedding = nn.Embedding(v_size, embed_d)
        self.lstm = nn.LSTM(embed_d, h_size, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.mlp = nn.Linear(2 * h_size * 2, num_classes)

        # Set static embedding vectors
        #self.embedding.weight.requires_grad = False
       
    def display(self):
        for param in self.parameters():
            print(param.data.size())
    
    def init_hidden(self):
        pass
    
    def forward(self, s1, s2, lengths):

        batch_size = s1.size()[0]

        # TODO Pack inputs 
        # Wants us to sort the input sentences into decreasing lengths... but training pairs will no longer match
        #s1_packed = rnn_utils.pack_padded_sequence(s1, lengths[0], batch_first=True)
        #s2_packed = rnn_utils.pack_padded_sequence(s2, lengths[1], batch_first=True)
        
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, s1.size(0), self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, s1.size(0), self.hidden_size)).cuda()
        
        embeds_1 = self.embedding(s1)
        embeds_2 = self.embedding(s2)
        _, (h_1_last, _) = self.lstm(embeds_1, (h0, c0)) 
        _, (h_2_last, _) = self.lstm(embeds_2, (h0, c0))
        
        #unpack
        #h_1_last, _ = rnn_utils.pad_packed_sequence(h_1_last, batch_first=True)
        #h_2_last, _ = rnn_utils.pad_packed_sequence(h_2_last, batch_first=True)
        
        h_1_last = h_1_last.transpose(0,1).contiguous().view(batch_size, -1)
        h_2_last = h_2_last.transpose(0,1).contiguous().view(batch_size, -1)
        concat = torch.cat( (h_1_last, h_2_last), dim=1) 
        scores = self.mlp(concat)
        
        return scores


if __name__ == '__main__':
    net = s2l_Net()
    net.display()
