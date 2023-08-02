from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn.functional as F

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        # TODO: model architecture
        self.num_class = num_class
        self.embeddings = embeddings
        
        self.embed = Embedding.from_pretrained(embeddings, freeze=True)
        self.lstm = torch.nn.LSTM(
            input_size=embeddings.size()[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
            )
        if bidirectional == True:
            self.classifier_1 =  torch.nn.Linear(hidden_size*2, hidden_size)
        elif bidirectional == False:
            self.classifier_1 =  torch.nn.Linear(hidden_size, hidden_size)
        
        self.classifer_2 = torch.nn.Linear(hidden_size, num_class)
        #self.layer_norm = torch.nn.LayerNorm([128, 150, 150])
        self.relu = torch.nn.ReLU()
        
        #activation function
        #self.softmax = torch.nn.Softmax(dim=1)
        
        #torch.nn.init.xavier_uniform_(self.classifier.weight)      
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedded = self.embed(batch)
        #padded_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, batch_lengths, batch_first=True)
        #packed sequence
        lstm_out, _ = self.lstm(embedded)
        hidden_state = lstm_out[:,-1,:]
        output = self.classifier_1(hidden_state)
        output = self.classifer_2(output)
        #output = self.layer_norm(output)
        output = self.relu(output)
        return output
        #return F.log_softmax(output, dim=1)
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedded = self.embed(batch)
        lstm_out, _ = self.lstm(embedded)
        output = self.classifier_1(lstm_out)
        #output = self.classifer_2(output)
        output = output.permute(0,2,1)
        #output = self.layer_norm(output)
        output = self.relu(output)
        return output
        raise NotImplementedError
