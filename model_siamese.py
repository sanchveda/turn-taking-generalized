import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.utils as utils
import numpy as np 
import pdb 

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.autograd import Variable

class MLP_Turns (nn.Module):
    def __init__(self, D_inp, D_hid, drop=0.3, n_classes=2):
        super(MLP_Turns, self).__init__()

        self.fc1 = nn.Linear(D_inp, D_hid)
        self.fc2 = nn.Linear(D_hid, D_hid//2)
        self.fc3 = nn.Linear(D_hid//2, n_classes)

        self.norm1= nn.BatchNorm1d (D_hid)
        self.norm2= nn.BatchNorm1d (D_hid//2)

        self.dropout = drop 
        self.drop =nn.Dropout(self.dropout) 
    def forward (self, x):
        "x = (batch, dim)"

    
        x= self.norm1(F.relu(self.fc1 (x)))
        x= self.drop (x)
        x= self.norm2(F.relu(self.fc2 (x)))
        x= self.fc3 (x)
    
        x= F.log_softmax (x, dim=1)
        
        return x


class MLP_general (nn.Module):
    def __init__(self, D_a, D_v, D_hid,  word_dict,embed_size=128,  drop= 0.3, output_frames=15, modal ='all'):
        super(MLP_general, self).__init__()

        #self.rnn = nn.LSTM(input_size=D_inp, hidden_size=D_hid, num_layers=2, dropout=0.3, bidirectional=False, batch_first=True)
        self.vocab_size = len (word_dict)
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size,self.embed_size)
        self.modal = modal 
        if self.modal == 'a':
            self.rnn = nn.GRU(input_size=2*(D_a), hidden_size=D_hid, num_layers=2, dropout=drop, bidirectional=False, batch_first=True)
        elif self.modal =='v':
            self.rnn = nn.GRU(input_size=2*(D_v), hidden_size=D_hid, num_layers=2, dropout=drop, bidirectional=False, batch_first=True)
        elif self.modal == 'av':
            self.rnn = nn.GRU(input_size=2*(D_a+D_v), hidden_size=D_hid, num_layers=2, dropout=drop, bidirectional=False, batch_first=True)
        elif self.modal == 'at':
            self.rnn = nn.GRU(input_size=2*(D_a + self.embed_size), hidden_size=D_hid, num_layers=2, dropout=drop, bidirectional=False, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=2*(D_a+ D_v + self.embed_size), hidden_size=D_hid, num_layers=2, dropout=drop, bidirectional=False, batch_first=True)
        self.fc = nn.Linear (D_hid, output_frames)

    def _get_embedding(self,token, props):
        for idx, num in enumerate(x1_token):
            for j_idx, num_2 in enumerate(num):
                pdb.set_trace()
        return
    def forward (self, audio=None, video=None, token=None, props=None, mask=None):

        x1_audio, x2_audio = audio
        x1_video, x2_video = video
        x1_token, x2_token = token
        x1_props, x2_props = props 

        x1_token = self.embedding (x1_token)
        x2_token = self.embedding (x2_token)
        mask_= mask.unsqueeze(-1).repeat (1,1, self.embed_size)
        x1_token = torch.sum( (x1_token  * x1_props.unsqueeze(-1)), axis=-2) / mask_ 
        x2_token = torch.sum( (x2_token  * x2_props.unsqueeze(-1)), axis=-2) / mask_ 
        
        if self.modal == 'a':
            x1 = x1_audio
            x2=  x2_audio
        elif self.modal =='v':
            x1= x1_video
            x2= x2_video
        elif self.modal == 'av':
            x1 = torch.cat( [x1_audio, x1_video], dim=-1)
            x2 = torch.cat( [x2_audio, x2_video], dim=-1)
        elif self.modal == 'at':
            x1 = torch.cat( [x1_audio, x1_token], dim=-1)
            x2 = torch.cat( [x2_audio, x2_token], dim=-1)
        else:
            x1 = torch.cat( [x1_audio, x1_video, x1_token], dim=-1)
            x2 = torch.cat( [x2_audio, x2_video, x2_token], dim=-1)
            
        #Joining both speaker and listener
        x = torch.cat ([x1, x2], dim=-1)

        x_out, x_hid = self.rnn(x)
        
        x_out = torch.sigmoid(self.fc(x_out))
        
        #self._get_embedding  (token, props) 


        return x_out
class MLP_Model (nn.Module):
	def __init__(self, D_inp, D_hid, drop=0.3):
		super(MLP_Model, self).__init__()

		self.fc1 = nn.Linear(D_inp, D_hid)
		self.fc2 = nn.Linear(D_hid, D_hid//2)
		#self.fc3 = nn.Linear(D_hid//2, n_classes)

		self.norm1= nn.BatchNorm1d (D_hid)
		self.norm2= nn.BatchNorm1d (D_hid//2)

		self.dropout = drop 
		self.drop =nn.Dropout(self.dropout) 
	def forward (self, x):
		"x = (batch, dim)"

	
		x= self.norm1(F.relu(self.fc1 (x)))
		x= self.drop (x)
		x= self.norm2(F.relu(self.fc2 (x)))
		#x= self.fc3 (x)
	   
		#x= F.log_softmax (x, dim=1)
		
		return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class MaskedLoss (nn.Module):

    def __init__(self, weight=None):
        super(MaskedLoss, self).__init__()
        self.weight=weight

        self.loss  = nn.NLLLoss(weight=self.weight, reduction='sum')


    def forward(self,pred, target, mask=None):
        
        if mask is not None: 
            mask_ = mask.view(-1,1)
           
            if type(self.weight) == type(None):
                if torch.sum(mask_) != 0:
                    loss = self.loss (pred * mask_,target) / torch.sum(mask_)
                else:
                    loss = self.loss (pred * mask_,target) / 0.001

            else:
                
                loss = self.loss (pred*mask_, target) \
                        /torch.sum(self.weight[target]*mask_.squeeze())
        else:

            if type(self.weight) == type(None):
                loss = self.loss (pred ,target) 
            else:
                loss = self.loss (pred, target) \
                        /torch.sum(self.weight[target])
                    
        return loss 