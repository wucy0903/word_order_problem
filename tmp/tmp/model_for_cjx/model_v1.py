import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import requests
import numpy as np
import sklearn

class Bi_GRU_Model (nn.Module):
    def __init__(self, total_tag, embeding_size, gru_hidden,gru_layer, output_size):
        super(Bi_GRU_Model, self).__init__()
        self.total_tag = total_tag
        self.embeding_size = embeding_size
        self.gru_hidden = gru_hidden
        self.gru_layer = gru_layer
        self.output_size = output_size
        self.Embeding = nn.Embedding(self.total_tag, self.embeding_size)
        self.GRU = nn.GRU(self.embeding_size , self.gru_hidden, self.gru_layer, bidirectional= True)
        
        # batch size is  set to 1 , output_size is set to 2
        self.linear = nn.Linear( self.gru_hidden*2 , self.output_size )
        
    def forward(self, input_data):  
        embed_res = self.Embeding (input_data)
        print ('Embedding :::: ' , embed_res.size())
        # let the input dimension be the (L , N , M) and the get the output with the dimension (L , N , H)
        gru_res, _ = self.GRU(embed_res.unsqueeze(1))
        print ("GRU :::: ", gru_res.size())
        linear_res = self.linear(gru_res[-1])
        final_res = F.sigmoid(linear_res)
        print ('final res :::: ' , final_res)
        return final_res


if __name__ == '__main__':
    model_v1 = Bi_GRU_Model(total_tag=16, embeding_size=8, gru_hidden= 6 , gru_layer=1, output_size=7 )
    test_vec = torch.LongTensor([1,2,3,4,5,6,7])
    model_v1(test_vec)
    print ('V1_Model_with_only_gru ::: \n', model_v1)