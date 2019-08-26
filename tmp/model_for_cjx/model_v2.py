import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import requests
import numpy as np
import sklearn

class Convolution_1D_Model (nn.Module):
    def __init__(self, total_tag, embeding_size, filter_num_1, filter_size_1, filter_num_2, filter_size_2 , output_size,max_word):
        super(Convolution_1D_Model, self).__init__()
        self.total_tag = total_tag
        self.max_word = max_word
        self.embeding_size = embeding_size
        self.filter_num_1 = filter_num_1
        self.filter_size_1 = filter_size_1
        self.filter_num_2 = filter_num_2
        self.filter_size_2 = filter_size_2
        self.output_size = output_size
        self.Embeding = nn.Embedding(self.total_tag, self.embeding_size)
        self.Conv1d_layer1 = nn.Sequential(nn.Conv1d(in_channels=self.embeding_size, 
                                                            out_channels=self.filter_num_1,
                                                            kernel_size=self.filter_size_1),
                                                         #nn.BatchNorm1d (self.filter_size_1),
                                                         nn.ReLU())
        
        self.Conv1d_layer2 = nn.Sequential(nn.Conv1d(in_channels=self.filter_num_1, 
                                                            out_channels=self.filter_num_2,
                                                            kernel_size=self.filter_size_2),
                                                         #nn.BatchNorm1d (self.filter_size_2),
                                                         nn.ReLU())
        # batch size is  set to 1 , output_size is set to 2
        L = (self.max_word- self.filter_size_1+ 1)- self.filter_size_2+ 1
        self.linear = nn.Linear( self.filter_num_2 * L, self.output_size )
        
    def forward(self, input_data):  
        embed_res = self.Embeding (input_data)
        if len(input_data) < self.max_word:
            for c in range (self.max_word - len(input_data)):
                embed_res = torch.cat((embed_res, torch.zeros(1, self.embeding_size)),dim = 0 )
        embed_res = embed_res.unsqueeze(0)
        print (' Embedding :::: ' , embed_res.size())
        # let the input dimension be the (N , Cin , L) and then the output with the dimension (N, Cout , Lout)
        conv_res1 = self.Conv1d_layer1 (embed_res.permute(0,2,1))
        print (' conv_res1 :::: ' , conv_res1.size())
        conv_res2 = self.Conv1d_layer2 (conv_res1)
        print ('conv_res2 :::: ' , conv_res2.size())
        linear_res = self.linear(conv_res2.view(-1))
        final_res = F.sigmoid(linear_res)
        print ("final_res :::: " , final_res)
        return final_res
    
if __name__ == '__main__':
        model_v2 = Convolution_1D_Model(total_tag=16, embeding_size=8, filter_num_1=5, filter_size_1=2, filter_num_2=5, filter_size_2=2 ,output_size = 7, max_word=10)
    test_vec = torch.LongTensor([1,2,3,4,5,6,7])
    model_v2(test_vec)
    print ('V2_Model_with_only_convolution1D ::: \n', model_v2)

