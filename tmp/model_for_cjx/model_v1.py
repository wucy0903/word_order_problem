import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import requests
import numpy as np
import sklearn
import monpa
import random
import pickle

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
        final_res = F.softmax(linear_res)
        print ('final res :::: ' , final_res)
        return final_res
def train_v1(dict_tag2idx, dict_idx2tag, training_data_tag_list, training_target_sentence_list):
        model_v1 = Bi_GRU_Model(total_tag=input_size_v1, 
                                embeding_size = embedding_size_v1,
                                gru_hidden=gru_hidden_size_v1,
                                output_size=output_size_v1,
                                gru_layer = gru_layer_v1)
        optimizer_v1 = torch.optim.SGD(model_v1.parameters(), lr=learning_rate_v1)
        loss_function_v1 = nn.CrossEntropyLoss()
        for epoch in range(epoch_v1):
            sub_epoch = 0
            for seq_tag_idx in range(len(training_data_tag_list)):
                sub_epoch += 1
                input_data = torch.LongTensor([dict_tag2idx[t] for t in training_data_tag_list[seq_tag_idx]])
                target_data = torch.LongTensor(training_target_sentence_list[seq_tag_idx]).unsqueeze(0)
                out = model_v1(input_data)
                loss = loss_function_v1(out, target_data)
                optimizer_v1.zero_grad()
                loss.backward()
                optimizer_v1.step()
                if sub_epoch % 100 == 0:
                    print ('Epoch '+  str(epoch) + ' '+ str(sub_epoch)+'/' + str(len(training_data_tag_list)) + '  Loss : ' + str(loss))
                torch.save(model_v1, '../pickle/model_v1.pt')


if __name__ == '__main__':
    training_data_sentence_list = list()
    training_data_tag_list = list()
    training_target_sentence_list = list()
    dict_idx2tag = dict()
    dict_tag2idx = dict()
    epoch_v1 = 10
    learning_rate_v1 = 0.0001
    input_size_v1 = len(dict_idx2tag.keys())
    gru_hidden_size_v1 = 8
    embedding_size_v1 = 32
    output_size_v1 = 2
    gru_layer_v1 = 1


    # Read both the postive data and negative data
    with open ('../CJX_Train_Test_Data/positive_data.txt', 'r', encoding = 'utf-8') as rf:
        for pos_s in rf.readlines():
            training_data_sentence_list.append(pos_s.strip('\n').strip(',').strip('，').strip('《').strip('》').strip('【').strip('】').strip('、').strip('、').strip('-'))
            training_target_sentence_list.append(1)
    with open ('../CJX_Train_Test_Data/positive_data.txt', 'r', encoding = 'utf-8') as rf:
        for pos_s in rf.readlines():
            training_data_sentence_list.append(pos_s.strip('\n').strip(',').strip('，').strip('《').strip('》').strip('【').strip('】').strip('、').strip('-'))
            training_target_sentence_list.append(0)

    print ('training data size : ',len(training_data_sentence_list))
    print ('training data target size : ',len(training_target_sentence_list))
    tmp_cnt = 0
    # Change the words to tags, because we need to use the sequential tags for training.
    for sentence in training_data_sentence_list:
        tmp_cnt += 1
        #print (tmp_cnt)
        tmp = monpa.pseg(sentence)
        tmp_list = list()
        for item in tmp:
            tmp_list.append(item[1])
        training_data_tag_list.append(tmp_list)

    print (len(training_data_tag_list))

    # Read the tags of the monpa
    with open ('./monpa_tag.txt' , 'r' , encoding = 'utf-8') as rf:
        cnt = 0
        for tag in rf.readlines():
            dict_idx2tag[cnt] = tag.strip('\n')
            dict_tag2idx[tag.strip('\n')] = cnt
            cnt += 1  

    print ('The dictionary of idxs 2 tags : ', dict_idx2tag)
    print ('The dictionary of tags 2 idxs : ' , dict_tag2idx)

    shuffle_zip = list(zip(training_data_tag_list, training_target_sentence_list))
    random.shuffle(shuffle_zip)
    training_data_tag_list[:], training_target_sentence_list[:] = zip(*shuffle_zip)
    print (training_data_tag_list, training_target_sentence_list)
    
    train_v1(dict_tag2idx, dict_idx2tag, training_data_tag_list,training_target_sentence_list)
    
            
    

