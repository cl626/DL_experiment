import pickle,pdb

import torch
import torch.utils.data as data
import numpy as np
import torch.nn.functional as F
from torch import optim,nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel
import matplotlib.pyplot as plt

with open('../pkl/stoi.pkl','rb') as f:
    sztoi=pickle.load(f)
with open('../pkl/vectors.pkl','rb') as f:
    iztov=pickle.load(f)
with open('../pkl/test_pru_txt_list.pkl','rb') as f:
    test_pru_txt_list=pickle.load(f)


test_num=1000
test_score_label=[[1,0]]*12500+[[0,1]]*12500

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class realtestDataset(data.Dataset):
    def __init__(self):
        self.x=test_pru_txt_list[5000:5000+test_num]+test_pru_txt_list[15000:15000+test_num]
        self.y=torch.tensor(test_score_label[5000:5000+test_num]+test_score_label[15000:15000+test_num]).float()

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

rtest_data=realtestDataset()
rtest_loader=data.DataLoader(rtest_data,shuffle=True,batch_size=64,drop_last=True)

class LSTM_last_ele(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers,batch_size,output_size):
        super(LSTM_last_ele, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=True, batch_first=False)
        self.layer = nn.Sequential(nn.Linear(2*self.hidden_size, 32), 
                        nn.Tanh(),nn.Linear(32,output_size),
                        nn.Softmax(dim=-1))
        self.hidd=self.init_hidd()
        self.cell=self.init_cell()
        
    def init_hidd(self):
        return torch.zeros(self.n_layers*2,self.batch_size,self.hidden_size).to(device)

    def init_cell(self):
        return torch.zeros(self.n_layers*2,self.batch_size,self.hidden_size).to(device)


    def forward(self,input):
        output, (self.hidd,self.cell) = self.lstm(input, (self.hidd, self.cell))    # 592,64,100->592,64,128
        output = self.layer(output[-1, :, :])           #592,64,128->64,128,只取最后一个元素,64,128->164,10
        return output


n_layer=1
vector_size=100
hidden_size=128
batch_size=64
output_size=2
fc_hidden_size=32
net=LSTM_last_ele(vector_size,hidden_size,n_layer,batch_size,output_size)

net.load_state_dict(torch.load('lr_adma_l2_2e-4.params'))
net.eval().to(device)
correct=0.0
total=0.0    

for input,label in rtest_loader:
    intoken=[]
    for bat in input:
        token_bat=[]
        for word in bat:
            token_bat.append(iztov[sztoi[word]].tolist())
        intoken.append(token_bat)

    intoken=torch.tensor(intoken)
    intoken,label=intoken.to(device),label.to(device)   #自动写成元组
    # print(f'device={intoken.device,label.device}')

    net.hidd=net.init_hidd().to(device)
    net.cell=net.init_cell().to(device)

    out=net(intoken)
    # print(out,label)
    # print(f'{out.shape,label.shape}')
    _,pred_index=torch.max(out,dim=1)        
    _,real_index=torch.max(label,dim=1)
    correct+=(pred_index==real_index).sum().item()
    total+=label.size(0)
print(f'total={total}')
print('Accuracy of the BiLSTM on the real test dataset: %.3f %%'%(100*correct/total))
    
# test bert
with open('../pkl/test_bert_txt_list.pkl','rb') as f:
    test_txt_list=pickle.load(f)

bert_rtest_texts = test_txt_list[:test_num]+test_txt_list[12500:12500+test_num] # 训练数据
bert_rscore_label=[[1,0]]*test_num+[[0,1]]*test_num

class BertDataSet(data.Dataset):
    def __init__(self,texts,label,tokenizer,max_len):
        self.tokenizer = tokenizer
        self.comment_text = texts
        self.targets = label
        self.max_len=max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text =' '.join(self.comment_text[index]).lower()    #小写
        # print(comment_text)
        
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN=512
bert_rtest_data=BertDataSet(bert_rtest_texts,bert_rscore_label,tokenizer,MAX_LEN)
bert_rtest_loader=data.DataLoader(bert_rtest_data,shuffle=True,batch_size=4)

class Bert(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Bert,self).__init__()

        self.bert=BertModel.from_pretrained('bert-base-uncased')

        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size),
        )

    def forward(self,input_ids,token_type_ids,attention_mask):
        out=self.bert(input_ids,token_type_ids,attention_mask)  
        # print(f'pos1:{out.last_hidden_state.size()}')   #(batch_size,seq_len,hidden_size)=(8,200,768)
        out=out.last_hidden_state[:,-1,:]
        # print(f'pos2:{out.shape}')
        out=self.classifier(out)        #(batch_size,output_size)=(8,10)
        # print(f'pos3:{out.shape}')
        return out

input_size=768
hidden_size=128
output_size=2
model=Bert(input_size,hidden_size,output_size)
model.load_state_dict(torch.load('bert.params'))
model.eval().to(device)

running_loss=0.0
correct=0.0
num=0.0    
for _, data in enumerate(bert_rtest_loader):
    ids = data['ids'].to(device, dtype = torch.long)
    mask = data['mask'].to(device, dtype = torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
    label = data['targets'].to(device)
    out = model(ids, mask, token_type_ids)

    _,pred_index=torch.max(out,dim=1)        
    _,real_index=torch.max(label,dim=1)
    correct+=(pred_index==real_index).sum().item()                
    num+=label.size(0)
    # print('finish 1 batch')
print('Accuracy of Bert on testDataset is %.3f %%'%(correct/num*100))
