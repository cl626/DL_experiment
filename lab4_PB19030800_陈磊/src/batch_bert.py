import pickle,pdb

import torch
import torch.utils.data as data
import numpy as np
import torch.nn.functional as F
from torch import optim,nn
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification,BertModel

with open('../pkl/train_bert_txt_list.pkl','rb') as f:
    train_txt_list=pickle.load(f)
with open('../pkl/test_bert_txt_list.pkl','rb') as f:
    test_txt_list=pickle.load(f)


# 加载预训练好的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10) # 3 表示类别数量

# 定义训练数据和标签
sample_num=2000
train_texts = train_txt_list[:sample_num]+train_txt_list[12500:12500+sample_num] # 训练数据
test_texts = test_txt_list[:sample_num]+test_txt_list[12500:12500+sample_num] # 训练数据
score_label=[[1,0]]*sample_num+[[0,1]]*sample_num

class BertDataSet(data.Dataset):
    def __init__(self,texts,tokenizer,max_len):
        self.tokenizer = tokenizer
        self.comment_text = texts
        self.targets = score_label
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
    
MAX_LEN=512
train_data=BertDataSet(train_texts,tokenizer,MAX_LEN)
test_data=BertDataSet(test_texts,tokenizer,MAX_LEN)

train_loader=data.DataLoader(train_data,shuffle=True,batch_size=8)
test_loader=data.DataLoader(test_data,shuffle=True,batch_size=4)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Net,self).__init__()

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
model=Net(input_size,hidden_size,output_size)

# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,eps=1e-8)
criterion=nn.CrossEntropyLoss() #交叉熵损失函数

train_loss_list=[]
train_accu_list=[]
epoch_num=15
valid_accu_list=[]

for epoch in range(epoch_num):
    model.train().to(device)
    if(epoch%5==4):
        optimizer.param_groups[0]['lr']*=0.5
    running_loss=0.0
    correct=0.0
    num=0.0
    for _, data in enumerate(train_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        label = data['targets'].to(device)
        
        out = model(ids, mask, token_type_ids)
        running_loss+=criterion(out,label).item()
        loss=criterion(out,label)
        loss.backward()
        optimizer.step()
        
        _,pred_index=torch.max(out,dim=1)        
        _,real_index=torch.max(label,dim=1)
        correct+=(pred_index==real_index).sum().item()
        # print(correct)
        num+=label.size(0)
        # print('finish 1 batch')

    print('epoch:%d;    loss:%.3f;  accu:%.3f'%(epoch+1,running_loss/num,correct/num))
    train_loss_list.append(running_loss/num)
    train_accu_list.append(correct/num)

    if(epoch%5==4):
        model.eval().to(device)
        running_loss=0.0
        correct=0.0
        num=0.0    
        with torch.no_grad():   #切断上下文求导计算
            for _, data in enumerate(test_loader):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                label = data['targets'].to(device)
                out = model(ids, mask, token_type_ids)

                running_loss+=criterion(out,label).item()
                _,pred_index=torch.max(out,dim=1)        
                _,real_index=torch.max(label,dim=1)
                correct+=(pred_index==real_index).sum().item()                
                num+=label.size(0)
                # print('finish 1 batch')
            print('test   loss:%.3f;  accu:%.3f'%(running_loss/num,correct/num))

plt.suptitle("bert training loss and result_2cls(2000samples)")
ax1=plt.subplot(121)
ax1.plot(train_loss_list,label='loss')
ax1.set_xlabel('epoch_num')
ax1.set_ylabel('loss')
ax1.legend()

ax2=plt.subplot(122)
ax2.plot(train_accu_list,label='accu')
ax2.set_xlabel('epoch_num')
ax2.set_ylabel('accu')
ax2.legend()

plt.savefig('bert.png')

torch.save(model.state_dict(),'bert.params')