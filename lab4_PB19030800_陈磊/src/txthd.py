from torchtext.vocab import GloVe,Vectors
from torchtext import data
import os
import pdb
import pickle
# .txt-->string list+score list
print("begin")
path=["../../aclImdb/train/neg","../../aclImdb/train/pos","../../aclImdb/test/neg","../../aclImdb/test/pos"]

txts=[[],[],[],[]]
score=[[],[],[],[]]

for i in range(4):
# 遍历指定目录下所有文件
    files=os.listdir(path[i])
    for file in files:
        file_path=path[i]+'/'+file
        # print(file_path)
        with open(file_path,'r',encoding='utf-8') as f:
            data=f.read()
            txts[i].append(data)
            score[i].append(int(file[-5]))

with open("../pkl/score.pkl","wb") as f:
    pickle.dump(score,f)
    print(type(score))
# pdb.set_trace()

print(txts[0][0])
print(len(txts[0]),len(txts[1]),len(txts[2]),len(txts[3]))
print(len(score[0]))
print(score[0][:5])


total_txts=txts[0]+txts[1]+txts[2]+txts[3]

##### 字符串-->字符列表
import re

txt_list=[]
str_list=[]
for line in total_txts:
#可能还有
    latstr=re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ",line)
    word_list=latstr.split(' ')
    txt_list.append(word_list)
    str_list.append(' '.join(word_list))
# print(txt_list[:5])

with open('../pkl/train_txt_list.pkl','wb') as f:
    pickle.dump(str_list[:25000],f)
with open('../pkl/test_txt_list.pkl','wb') as f:
    pickle.dump(str_list[25000:],f)

# pdb.set_trace()

##### 字符串修剪
import numpy as np

len_list=[]
for word_list in txt_list:
    len_list.append(len(word_list)) 
len_list=np.array(len_list)
max_len=np.mean(len_list)+2*np.std(len_list)
print(int(max_len))
# pdb.set_trace()

pruned_txt_list=[]
for word_list in txt_list:
    if(len(word_list)<max_len):
        word_list=[' ']*(int(max_len)-len(word_list))+word_list
        pruned_txt_list.append(word_list)
    elif len(word_list)>max_len:
        pruned_txt_list.append(word_list[:int(max_len)])

max_len2=257
bert_txt_list=[]
for word_list in txt_list:
    if(len(word_list)<max_len2):
        bert_txt_list.append(word_list)
    elif len(word_list)>max_len2:
        bert_txt_list.append(word_list[:int(max_len2)])
pdb.set_trace()

print(len(pruned_txt_list))
print(pruned_txt_list[0])
print(pruned_txt_list[1])


# with open("./pkl/train_pru_txt_list.pkl","wb") as f:
#     pickle.dump(pruned_txt_list,f)
#     print(type(pruned_txt_list))
with open("../pkl/train_pru_txt_list.pkl","wb") as f:
    pickle.dump(pruned_txt_list[:25000],f)
    print(type(pruned_txt_list))
with open("../pkl/test_pru_txt_list.pkl","wb") as f:
    pickle.dump(pruned_txt_list[25000:],f)
    print(type(pruned_txt_list))
with open("../pkl/train_bert_txt_list.pkl","wb") as f:
    pickle.dump(bert_txt_list[:25000],f)
    print(type(bert_txt_list))
with open("../pkl/test_bert_txt_list.pkl","wb") as f:
    pickle.dump(bert_txt_list[25000:],f)
    print(type(bert_txt_list))
# pdb.set_trace()

##### 字符列表-->字符向量

TEXT = data.Field(sequential=True,use_vocab=True)
vectors=Vectors(name="../../src/glove.6B.100d.txt")
TEXT.build_vocab(pruned_txt_list, vectors=vectors)

print(len(TEXT.vocab.stoi))
print(TEXT.vocab.vectors[:5])

##### 导出
with open("../pkl/stoi.pkl","wb") as f:
    pickle.dump(TEXT.vocab.stoi,f)
    print(type(TEXT.vocab.stoi))
with open("../pkl/itos.pkl","wb") as f:
    pickle.dump(TEXT.vocab.itos,f)
    print(type(TEXT.vocab.itos))
with open("../pkl/vectors.pkl","wb") as f:
    pickle.dump(TEXT.vocab.vectors,f)
    print(type(TEXT.vocab.vectors))
##### 字符转为向量示例
# import pickle

# with open('./pkl/stoi.pkl','rb') as f:
#     sztoi=pickle.load(f)
# with open('./pkl/vectors.pkl','rb') as f:
#     iztov=pickle.load(f)

# index=sztoi['hello']
# print(index)
# print(iztov[index])