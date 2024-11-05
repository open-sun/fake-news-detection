from transformers import BertTokenizer,BertForSequenceClassification,BertModel
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import numpy as np
def dataset(texts, labels=None):
    input_ids = []
    attention_masks = []
    for text in texts:
        textencode = token.encode_plus(text, add_special_tokens=True,truncation=True, max_length=50, padding='max_length',  return_tensors='pt')
        input_ids.append(textencode['input_ids'])
        attention_masks.append(textencode['attention_mask'])
    input_ids=torch.cat(input_ids)
    attention_masks=torch.cat(attention_masks)
    if labels is not None:
        labels = labels
        return TensorDataset(input_ids, attention_masks, labels)
    else:
        return TensorDataset(input_ids, attention_masks)
if __name__ == "__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path ='D:/桌面/fakenews2/bert-base-chinese'
    token = BertTokenizer.from_pretrained(model_path)
    train=pd.read_csv('newtrain2.csv')
    test=pd.read_csv('newtest2.csv')
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)
    t = pd.DataFrame(train.astype(str))
    train['data']=t["Title"]#+t["Ofiicial Account Name"]+t['News Url']
    #print(train['data'][0])
    t = pd.DataFrame(test.astype(str))
    test["data"] = t["Title"]#+t["Ofiicial Account Name"]#+t['News Url']
    #train = train.dropna(subset=['data', 'label']) 
    #test = test.dropna(subset=['data']) 
    labels = train['label'].values
    labels=torch.tensor(labels)
    print(labels)
    print(train['data'])
    texts=train['data'].tolist()
    tests=test['data'].tolist()
    traindataset=dataset(texts,labels)
    testdataset=dataset(tests)
    print(testdataset[0])
    print(traindataset[1])
    batchsize=16
    traindataloader=DataLoader(traindataset,batch_size=batchsize,shuffle=True)
    testdataloader=DataLoader(testdataset,batch_size=batchsize,shuffle=False)
    epochs=2
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    totalstep=0
    model.train()
    for i in range(epochs):
        j=0
        totalloss=0.0
        for batch in traindataloader:
            input_ids,attention_masks,labels=batch
            input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs=model(input_ids,attention_mask=attention_masks, labels=labels)
            loss=outputs[0]
            loss.backward()
            optimizer.step()
            totalstep=totalstep+1
            totalloss += loss.item()
            j=j+1
            if totalstep%10==0:
                print(f'我是{i}轮第{j}个,我的损失率是{totalloss/j}')
    model.eval()
    predictions = []
    with torch.no_grad():
        num=0
        for batch in testdataloader:
            input_ids,attention_masks=batch
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)
            predictions.extend(predicted_labels.numpy())
            num=num+1
            if num%10==0:
                print(f'我已经预测了{num}个了')
    print("我好了")
    submission = pd.DataFrame({'id': test['id'], 'label': predictions})
    submission.to_csv('submit_bert10.csv', index=False)
