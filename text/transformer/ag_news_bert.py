import transformers
from transformers import *
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

class AGNEWSDataset(Dataset):
    def __init__(self, sentences, labels):
        self.dataset = sentences
        self.labels = labels


    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)


from datasets import load_dataset
dataset = load_dataset("ag_news")


train_set = AGNEWSDataset(dataset['train']['text'], dataset['train']['label'])
train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
train_eval_loader = DataLoader(train_set, shuffle=False, batch_size=16)

dev_set = AGNEWSDataset(dataset['test']['text'], dataset['test']['label'])
dev_eval_loader = DataLoader(dev_set, shuffle=False, batch_size=16)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:3")


bert = BertModel.from_pretrained('bert-base-uncased')
import torch.nn as nn
class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        self.bert = bert
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, **text):
        with torch.no_grad():
            embedded = self.bert(**text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        output = self.out(hidden)
        return output
    
    def feature(self,text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return hidden
    
    def prediction(self, hidden):
        output = self.out(hidden)
        #output = [batch size, out dim]
        return output


HIDDEN_DIM = 256
OUTPUT_DIM = 4  # 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                        HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

model = model.to(device)

for name, param in model.named_parameters():   
    print(name)                
    if name.startswith('bert'):
        param.requires_grad = False

optimizer = transformers.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
criterion = nn.CrossEntropyLoss()


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # #round predictions to the closest integer
    # rounded_preds = torch.round(torch.sigmoid(preds))
    # correct = (rounded_preds == y).float() #convert into float for division 
    # acc = correct.sum() / len(correct)
    prediction = preds.argmax(axis=1)
    correct = (prediction==y).float()
    acc = correct.sum()/len(correct)
    return acc

def train():
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for text, label in tqdm(train_loader):
        inputs = tokenizer(text, return_tensors='pt', padding=True)
        
        for w in inputs:
                inputs[w] = inputs[w].to(device)

        label = label.to(dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        
        predictions = model(**inputs)
        
        loss = criterion(predictions, label)
        
        acc = binary_accuracy(predictions, label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


def test(loader, data_set_name):
    with torch.no_grad():
        model.eval()
        epoch_acc = 0
        for text, label in tqdm(loader):
            inputs = tokenizer(text, return_tensors='pt', padding=True)
        
            for w in inputs:
                    inputs[w] = inputs[w].to(device)

            label = label.to(dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            predictions = model(**inputs)
            
            acc = binary_accuracy(predictions, label)
        
            epoch_acc += acc.item()

        print(data_set_name, epoch_acc / len(loader))

def split(loader, data_set_name):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for x, y in tqdm(loader):
            inputs = tokenizer(x, return_tensors='pt', padding=True)
            labels = y
            for w in inputs:
                inputs[w] = inputs[w].to(device)
            labels = labels.to(device)
            out = model(**inputs)
            outputs = model.bert(**inputs)
            pooled_output = outputs[1]
            logits = model.classifier(pooled_output)

            o = model.feature()(**inputs)
            p = o[1]
            l = model.prediction()(p)
            
            pred = torch.argmax(out[0], dim=-1, keepdim=False)
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()
            correct += np.sum(pred == labels)
            total += pred.shape[0]
        print(data_set_name, correct / total)


for epoch in range(20):
    train()
    test(train_eval_loader, "train")
    test(dev_eval_loader, 'dev')
    scheduler.step()
