from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetConfig
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

# load dataset

class SSTDataset(Dataset):
    def __init__(self, sentences, labels):
        self.dataset = sentences
        self.labels = labels

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)


from datasets import load_dataset
dataset = load_dataset("sst2")

device=torch.device("cuda:3")

train_set = SSTDataset(dataset['train']['sentence'], dataset['train']['label'])
train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
train_eval_loader = DataLoader(train_set, shuffle=False, batch_size=16)

dev_set = SSTDataset(dataset['validation']['sentence'], dataset['validation']['label'])
dev_eval_loader = DataLoader(dev_set, shuffle=False, batch_size=16)

# tokenizer

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
xlnet = xlnet.to(device)
# hidden_states

import torch.nn as nn
class XLNETGRU(nn.Module):
    def __init__(self,
                 xlnet,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        self.xlnet = xlnet
        embedding_dim = self.xlnet.config.to_dict()['d_model']
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
            embedded = self.xlnet(**text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        output = self.out(hidden)
        return output
    
    def feature(self,**text):
        with torch.no_grad():
            embedded = self.xlnet(**text).hidden_states
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
OUTPUT_DIM = 2  # 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = XLNETGRU(xlnet,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT)

for name, param in model.named_parameters():
    if name.startswith('xlnet'):
        param.requires_grad = False

# TODO lr
optimizer = torch.optim.AdamW(model.parameters())
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

model = model.to(device)
criterion = criterion.to(device)

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

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for text, label in tqdm(iterator):
        inputs = tokenizer(text, return_tensors="pt", padding=True)

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
    print(f'train, loss:\t{epoch_loss / len(iterator)}, acc:\t{epoch_acc / len(iterator)}')
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test(model, iterator):

    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for text, label in tqdm(iterator):
            inputs = tokenizer(text, return_tensors="pt", padding=True)

            for w in inputs:
                inputs[w] = inputs[w].to(device)
            label = label.to(dtype=torch.long, device=device)

            predictions = model(**inputs)
            
            acc = binary_accuracy(predictions, label)

            epoch_acc += acc.item()
    print(f'test acc:\t{epoch_acc / len(iterator)}')
        
    return epoch_acc / len(iterator)

for epoch in range(2):
    train(model, train_loader, optimizer, criterion)
    test(model, train_eval_loader)
    test(model, dev_eval_loader)
    scheduler.step()