import transformers
from transformers import *
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

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


train_set = SSTDataset(dataset['train']['sentence'], dataset['train']['label'])
train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
train_eval_loader = DataLoader(train_set, shuffle=False, batch_size=16)

dev_set = SSTDataset(dataset['validation']['sentence'], dataset['validation']['label'])
dev_eval_loader = DataLoader(dev_set, shuffle=False, batch_size=16)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, hidden_dropout_prob=0.3)
model.cuda(device=0)


def feature(self):
    return self.bert

def prediction(self):
    return self.classifier

from types import MethodType
model.feature = MethodType(feature, model)
model.prediction = MethodType(prediction, model)

for name, param in model.named_parameters():    
    print(name)            
    if name.startswith('bert'):
        param.requires_grad = False

optimizer = transformers.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

def train():
    loss = 0.0
    print('epoch', epoch + 1)
    model.train()
    for x, y in tqdm(train_loader):
        inputs = tokenizer(x, return_tensors='pt', padding=True)
        labels = y
        for w in inputs:
            inputs[w] = inputs[w].cuda(device=0)
        labels = labels.cuda(device=0)
        out = model(**inputs, labels=labels)
        out[0].backward()
        optimizer.step()
        loss += out[0].item()
        optimizer.zero_grad()
    print(loss / len(train_loader))


def test(loader, data_set_name):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for x, y in tqdm(loader):
            inputs = tokenizer(x, return_tensors='pt', padding=True)
            labels = y
            for w in inputs:
                inputs[w] = inputs[w].cuda(device=0)
            labels = labels.cuda(device=0)
            out = model(**inputs)
            pred = torch.argmax(out[0], dim=-1, keepdim=False)
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()
            correct += np.sum(pred == labels)
            total += pred.shape[0]
        print(data_set_name, correct / total)

def split(loader, data_set_name):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for x, y in tqdm(loader):
            inputs = tokenizer(x, return_tensors='pt', padding=True)
            labels = y
            for w in inputs:
                inputs[w] = inputs[w].cuda(device=0)
            labels = labels.cuda(device=0)
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
    # test(test_eval_loader, "test")
    test(dev_eval_loader, 'dev')
    scheduler.step()
