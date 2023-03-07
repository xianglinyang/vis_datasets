from transformers import *
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from text_dataset import TextDataset, load_text_dataset
from models import BERTGRUSentiment

parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--dataset','-d', type=str, choices=["SetFit/20_newsgroups","ag_news","sst2","imdb"], default="SetFit/20_newsgroups")
parser.add_argument('--batch_size','-b', type=int, default=16)
parser.add_argument("--gpu", '-g', type=int, choices=[0,1,2,3], default=0)
parser.add_argument("--num_classes", '-n', type=int)
parser.add_argument("--emsize", '-s', type=int, default=256)
parser.add_argument("--epoch", '-e', type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-5)

args = parser.parse_args()
DATASET = args.dataset
BATCH_SIZE = args.batch_size
GPU = args.gpu
NUM_CLASSES = args.num_classes
EMSIZE = args.emsize
EPOCHS = args.epoch # epoch
LR = args.lr  # learning rate

# Set hyperparameter
device = torch.device(f'cuda:{GPU}')

# load dataset
train_set, test_set = load_text_dataset(DATASET)

train_dataset = TextDataset(train_set[0], train_set[1])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
train_eval_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)

test_dataset = TextDataset(test_set[0], test_set[1])
test_eval_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

bert = BertModel.from_pretrained('bert-base-uncased')
model = BERTGRUSentiment(bert,
                         EMSIZE,
                         NUM_CLASSES,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

model = model.to(device)

for name, param in model.named_parameters():   
    print(name)                
    if name.startswith('bert'):
        param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
criterion = torch.nn.CrossEntropyLoss()


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    prediction = preds.argmax(axis=1)
    correct = (prediction==y).float()
    acc = correct.sum()/len(correct)
    return acc

def train():
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for text, label in tqdm(train_loader):
        inputs = tokenizer(text, return_tensors='pt',padding=True)
        
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


for epoch in range(EPOCHS):
    train()
    test(train_eval_loader, "train")
    test(test_eval_loader, 'test')
    scheduler.step()
