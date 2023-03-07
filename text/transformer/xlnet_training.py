from transformers import XLNetTokenizer, XLNetModel
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse

from text_dataset import TextDataset, load_text_dataset
from models import XLNETGRU

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
NET = "xlnet"

# set hyperparameters
DEVICE = torch.device(f"cuda:{GPU}")
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

# load dataset
train_set, test_set = load_text_dataset(DATASET)

train_dataset = TextDataset(train_set[0], train_set[1])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
train_eval_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)

test_dataset = TextDataset(test_set[0], test_set[1])
test_eval_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

# tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
xlnet = xlnet.to(DEVICE)

model = XLNETGRU(xlnet,
                EMSIZE,
                NUM_CLASSES,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT)

for name, param in model.named_parameters():
    if name.startswith(f'{NET}'):
        param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
model = model.to(DEVICE)
criterion = criterion.to(DEVICE)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
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
            inputs[w] = inputs[w].to(DEVICE)
        label = label.to(dtype=torch.long, device=DEVICE)
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
                inputs[w] = inputs[w].to(DEVICE)
            label = label.to(dtype=torch.long, device=DEVICE)
            predictions = model(**inputs)
            acc = binary_accuracy(predictions, label)
            epoch_acc += acc.item()
    print(f'test acc:\t{epoch_acc / len(iterator)}')
    return epoch_acc / len(iterator)

for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, criterion)
    test(model, test_eval_loader)
    scheduler.step()