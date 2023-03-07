import torch
from torch.utils.data.dataloader import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import argparse

from text_dataset import TextEmbeddingDataset, load_text_dataset
from models import TextClassificationModel


parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--dataset','-d', type=str, choices=["SetFit/20_newsgroups","ag_news","sst2","imdb"], default="SetFit/20_newsgroups")
parser.add_argument('--batch_size','-b', type=int, default=16)
parser.add_argument("--gpu", '-g', type=int, choices=[0,1,2,3], default=0)
parser.add_argument("--num_classes", '-n', type=int)
parser.add_argument("--emsize", '-s', type=int, default=64)
parser.add_argument("--epoch", '-e', type=int, default=2)
parser.add_argument("--lr", type=float, default=5)

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

# load tokenizer
tokenizer = get_tokenizer("basic_english")

def yield_tokens(sentence):
    for sent in sentence:
        yield tokenizer(sent)

vocab = build_vocab_from_iterator(yield_tokens(train_set[0]), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list, offsets, label_list

train_dataset = TextEmbeddingDataset(train_set[0], train_set[1], vocab=vocab, tokenizer=tokenizer)
test_dataset = TextEmbeddingDataset(test_set[0], test_set[1], vocab=vocab, tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_batch)
train_eval_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_batch)
dev_eval_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_batch)


vocab_size = len(vocab)
text_model = TextClassificationModel(vocab_size, EMSIZE, NUM_CLASSES).to(device)

def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0

    for text, offsets, label in dataloader:
        text = text.to(device)
        offsets = offsets.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

    print('| epoch {:3d} | accuracy {:8.3f}'.format(epoch, total_acc/total_count))

def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for (text, offsets, label) in dataloader:
            text = text.to(device)
            offsets = offsets.to(device)
            label = label.to(device)
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(text_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

total_accu = None

for epoch in range(1, EPOCHS + 1):
    train(text_model, train_loader, optimizer, criterion, epoch)
    accu_val = evaluate(text_model, dev_eval_loader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | valid accuracy {:8.3f} '.format(epoch, accu_val))
    print('-' * 59)

