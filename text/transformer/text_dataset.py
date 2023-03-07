from torch.utils.data.dataset import Dataset
from datasets import load_dataset

class TextEmbeddingDataset(Dataset):
    def __init__(self, sentences, labels,vocab, tokenizer):
        self.dataset = sentences
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)

class TextDataset(Dataset):
    def __init__(self, sentences, labels):
        self.dataset = sentences
        self.labels = labels

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)


def load_text_dataset(dataset_name):
    """
    options:
        SetFit/20_newsgroups->20
        ag_news->4: "World","Sports", "Business", Sci/Tech"
        sst2->2
        imdb->2
    """
    dataset = load_dataset(dataset_name)
    if dataset_name == "sst2":
        train_set = dataset['train']['sentence'], dataset['train']['label']
        test_set = dataset['validation']['sentence'], dataset['validation']['label']
    else:
        train_set = dataset['train']['text'], dataset['train']['label']
        test_set = dataset['test']['text'], dataset['test']['label']
    return train_set, test_set

if __name__ == "__main__":
    train_set, _ = load_text_dataset("imdb")
    print(max(train_set[1]))



