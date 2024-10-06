from torch.utils.data import Dataset
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import itertools
import torch

import numpy as np

class AGNewsMergedDataset(Dataset):
    def __init__(self, transform=None):
        # tokenizer = get_tokenizer('basic_english')
        
        # Load and merge both splits
        merged_data = list(itertools.chain(AG_NEWS(split='train'), AG_NEWS(split='test')))
        
        # # Build vocabulary from merged dataset
        # vocab = build_vocab_from_iterator(map(tokenizer, 
        #                                       (text for _, text in merged_data)),
        #                                   specials=["<unk>"])
        # vocab.set_default_index(vocab["<unk>"])

        self.transform = transform
        
        # Pre-tokenize and preprocess the data, then store in memory
        self.data = []
        targets = []
        for label, text in merged_data:
            # tokenized_text = [vocab[token] for token in tokenizer(text)]
            # data.append(torch.tensor(tokenized_text, dtype=torch.long))
            self.data.append(text)
            targets.append(label - 1)  # Adjust label to be zero-based
        
        # Pad sequences for uniformity
        # self.data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.targets[idx]
        return self.data[idx], self.targets[idx]
    
    @staticmethod
    def domains():
        return [
            "none"
        ]