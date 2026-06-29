import torch
from torch.utils.data import Dataset


class MultiLabelNERDataset(Dataset):
    def __init__(self, data, tokenizer, label2idx, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_len = max_len
        self.num_labels = len(label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = [str(w) for w in item["words"]]
        word_labels = item["labels"]

        encoding = self.tokenizer(words, is_split_into_words=True, truncation=True,
                                  padding='max_length', max_length=self.max_len,
                                  return_tensors='pt', add_special_tokens=True)

        labels = torch.full((self.max_len, self.num_labels), -100.0)
        word_ids = encoding.word_ids(batch_index=0)
        prev_word_idx = None

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                multi_hot = torch.zeros(self.num_labels)
                for label in word_labels[word_idx]:
                    if label in self.label2idx:
                        multi_hot[self.label2idx[label]] = 1.0
                labels[token_idx] = multi_hot
            prev_word_idx = word_idx

        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "labels": labels
        }
