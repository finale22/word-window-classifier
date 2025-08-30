import torch
from torch.utils.data import DataLoader
from functools import partial
from data_preprocessing import train_data

def vocabulary():
    vocabulary = set(w for s in train_data() for w in s)
    vocabulary.add("<unk>")
    vocabulary.add("<pad>")
    return vocabulary

def pad_window(sentence, window_size, pad_token="<pad>"):
    window = [pad_token] * window_size
    return window + sentence + window

def word_to_ix():
    word_to_ix = sorted(list(vocabulary()))
    return {word:ind for ind, word in enumerate(word_to_ix)}

def convert_token_to_indices(sentence, word_to_ix):
    return [word_to_ix.get(token, word_to_ix["<unk>"]) for token in sentence]

def custom_collate_fn(batch, window_size, word_to_ix):
    # Prepare the datapoints
    x, y = zip(*batch)
    x = [pad_window(s, window_size=window_size) for s in x]     # window padding
    x = [convert_token_to_indices(s, word_to_ix) for s in x]    # convert to indices
    
    # Pad x so that all the examples in the batch have the same size
    pad_token_ix = word_to_ix["<pad>"]
    x = [torch.LongTensor(x_i) for x_i in x]
    x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_ix)

    # Pad y and record the lengths
    lengths = [len(label) for label in y]
    lengths = torch.LongTensor(lengths)
    y = [torch.LongTensor(y_i) for y_i in y]
    y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    
    return x_padded, y_padded, lengths

def test_collate_fn(batch, window_size, word_to_ix):
    x = batch
    x = [pad_window(s, window_size=window_size) for s in x]     # window padding
    x = [convert_token_to_indices(s, word_to_ix) for s in x]
    pad_token_ix = word_to_ix["<pad>"]
    x = [torch.LongTensor(x_i) for x_i in x]
    x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_ix)
    
    return x_padded

def get_dataloader(data, batch_size, shuffle, window_size, collate_fn):
    collate_fn = partial(collate_fn, window_size=window_size, word_to_ix=word_to_ix())
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)