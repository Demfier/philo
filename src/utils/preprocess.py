import os
import re
import sys
import h5py
import math
import torch
import gensim
import random
import unicodedata
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


def process_raw(config):
    """
    convert raw datafiles from different datasets:
        how2 => machine translation
        daily-dialog => dialog generation
    to one desired format
    """
    if config['task'] == 'mt':
        src, trg = config['lang_pair'].split('-')
        train_data, val_data, test_data = process_how2(src, trg)
    elif config['task'] == 'dialog':
        train_data, val_data, test_data = process_dailydial()
    elif config['task'] == 'rec':
        train_data, val_data, test_data = process_snli()
    elif config['task'] == 'dialog-rec':
        train_data, val_data, test_data = process_dailydial(for_rec=True)

    save_dir = '{}{}/'.format(config['data_dir'], config['task'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Save the processed files
    with open('{}/{}-train.txt'.format(save_dir, config['lang_pair']), 'w') as f:
        f.write(train_data)
    with open('{}/{}-val.txt'.format(save_dir, config['lang_pair']), 'w') as f:
        f.write(val_data)
    with open('{}/{}-test.txt'.format(save_dir, config['lang_pair']), 'w') as f:
        f.write(test_data)


def process_how2(src, trg):
    how2_dir = 'data/raw/mt/how2'

    train_data, val_data, test_data = '', '', ''

    # NOTE: Below, assumption is made that *_files have only two elements
    # Build train_data
    sources = open('{}/train.{}'.format(how2_dir, src), 'r').readlines()
    targets = open('{}/train.{}'.format(how2_dir, trg), 'r').readlines()
    for i in range(len(sources)):
        train_data += '{}\t{}\n'.format(sources[i].strip(), targets[i].strip())
    # Build val_data
    sources = open('{}/val.{}'.format(how2_dir, src), 'r').readlines()
    targets = open('{}/val.{}'.format(how2_dir, trg), 'r').readlines()
    for i in range(len(sources)):
        val_data += '{}\t{}\n'.format(sources[i].strip(), targets[i].strip())
    # Build test_data
    sources = open('{}/test.{}'.format(how2_dir, src), 'r').readlines()
    targets = open('{}/test.{}'.format(how2_dir, trg), 'r').readlines()
    for i in range(len(sources)):
        test_data += '{}\t{}\n'.format(sources[i].strip(), targets[i].strip())
    return train_data, val_data, test_data


def process_dailydial(for_rec=False):
    dailydial_dir = 'data/raw/dialog'
    train_data, val_data, test_data = '', '', ''
    for mode in ['train', 'valid', 'test']:
        # read the respective csv
        pairs = pd.read_csv('{}/df_daily_{}.csv'.format(dailydial_dir, mode))
        # Build data)
        if for_rec:
            for i in range(pairs.shape[0]):
                if mode == 'train':
                    train_data += '{0}\t{0}\n'.format(pairs['line'].get(i))
                    train_data += '{0}\t{0}\n'.format(pairs['reply'].get(i))
                elif mode == 'valid':
                    val_data += '{0}\t{0}\n'.format(pairs['line'].get(i))
                    val_data += '{0}\t{0}\n'.format(pairs['reply'].get(i))
                elif mode == 'test':
                    test_data += '{0}\t{0}\n'.format(pairs['line'].get(i))
                    test_data += '{0}\t{0}\n'.format(pairs['reply'].get(i))
        else:
            for i in range(pairs.shape[0]):
                if mode == 'train':
                    train_data += '{}\t{}\n'.format(pairs['line'].get(i),
                                                    pairs['reply'].get(i))
                elif mode == 'valid':
                    val_data += '{}\t{}\n'.format(pairs['line'].get(i),
                                                  pairs['reply'].get(i))
                elif mode == 'test':
                    test_data += '{}\t{}\n'.format(pairs['line'].get(i),
                                                   pairs['reply'].get(i))
    return train_data, val_data, test_data


def process_snli():
    with open('data/raw/rec/snli_all.txt', 'r') as f:
        snli_all = f.readlines()
    random.shuffle(snli_all)
    total = len(snli_all)
    # use train:val:test::70:15:15
    train_idx = math.floor(0.7*total)
    val_idx = math.floor(0.85*total)
    train = snli_all[:train_idx]
    val = snli_all[train_idx:val_idx]
    test = snli_all[-val_idx:]

    # Build data
    train_data, val_data, test_data = '', '', ''
    for mode in ['train', 'val', 'test']:
        if mode == 'train':
            for s in train:
                s = s.strip()
                train_data += '{}\t{}\n'.format(s, s)
        if mode == 'val':
            for s in val:
                s = s.strip()
                val_data += '{}\t{}\n'.format(s, s)
        if mode == 'test':
            for s in test:
                s = s.strip()
                test_data += '{}\t{}\n'.format(s, s)
    return train_data, val_data, test_data


# Vocab and file-reading part
class Vocabulary(object):
    """Vocabulary class"""
    def __init__(self, languages):
        super(Vocabulary, self).__init__()
        self.languages = languages
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4  # count the special tokens above
        self.index2language = {}

    def add_sentence(self, sentence, lang):
        for word in sentence.strip().split():
            self.add_word(word, lang)

    def add_word(self, word, lang):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.word2count[word] = 1
            self.index2word[self.size] = word
            self.index2language[self.size] = lang
            self.size += 1
        else:
            self.word2count[word] += 1

    def sentence2index(self, sentence):
        indexes = []
        for w in sentence.split():
            try:
                indexes.append(self.word2index[w])
            except KeyError as e:  # handle OOV
                indexes.append(self.word2index['<UNK>'])
        return indexes

    def index2sentence(self, indexes):
        return [self.index2word[i] for i in indexes]

    def filter(self, min_freq):
        exclude_keys = {'<PAD>', '<SOS>', '<EOS>', '<UNK>'}
        # Reindex and filter
        _word2index = {}
        _word2index.update({'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3})
        index = 4
        for w, i in self.word2index.items():
            if w not in exclude_keys and self.word2count[w] > min_freq:
                _word2index[w] = index
                index += 1
        self.word2index = _word2index
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.word2count = {k: v for k, v in self.word2count.items() if k in self.word2index}


def build_vocab(config):
    # TODO: Make different vocabs for source and target language
    languages = config['lang_pair']
    all_pairs = read_pairs('{}{}/'.format(config['data_dir'], config['task']),
                           languages, mode='all')
    all_pairs = filter_pairs(all_pairs, config['MAX_LENGTH'])
    vocab = Vocabulary(languages)
    src_lang, target_lang = languages.split('-')
    for pair in all_pairs:
        vocab.add_sentence(pair[0], src_lang)
        vocab.add_sentence(pair[1], target_lang)
    vocab.filter(config['min_freq'])
    print('Vocab size: {}'.format(vocab.size))
    np.save(config['vocab_path'], vocab, allow_pickle=True)
    return vocab


def read_pairs(data_dir, languages, mode='train'):
    """
    Reads src-target sentence pairs given a mode
    =============
    Params:
    =============
    data_dir (Str): location of the appropriate file
    languages (Str): the language pair (e.g. - en-pt)
    mode (Str): One of train/val/test/all.
        Shows the mode for which to read pairs
    """
    if mode == 'all':
        file_paths = ['{}{}-{}.txt'.format(data_dir, languages, m)
                      for m in ['train', 'val', 'test']]
    else:
        file_paths = ['{}{}-{}.txt'.format(data_dir, languages, mode)]

    # No one line loop to maintain flatness in `lines`
    lines = []
    for p in file_paths:
        lines += open(p, encoding='utf-8').readlines()

    pairs = []
    for line in lines:
        s1, s2 = line.split('\t', 1)
        pairs.append((normalize_string(s1), normalize_string(s2)))
    return pairs


def normalize_string(x):
    """Lower-case, trip and remove non-letter characters
    ==============
    Params:
    ==============
    x (Str): the string to normalize
    """
    x = unicode_to_ascii(x.lower().strip())
    x = re.sub(r'([.!?])', r'\1', x)
    x = re.sub(r'[^a-zA-Z.!?]+', r' ', x)
    return x


def unicode_to_ascii(x):
    return ''.join(
        c for c in unicodedata.normalize('NFD', x)
        if unicodedata.category(c) != 'Mn')


def filter_pairs(pairs, max_len):
    """
    Truncate pairs to have at most max_len tokens
    ==============
    Params:
    ==============
    pairs (list of tuples): each tuple is a src-target sentence pair
    max_len (Int): Max allowable sentence length
    """
    return [(
        (' '.join(pair[0].split()[:max_len])),
        (' '.join(pair[1].split()[:max_len]))
        ) for pair in pairs]


# Embeddings part
def generate_word_embeddings(vocab, config):
    # Load original (raw) embeddings
    wemb_type = config['wemb_type']
    src_lang, target_lang = vocab.languages.split('-')
    ftype = 'vec' if wemb_type == 'fasttext' else 'bin'

    # Train w2v models if not already trained
    if wemb_type == 'w2v' and not os.path.exists('{}raw/{}-{}.{}'.format(
            config['emb_dir'], wemb_type, src_lang, ftype)):
        train_w2v_model(config)

    src_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
        '{}raw/{}-{}.{}'.format(config['emb_dir'], wemb_type,
                                src_lang, ftype), binary=True)
    # Avoid extra memory footprint if src_lang == target_lang
    target_embeddings = \
        (src_embeddings if src_lang == target_lang else
         gensim.models.KeyedVectors.load_word2vec_format(
            '{}raw/{}-{}.{}'.format(config['emb_dir'], wemb_type,
                                    target_lang, ftype), binary=True))

    # Create filtered embeddings
    # Initialize filtered embedding matrix
    combined_embeddings = np.zeros((vocab.size, config['embedding_dim']))
    for index, word in vocab.index2word.items():
        try:  # random normal for special and OOV tokens
            if index <= 4:
                combined_embeddings[index] = \
                    np.random.normal(size=(config['embedding_dim'], ))
                continue  # use continue to avoid extra `else` block
            combined_embeddings[index] = src_embeddings[word] \
                if vocab.index2language[index - 4] == src_lang \
                else target_embeddings[word]
        except KeyError as e:
            combined_embeddings[index] = \
                np.random.normal(size=(config['embedding_dim'], ))

    with h5py.File(config['filtered_emb_path'], 'w') as f:
        f.create_dataset('data', data=combined_embeddings, dtype='f')
    return torch.from_numpy(combined_embeddings).float()


def train_w2v_model(config):
    languages = config['lang_pair']
    wemb_type = config['wemb_type']
    src_lang, target_lang = languages.split('-')
    all_pairs = read_pairs('{}{}/'.format(config['data_dir'], config['task']),
                           languages, mode='all')
    all_pairs = filter_pairs(all_pairs, config['MAX_LENGTH'])
    random.shuffle(all_pairs)
    src_sentences, target_sentences = [], []
    for pair in all_pairs:
        src_sentences.append(pair[0].split())
        target_sentences.append(pair[1].split())

    src_w2v = gensim.models.Word2Vec(src_sentences, size=300,
                                     min_count=1, iter=50)
    src_w2v.wv.save_word2vec_format('{}raw/{}-{}.bin'.format(
        config['emb_dir'], wemb_type, src_lang), binary=True)

    target_w2v = gensim.models.Word2Vec(target_sentences, size=300,
                                        min_count=1, iter=50)
    target_w2v.wv.save_word2vec_format('{}raw/{}-{}.bin'.format(
        config['emb_dir'], wemb_type, target_lang), binary=True)


def load_word_embeddings(config):
    with h5py.File(config['filtered_emb_path'], 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float()


def prepare_data(config):
    data_dir = config['data_dir']
    task = config['task']
    languages = config['lang_pair']
    max_len = config['MAX_LENGTH']

    train_pairs = filter_pairs(read_pairs('{}{}/'.format(data_dir, task),
                                          languages, 'train'), max_len)
    val_pairs = filter_pairs(read_pairs('{}{}/'.format(data_dir, task),
                                        languages, 'val'), max_len)
    test_pairs = filter_pairs(read_pairs('{}{}/'.format(data_dir, task),
                                         languages, 'test'), max_len)

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)
    return train_pairs, val_pairs, test_pairs


def batch_to_model_compatible_data(vocab, pairs, device):
    """
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source and target sentence pairs
    device (Str): Device to place the tensors in
    """
    sos_token = vocab.word2index['<SOS>']
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']

    src_indexes, src_lens, target_indexes = [], [], []
    for pair in pairs:
        src_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(pair[0]) + [eos_token]
                ))
        # extra 2 for sos_token and eos_token
        src_lens.append(len(pair[0].split()) + 2)

        target_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(pair[1]) + [eos_token]
                ))

    # pad src and target batches
    src_indexes = pad_sequence(src_indexes,
                               batch_first=True,
                               padding_value=pad_token)
    target_indexes = pad_sequence(target_indexes,
                                  batch_first=True,
                                  padding_value=pad_token)

    src_lens = torch.tensor(src_lens)
    return src_indexes.to(device), src_lens.to(device), \
        src_indexes.shape[1], target_indexes.to(device)


def _btmcd(vocab, pairs, device):
    """
    Alias for `batch_to_model_compatible_data`
    """
    return batch_to_model_compatible_data(vocab, pairs, device)
