import os
import re
import sys
import h5py
import math
import torch
import gensim
import pickle
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from .classes import Vocabulary
from torch.nn.utils.rnn import pad_sequence


def process_raw(config):
    """
    convert raw datafiles from different datasets to one desired format
    NOTE: The dataset stored below will be a list of dictionaries with
    appropriate keys. This will make easier to construct dataframes later on
    """
    if config['dataset'] == 'how2':
        src, trg = config['lang_pair'].split('-')
        combined_data, train_data, val_data, test_data = process_how2(src, trg)
    elif config['dataset'] == 'multi30k':
        combined_data, train_data, val_data, test_data = process_multi30k()
    elif config['dataset'] == 'iemocap':
        combined_data, train_data, val_data, test_data = process_iemocap()
    elif config['dataset'] == 'savee':
        combined_data, train_data, val_data, test_data = process_savee()

    save_dir = '{}{}/{}/{}/'.format(config['data_dir'],
                                    config['dataset'],
                                    config['modalities'],
                                    config['lang_pair'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Save the processed files
    with open('{}train.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(train_data, f)
    with open('{}val.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(val_data, f)
    with open('{}test.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(test_data, f)
    with open('{}combined.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(combined_data, f)

    print('Saved train/val/test split for {}'.format(config['dataset']))


def process_how2(src, trg):
    how2_dir = 'data/raw/how2'

    train_data, val_data, test_data = [], [], []

    # NOTE: Below, assumption is made that *_files have only two elements
    # Build train_data
    sources = open('{}/train.{}'.format(how2_dir, src), 'r').readlines()
    targets = open('{}/train.{}'.format(how2_dir, trg), 'r').readlines()
    for i in range(len(sources)):
        train_data.append('{}\t{}'.format(sources[i].strip(), targets[i].strip()))
    # Build val_data
    sources = open('{}/val.{}'.format(how2_dir, src), 'r').readlines()
    targets = open('{}/val.{}'.format(how2_dir, trg), 'r').readlines()
    for i in range(len(sources)):
        val_data.append('{}\t{}'.format(sources[i].strip(), targets[i].strip()))
    # Build test_data
    sources = open('{}/test.{}'.format(how2_dir, src), 'r').readlines()
    targets = open('{}/test.{}'.format(how2_dir, trg), 'r').readlines()
    for i in range(len(sources)):
        test_data.append('{}\t{}'.format(sources[i].strip(), targets[i].strip()))
    return train_data, val_data, test_data


def process_iemocap():
    """
    This function prepares data for all experiment modes for IEMOCAP dataset
    NOTE: val == test for IEMOCAP (there are already very less data samples)
    NOTE: Same files will be used for all different modalities (s-t, t-t, st-t)
    The read_data function for iemocap will extract the appropriate values
    at training/testing time.
    """
    iemocap_dir = 'data/interim/iemocap'

    # read audio data
    audio_train = pd.read_csv('{}/s-t/en-en/audio_train.csv'.format(iemocap_dir))
    audio_test = pd.read_csv('{}/s-t/en-en/audio_test.csv'.format(iemocap_dir))

    # read text data
    text_train = pd.read_csv('{}/t-t/en-en/text_train.csv'.format(iemocap_dir))
    text_test = pd.read_csv('{}/t-t/en-en/text_test.csv'.format(iemocap_dir))

    # check if dataframes for the two modalties are aligned
    assert audio_train.shape[0] == text_train.shape[0]
    assert audio_test.shape[0] == text_test.shape[0]
    assert audio_train[['wav_file', 'label']].equals(text_train[['wav_file', 'label']])
    assert audio_test[['wav_file', 'label']].equals(text_test[['wav_file', 'label']])

    # prepare train data for all modalities
    print('Preparing training set for IEMOCAP...')
    train = []
    for idx in tqdm(range(text_train.shape[0])):
        sample = {}
        sample['label'] = text_train.iloc[idx]['label']
        sample['text'] = text_train.iloc[idx]['transcription']
        sample['audio'] = np.array(audio_train.iloc[idx][2:])
        train.append(sample)

    # prepare test data for all modalities
    print('Preparing test set for IEMOCAP...')
    test = []
    for idx in tqdm(range(text_test.shape[0])):
        sample = {}
        sample['label'] = text_test.iloc[idx]['label']
        sample['text'] = text_test.iloc[idx]['transcription']
        sample['audio'] = np.array(audio_test.iloc[idx][2:])
        test.append(sample)

    return train + test, train, test, test


def prepare_for_vocab(config):
    """
    returns data necessary for creating vocab for an experiment
    """
    if 't' in config['modalities'].split('-')[0]:
        return filter_data(config, read_data(config))['text']


def build_vocab(config, sentences, language):
    vocab = Vocabulary()
    for s in sentences:
        vocab.add_sentence(s)
    vocab.filter(config['min_freq'])
    print('Vocab size: {}'.format(vocab.size))
    vocab_path = '{}vocab-{}.npy'.format(config['vocab_dir'], language)
    np.save(vocab_path, vocab, allow_pickle=True)
    return vocab


def read_data(config, mode='all'):
    """
    returns the appropriate dataset file and return data points in appropriate
    format. See the description below for more details.

    Parameters:
    ===========
    config (Dict):
        The main configuration dictionary.
    mode (Str) (optional):
        The mode for which to read the dataset. The possible values are
        train/val/test/all

    Returns:
    ========
    samples (pd.DataFrame):
        The samples df will have appropriate columns depending on the modalties
        in use
        For instance, purely text-based classifiers will have only two
        columns - 'text' and 'label'
    """
    data_dir = '{}{}/{}/{}/{}.pkl'.format(config['data_dir'],
                                          config['dataset'],
                                          config['modalities'],
                                          config['lang_pair'],
                                          'combined' if mode == 'all' else mode)
    with open(data_dir, 'rb') as f:
        dataset = pickle.load(f)

    if config['dataset'] == 'iemocap':
        return read4iemocap(dataset, config['modalities'])


def read4iemocap(dataset, modalities):
    df = pd.DataFrame(dataset)
    if modalities == 't-t':
        # remove input from audio modality
        df = df[df.columns.difference(['audio'])]
        # normalize sentences
        df['text'] = df['text'].map(normalize_string)
    elif modalities == 's-t':
        # remove input from text modality
        df = df[df.columns.difference(['text'])]
    elif modalities == 'st-t':
        df['text'] = df['text'].map(normalize_string)
    return df


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


def filter_data(config, df):
    """
    NOTE: currently only includes code for IEMOCAP. Add other datasets later
    Parameters:
    ===========
    df (pd.DataFrame):
        The dataframe from where sentences are to be filtered
    max_len (Int):
        max tokens to include in a sentence
    """
    # using apply rather than map as we need to pass an extra argument
    max_len = config['MAX_LENGTH']
    df['text'] = df.apply(lambda x: reduce_sentences(x['text'], max_len),
                          axis=1)
    return df[df['text'] != '']


def reduce_sentences(sentence, max_len):
    """Assumes the text is already normalized"""
    return ' '.join(sentence.split()[:max_len]).strip()


# Embeddings part
def generate_word_embeddings(config, vocab):
    # Load original (raw) embeddings
    src_lang, target_lang = config['lang_pair'].split('-')
    ftype = 'vec' if config['word_embedding'] == 'fasttext' else 'bin'

    # Train w2v models if not already trained
    if config['word_embedding'] == 'w2v' and not \
        os.path.exists('{}raw/{}/{}/{}.{}'.format(config['emb_dir'],
                                                  config['dataset'],
                                                  config['word_embedding'],
                                                  src_lang, ftype)):
        train_w2v_model(config)

    for lang in set([src_lang, target_lang]):
        filter_word_embeddings(config, lang, ftype)


def train_w2v_model(config):
    print('Training w2v model for {}'.format(config['dataset']))
    languages = config['lang_pair']
    wemb_type = config['word_embedding']
    src_lang, target_lang = languages.split('-')

    all_pairs = read_data(config)
    all_pairs = filter_data(config, all_pairs)
    all_pairs.sample(frac=1).reset_index(drop=True)

    # For IEMOCAP
    for lang in set([src_lang, target_lang]):
        print('---Language: {}'.format(lang))
        sentences = all_pairs['text'].map(str.split)
        w2v = gensim.models.Word2Vec(sentences, size=300,
                                     min_count=1, iter=50)
        save_dir = '{}raw/{}/{}/{}.bin'.format(config['emb_dir'],
                                               config['dataset'],
                                               wemb_type, lang)
        w2v.wv.save_word2vec_format(save_dir, binary=True)


def filter_word_embeddings(config, language, ftype):
    w2v_dir = '{}raw/{}/{}/{}.{}'.format(config['emb_dir'],
                                         config['dataset'],
                                         config['word_embedding'],
                                         language, ftype)
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir,
                                                                 binary=True)

    print('Loading vocab for {}'.format(language))
    vocab = np.load('{}vocab-{}.npy'.format(config['vocab_dir'], language),
                    allow_pickle=True).item()

    # Create filtered embeddings
    filtered_embeddings = np.zeros((vocab.size, config['embedding_dim']))
    for index, word in vocab.index2word.items():
        try:  # random normal for special and OOV tokens
            if index <= 4:
                filtered_embeddings[index] = \
                    np.random.normal(size=(config['embedding_dim'], ))
                continue  # use continue to avoid extra `else` block
            filtered_embeddings[index] = embeddings[word]
        except KeyError as e:
            filtered_embeddings[index] = \
                np.random.normal(size=(config['embedding_dim'], ))

    save_dir = '{}processed/{}/{}/{}.{}'.format(config['emb_dir'],
                                                config['dataset'],
                                                config['word_embedding'],
                                                language, ftype)
    with h5py.File(save_dir, 'w') as f:
        f.create_dataset('data', data=filtered_embeddings, dtype='f')
    print('Saved filtered embedding for {}'.format(language))


def load_word_embeddings(config, languages):
    embeddings_wts = {}
    for lang in languages:
        embeddings_wts[lang] = load_lang_word_embeddings(config, lang)
    return embeddings_wts


def load_lang_word_embeddings(config, language):
    wemb_type = config['word_embedding']
    ftype = 'vec' if wemb_type == 'fasttext' else 'bin'
    w2v_dir = '{}processed/{}/{}/{}.{}'.format(config['emb_dir'],
                                               config['dataset'],
                                               wemb_type, language,
                                               ftype)
    with h5py.File(w2v_dir, 'r') as f:
        return torch.from_numpy(
            np.array(f['data'])).float().to(config['device'])


def prepare_data(config):
    data_dir = '{}{}/{}/'.format(config['data_dir'],
                                 config['dataset'],
                                 config['modalities'],
                                 config['lang_pair'])

    train_pairs = read_data(config, 'train')
    val_pairs = read_data(config, 'val')
    test_pairs = read_data(config, 'test')

    # filter iff text is present in the input modality
    if 't' in config['modalities'].split('-')[0]:
        train_pairs = filter_data(config, train_pairs)
        val_pairs = filter_data(config, val_pairs)
        test_pairs = filter_data(config, test_pairs)

    # shuffle data samples
    train_pairs.sample(frac=1).reset_index(drop=True)
    val_pairs.sample(frac=1).reset_index(drop=True)
    test_pairs.sample(frac=1).reset_index(drop=True)
    return train_pairs, val_pairs, test_pairs


def batch_to_model_compatible_data_seq2seq(vocab, pairs, device):
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


def batch_to_model_compatible_data_clf(vocab, pairs, config):
    """
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source and target sentence pairs
    config (Dict): Configuration dictionary
    """
    device = config['device']
    src_modalities = config['modalities'][:-2]  # trim -t from the end

    x_train = {'text': [], 'text_lengths': [], 'audio': []}
    y_train = torch.tensor(np.array(pairs['label'])).to(device)

    if 's' in src_modalities:
        audio_batch = np.array(list(pairs['audio'])).astype('float32')
        x_train['audio'] = torch.tensor(audio_batch).to(device)
        if src_modalities == 's':
            return x_train, y_train

    sos_token = vocab.word2index['<SOS>']
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']
    for idx, row in pairs.iterrows():
        x_train['text'].append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(row.text) + [eos_token]
                ))
        # extra 2 for sos_token and eos_token
        x_train['text_lengths'].append(len(row.text.split()) + 2)

    # pad src and target batches
    x_train['text'] = pad_sequence(x_train['text'],
                                   batch_first=True,
                                   padding_value=pad_token).to(device)

    x_train['text_lengths'] = \
        torch.tensor(x_train['text_lengths']).long().to(device)

    return x_train, y_train.to(device)


def _btmcd(vocab, pairs, config):
    """
    Alias for `batch_to_model_compatible_data`
    """
    if config['model_type'] == 'classifier':
        return batch_to_model_compatible_data_clf(vocab, pairs, config)
    return batch_to_model_compatible_data(vocab, pairs, config)
