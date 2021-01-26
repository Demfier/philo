"""
This module contains utilities mostly used during training/testing the networks
"""

import os
import torch
import numpy as np
from . import preprocess_utils
from models.config import config


def load_vocabulary(config):
    vocab_path = '{}vocab-en.npy'.format(config['vocab_dir'])
    if os.path.exists(vocab_path) and not config['first_run?']:
        # need to use .item() to access a class object
        vocab = np.load(vocab_path, allow_pickle=True).item()
        print('Loaded vocabulary.')
    else:
        # build a single vocab for both the languages
        sentences = preprocess_utils.prepare_for_vocab(config)
        if sentences is None:
            return
        print('Building vocabulary...')
        vocab = preprocess_utils.build_vocab(config, sentences, 'en')
        print('Built vocabulary for en.')
    return vocab


def get_embedding_wts(vocab):
    if config['first_run?']:
        print('First run: GENERATING filtered embeddings.')
        preprocess_utils.generate_word_embeddings(config, vocab)
    print('LOADING filtered embeddings.')
    languages = set(config['lang_pair'].split('-'))
    return preprocess_utils.load_word_embeddings(config, languages)


def save_snapshot(model, epoch_num):
    if not os.path.exists(config['save_dir']):
        os.mkdir(config['save_dir'])
    if not os.path.exists('{}/{}'.format(config['save_dir'], config['dataset'])):
        os.mkdir('{}/{}'.format(config['save_dir'], config['dataset']))
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch_num
        }, '{}{}/{}/{}/{}/{}{}-{}L-{}{}-{}'.format(
            config['save_dir'],
            config['dataset'],
            config['fusion_type'] if config['fusion_type'] else 'no_align',
            config['modalities'],
            config['lang_pair'],
            config['model_code'],
            '-attn' if config['attn_model'] is not None else '',
            config['enc_n_layers'],
            'bi' if config['bidirectional'] else '',
            config['unit'], epoch_num)
        )


def translate(vocab, logits, y, x, inputs, generated, ground_truth):
    """
    Converts model output logits and tokenized batch y into to sentences
    logits -> (bs, max_length, vocab_size)
    y -> (bs, max_y_len)
    x -> (bs, max_x_len)

    Effects: Mutates generated, and ground_truth
    """
    inp_tokens = x[:, 1:]
    _, pred_tokens = torch.max(logits.transpose(0, 1)[1:], dim=2)
    pred_tokens = pred_tokens.permute(1, 0)
    gt_tokens = y[:, 1:]

    # Get sentences from token ids
    for token_list in inp_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        inputs.append(sentence)

    for token_list in pred_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        generated.append(sentence)

    for token_list in gt_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        # making it a listof(listof Str) as we need to calculate the bleu score
        ground_truth.append([sentence])
