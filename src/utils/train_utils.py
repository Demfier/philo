"""
This module contains helper functions for the main() functions in respective
training scripts. The helpers here aid towards clean training/testing scripts
as well
"""
import torch
from torch import nn
import numpy as np
from models import classifiers
from . import model_utils, preprocess_utils

from tqdm import tqdm
from datetime import datetime


def load_data(config, quick_test=False):
    """
    returns vocabulary, the train/val/test pairs, embeddings

    Parameters:
    ===========
    config (Dict):
        The configuration dictionary with all the crucial paramters
    quick_test (Bool) (optional):
        Flag to denote that a quick test of the pipeline is desired. If True,
        only consider a small subset of the dataset.
    """
    batch_size = config['batch_size']
    vocab = model_utils.load_vocabulary(config)

    print('Loading train, validation and test pairs.')
    train_pairs, val_pairs, test_pairs = preprocess_utils.prepare_data(config)

    if quick_test:
        train_pairs = train_pairs[:1000]
        val_pairs = val_pairs[:500]
        test_pairs = test_pairs[:500]

    n_train = len(train_pairs)
    # make #training samples a multiple of batch_size
    train_pairs = train_pairs[:batch_size * (n_train // batch_size)]
    print(train_pairs.sample(n=2))
    n_val = len(val_pairs)
    n_test = len(test_pairs)
    embedding_wts = model_utils.get_embedding_wts(vocab) \
        if config['word_embedding'] and \
        't' in config['modalities'][:-2] else None
    print('#Train: {} | #Test: {} | #Val: {}'.format(n_train, n_test, n_val))
    return vocab, train_pairs, val_pairs, test_pairs, embedding_wts


def build_model(config, device, vocab=None, embedding_wts=None):
    """
    return instance of the appropriate the neural network along with
    the epoch num from where to start the training process
    """
    print('Building model.')
    no_modality_support = False
    src_modalities = config['modalities'][:-2]
    if config['dataset'] == 'iemocap':
        if src_modalities == 't':  # trim '-t' from modalities
            model = classifiers.TextEmotionClassifier(
                config, embedding_wts=embedding_wts)
        elif src_modalities == 's':
            model = classifiers.AudioEmotionClassifier(config)
        elif set(src_modalities) == {'s', 't'}:
            model = classifiers.BiModalEmotionClassifier(
                config, embedding_wts=embedding_wts)
        else:
            no_modality_support = True
    else:
        raise ValueError('Invalid model code: {}'.format(config['model_code']))

    if no_modality_support:
        raise ValueError('Modality {} not yet supported for {}'.format(
            src_modalities, config['dataset']))

    model = model.to(device)

    if config['pretrained_model']:
        print('Restoring {}...'.format(config['pretrained_model']))
        checkpoint = torch.load('{}{}'.format(config['save_dir'],
                                              config['pretrained_model']),
                                map_location=device)
        model.load_state_dict(checkpoint['model'])
        epoch_init = checkpoint['epoch']
    else:
        epoch_init = 0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Parameters: {}'.format(num_params))
    return epoch_init, model


def train_epoch(model, criterion, optimizer,
                vocab, train_pairs, batch_size,
                clip, config, writer, epoch):
    model.train()
    epoch_loss = []
    n_train = len(train_pairs)
    train_iter = epoch*n_train
    for stage in ['disc_real', 'disc_fake', 'gen']:
        for iter in tqdm(range(0, n_train, batch_size)):
            optimizer.zero_grad()
            iter_pairs = train_pairs[iter:iter + batch_size]
            # get data batch
            x_train, y_train = \
                preprocess_utils._btmcd(vocab, iter_pairs, config)

            # do a forward pass through the model
            model_output_dict = model(x_train, 'train', stage)
            loss = criterion(model_output_dict['logits'], y_train)
            if model_output_dict['aux_loss'] is not None:
                loss += model_output_dict['aux_loss']
            loss.backward()
            loss = torch.mean(loss).item()

            # Clip gradients (wt. update) (very important)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()


            if 'disc' in stage:
                model.fusion_method.text_gan.discriminator.optimize(clip)
                model.fusion_method.speech_gan.discriminator.optimize(clip)
                model.fusion_method.video_gan.discriminator.optimize(clip)
            else:
                model.fusion_method.text_gan.generator.optimize(clip)
                model.fusion_method.speech_gan.generator.optimize(clip)
                model.fusion_method.video_gan.generator.optimize(clip)

            epoch_loss.append(loss)
            writer.add_scalar('data/train_loss', loss, train_iter)
            train_iter += 1
        # Print average batch loss
    print('{}>> Epoch [{}/{}]: Mean Train Loss: {}'.format(
        datetime.now().time(), epoch, config['n_epochs'], np.mean(epoch_loss)))
    return np.mean(epoch_loss)


def validate_epoch(model, criterion, vocab, val_pairs,
                   batch_size, config, writer, epoch):
    test_mode = True if epoch > 0 and (epoch+1) % 5 == 0 else False
    n_val = len(val_pairs)
    val_iter = epoch*n_val
    epoch_loss = []
    y = np.array([])
    all_preds = np.array([])
    with torch.no_grad():
        model.eval()
        for iter in tqdm(range(0, n_val, batch_size)):
            iter_pairs = val_pairs[iter:iter + batch_size]
            x_val, y_val = preprocess_utils._btmcd(vocab, iter_pairs, config)
            model_output_dict = model(x_val, 'infer')
            preds = torch.argmax(model_output_dict['logits'], dim=1)
            loss = criterion(model_output_dict['logits'], y_val)
            if model_output_dict['aux_loss'] is not None:
                loss += model_output_dict['aux_loss']
            loss = torch.mean(loss).item()
            epoch_loss.append(loss)
            writer.add_scalar(
                'data/{}_loss'.format('test' if test_mode else 'val'),
                loss, val_iter)
            val_iter += 1
            y = np.concatenate((y, y_val.cpu().numpy()))
            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
    print('Mean {} Loss: {}\n{}\nSamples:\n'.format(
                'Test' if test_mode else 'Val', np.mean(epoch_loss), '-'*30))
    return np.mean(epoch_loss), all_preds, y
