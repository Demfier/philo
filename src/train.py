import os
import h5py
import numpy as np
from datetime import datetime

import torch
from torch import nn, optim

from tqdm import tqdm
from pprint import pprint
from tensorboardX import SummaryWriter

from models import dae, vae
from utils import preprocess, metrics, model_utils
from models.config import model_config as conf


def load_vocabulary():
    if os.path.exists(conf['vocab_path']) and not conf['first_run?']:

        # Trick for allow_pickle issue in np.load
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        # need to use .item() to access a class object
        vocab = np.load(conf['vocab_path']).item()
        # restore np.load for future normal usage
        np.load = np_load_old
        print('Loaded vocabulary.')
    else:
        # build a single vocab for both the languages
        print('Building vocabulary...')
        vocab = preprocess.build_vocab(conf)
        print('Built vocabulary.')
    return vocab


def get_embedding_wts(vocab):
    if conf['first_run?']:
        print('First run: GENERATING filtered embeddings.')
        embedding_wts = preprocess.generate_word_embeddings(vocab, conf)
    else:
        print('LOADING filtered embeddings.')
        embedding_wts = preprocess.load_word_embeddings(conf)
    return embedding_wts.to(conf['device'])


def main():
    # Initialize tensorboardX writer
    writer = SummaryWriter()
    vocab = load_vocabulary()
    pprint(conf)

    print('Loading train, validation and test pairs.')
    train_pairs, val_pairs, test_pairs = preprocess.prepare_data(conf)
    # train_pairs = train_pairs[:1000]
    # val_pairs = val_pairs[:500]
    # test_pairs = test_pairs[:500]
    n_train = len(train_pairs)
    train_pairs = train_pairs[: conf['batch_size'] * (
        n_train // conf['batch_size'])]
    print(np.random.choice(train_pairs))
    n_val = len(val_pairs)
    n_test = len(test_pairs)
    device = conf['device']
    num_gpus = torch.cuda.device_count()
    embedding_wts = get_embedding_wts(vocab) if conf['word_embedding'] else None

    print('Building model.')
    model = dae.AutoEncoder(conf, vocab, embedding_wts)
    # if conf.get('multi_gpu_mode?') is not None:
    #     print('Using {} GPUs'.format(num_gpus))
    #     model = nn.DataParallel(model)
    model = model.to(device)

    if conf['pretrained_model']:
        print('Restoring {}...'.format(conf['pretrained_model']))
        checkpoint = torch.load('{}{}'.format(conf['save_dir'],
                                              conf['pretrained_model']),
                                map_location=device)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    print(model)

    optimizer = optim.Adam(model.parameters(), conf['lr'])
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=1,
    #     gamma=conf['gamma'],
    #     last_epoch=-1)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total {} trainable parameters'.format(num_params))

    print('Training started..')
    train_iter, val_iter, test_iter = 0, 0, 0
    for e in range(epoch, conf['n_epochs']):
        epoch_loss = []
        # Train for an epoch
        for iter in tqdm(range(0, n_train, conf['batch_size'])):
            optimizer.zero_grad()
            iter_pairs = train_pairs[iter: iter + conf['batch_size']]
            if len(iter_pairs) == 0:  # handle the strange error
                continue

            # get data batch
            x_train, x_lens, total_length, y_train = \
                preprocess._btmcd(vocab, iter_pairs, device)

            for stage in ['disc_real', 'disc_fake', 'gen']:
                pred_dict = model(x_train, x_lens, total_length, y_train, stage)

                # Clip gradients (wt. update) (very important)
                nn.utils.clip_grad_norm_(model.parameters(), conf['clip'])
                optimizer.step()
                loss = torch.mean(pred_dict['loss']).item()
                # y_train -> (max_y_len, bs)
                epoch_loss.append(loss)
                writer.add_scalar('data/train_loss', loss, train_iter)
            train_iter += 1

        # Print average batch loss
        print('{}>> Epoch [{}/{}]: Mean Train Loss: {}'.format(
            datetime.now().time(), e, conf['n_epochs'], np.mean(epoch_loss)))

        epoch_loss = []

        # Validate
        with torch.no_grad():
            inputs = []
            generated = []
            actual = []
            for iter in tqdm(range(0, n_val, conf['batch_size'])):
                iter_pairs = val_pairs[iter: iter + conf['batch_size']]

                x_val, x_lens, total_length, y_val = \
                    preprocess._btmcd(vocab, iter_pairs, device)

                pred_dict = model(x_val, x_lens, total_length)
                outputs = pred_dict['pred_outputs']
                # outputs => (bs, T, vocab.size)
                loss = torch.mean(pred_dict['loss']).item()
                epoch_loss.append(loss)
                writer.add_scalar('data/val_loss', loss, val_iter)
                # Get sentences from logits and token ids
                model_utils.translate(vocab, outputs, y_val, x_val,
                                      inputs, generated, actual)
                val_iter += 1
            print('Mean Validation Loss: {}\n{}\nSamples:\n'.format(
                np.mean(epoch_loss), '-'*30))
            # scheduler.step()
            # Sample some val sentences randomly
            for sample_id in random.sample(list(range(n_val)), 3):
                print('I: {}\nG: {}\nA: {}\n'.format(
                    ' '.join(inputs[sample_id]),
                    ' '.join(generated[sample_id]),
                    ' '.join(actual[sample_id][0]))
                )

            # Get all scores
            scores = metrics.calculate_all_metrics(generated, actual)
            writer.add_scalars('metrics/val_bleus', scores['bleus'], val_iter)
            writer.add_scalar('metrics/val_meteor', scores['meteor'], val_iter)
            print('Validation BLEU (1-4) scores:')
            print('{bleu1} | {bleu2} | {bleu3} | {bleu4}'.format(**scores['bleus']))
            print('Validation Meteor score: {}'.format(scores['meteor']))
            print('{}{}\n'.format('<'*15, '>'*15))

            # Save model
            model_utils.save_snapshot(model, e)

            epoch_loss = []
            inputs = []
            generated = []
            actual = []

            # Get test BLEU scores every 5 epochs
            if e > 0 and e % 5 == 0:
                for iter in tqdm(range(0, n_test, conf['batch_size'])):
                    iter_pairs = test_pairs[iter: iter + conf['batch_size']]

                    x_test, x_lens, total_length, y_test = \
                        preprocess._btmcd(vocab, iter_pairs, device)

                    pred_dict = model(x_test, x_lens, total_length)
                    outputs = pred_dict['pred_outputs']
                    loss = torch.mean(pred_dict['loss']).item()
                    epoch_loss.append(loss)
                    writer.add_scalar('data/test_loss', loss, test_iter)
                    # Get sentences
                    model_utils.translate(vocab, outputs, y_test, x_test,
                                          inputs, generated, actual)
                    print(len(inputs), len(generated), len(actual), n_test)
                    test_iter += 1

                print('Mean Test Loss: {}\n{}\nSamples:\n'.format(
                    np.mean(epoch_loss), '-'*30))
                # Sample some test sentences randomly
                for sample_id in random.sample(list(range(n_test)), 3):
                    print(sample_id)
                    print('I: {}\nG: {}\nA: {}\n'.format(
                        ' '.join(inputs[sample_id]),
                        ' '.join(generated[sample_id]),
                        ' '.join(actual[sample_id][0]))
                    )
                scores = metrics.calculate_all_metrics(generated, actual)
                writer.add_scalars('metrics/test_bleus', scores['bleus'], test_iter)
                writer.add_scalar('metrics/test_meteor', scores['meteor'], test_iter)
                print('Test BLEU (1-4) scores:\n{}'.format('-'*30))
                print('{bleu1} | {bleu2} | {bleu3} | {bleu4}'.format(**scores['bleus']))
                print('Test Meteor score: {}'.format(scores['meteor']))
                print('{}{}\n'.format('<'*15, '>'*15))
    writer.close()


if __name__ == '__main__':
    main()
