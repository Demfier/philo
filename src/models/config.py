import os
import torch


class StupidError(Exception):
    """Class to handle user's stupidity"""
    pass


model_config = {
    # hyperparameters
    'lr': 1e-3,
    'dropout': 0.2,
    'patience': 3,  # number of epochs to wait before decreasing lr
    'gamma': 0.95,  # lr decay rate
    'min_lr': 1e-7,  # minimum allowable value of lr

    # model-specific hyperparams
    'dataset': 'how2',  # how2/multi30k/iemocap/savee
    'model_code': 'transfusion',  # None/transfusion/gan
    'modalities': 't-t',  # t-t/svt-t/sv-t/s-t/v-t


    'clip': 50.0,  # values above which to clip the gradients
    'tf_ratio': 1.0,  # teacher forcing ratio

    'unit': 'lstm',
    'n_epochs': 100,
    'batch_size': 100,
    'enc_n_layers': 1,
    'dec_n_layers': 1,
    'disc_n_layers': 1,
    'dec_mode': 'greedy',  # type of decoding to use {greedy, beam}
    'bidirectional': True,  # make the encoder bidirectional or not
    'attn_model': 'dot',  # None/dot/concat/general

    # dimensions
    'latent_dim': 256,
    'hidden_dim': 256,
    'embedding_dim': 300,

    # vocab-related params
    'PAD_TOKEN': 0,
    'SOS_TOKEN': 1,
    'EOS_TOKEN': 2,
    'UNK_TOKEN': 3,
    'MAX_LENGTH': 30,  # Max length of a sentence

    # run-time conf
    'device': 'x' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' for multiGPU mode)
    'num_gpus': torch.cuda.device_count(),
    'wemb_type': 'w2v',  # type of word embedding to use: w2v/fasttext
    'use_scheduler': True,  # half lr every 3 non-improving batches
    'use_embeddings?': True,  # use word embeddings or not
    'first_run?': True,  # True for the very first run
    'min_freq': 2,  # min frequency for the word to be a part of the vocab

    # directory paths
    'save_dir': 'saved_models/',
    'data_dir': 'data/processed/',
    'emb_dir': 'embeddings/',
    'pretrained_model': None,
    'fixed_ae_path': None,  # for GAN (or any other two-step training regime)
}


def get_dependent_params(model_config):
    if model_config['task'] == 'how2':
        model_config['lang_pair'] = 'en-pt'  # src-target language pair
    elif model_config['task'] == 'multi30k':
        model_config['lang_pair'] = 'en-fr'
    else:
        model_config['lang_pair'] = 'en-en'

    emb_dir = '{}processed/{}/{}'.format(model_config['emb_dir'],
                                         model_config['dataset'],
                                         model_config['wemb_type'])

    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)

    model_config['filtered_emb_path'] = \
        '{}{}-filtered.hdf5'.format(emb_dir, model_config['lang_pair'])

    vocab_dir = '{}{}/{}/'.format(model_config['data_dir'],
                                  model_config['dataset'],
                                  model_config['modalities'])

    model_config['vocab_path'] = '{}vocab.npy'.format(vocab_dir)
    model_config['beam_size'] = 3 if model_config['dec_mode'] == 'beam' else 1

    if model_config['device'] == 'x':
        if model_config['num_gpus'] > 1:
            model_config['device'] = 'cuda'
            model_config['multi_gpu_mode?'] = True
        else:
            raise StupidError("You don't have multiple GPUs stupid!")


get_dependent_params(model_config)
