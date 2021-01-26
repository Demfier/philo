import os
import torch


class StupidError(Exception):
    """Class to handle user's stupidity"""
    pass


config = {
    # hyperparameters
    'lr': 1e-3,
    'dropout': 0.2,
    'patience': 3,  # number of epochs to wait before decreasing lr
    'gamma': 0.95,  # lr decay rate
    'min_lr': 1e-7,  # minimum allowable value of lr

    # model-specific hyperparams
    'dataset': 'iemocap',  # how2/multi30k/iemocap/savee
    'fusion_type': 'auto',  # None/'auto'/'gan'
    # NOTE: the modes s-t and v-t are equivalent to st-t and vt-t when NOT
    # training a classifier as such a mode as s-t doesn't make sense when
    # training a generative model (imagine translating just an english audio
    # vector into a portuguese sentence)
    'modalities': 'st-t',  # t-t/svt-t/sv-t/s-t/v-t/st-t/vt-t


    'clip': 50.0,  # values above which to clip the gradients
    'tf_ratio': 1.0,  # teacher forcing ratio

    'unit': 'lstm',
    'n_epochs': 300,
    'batch_size': 100,
    'enc_n_layers': 1,
    'dec_n_layers': 1,
    'disc_n_layers': 1,
    'dec_mode': 'greedy',  # type of decoding to use {greedy, beam}
    'bidirectional': True,  # make the encoder bidirectional or not
    'attn_model': None,  # None/'dot'/'concat'/'general'

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
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' => multiGPU)
    'num_gpus': torch.cuda.device_count(),
    'word_embedding': 'w2v',  # word embedding type to use: w2v/fasttext/glove/None
    'use_scheduler?': False,  # half lr every 3 non-improving batches
    'first_run?': True,  # True for the very first run
    'min_freq': 2,  # min frequency for the word to be a part of the vocab

    # directory paths
    'save_dir': 'saved_models/',
    'data_dir': 'data/processed/',
    'emb_dir': 'embeddings/',
    'pretrained_model': None,
    'fixed_ae_path': None,  # for GAN (or any other two-step training regime)
}


def get_dependent_params(config):
    if config['dataset'] == 'how2':
        config['lang_pair'] = 'en-pt'  # src-target language pair
        config['model_type'] = 'generative'
    elif config['dataset'] == 'multi30k':
        config['lang_pair'] = 'en-fr'
        config['model_type'] = 'generative'
    else:
        config['lang_pair'] = 'en-en'
        config['model_code'] = 'bilstm'
        config['model_type'] = 'classifier'
        if config['dataset'] == 'iemocap':
            # 8 input hand-crafted features and 300 dimensions after encoding
            config['audio_dim'] = (8, 300)
            # for the 4 classes - angry, happy, sad, neutral
            config['classes'] = ['angry', 'happy', 'sad', 'neutral']
            config['output_dim'] = len(config['classes'])

    # embedding directory
    emb_dir = '{}processed/{}/{}'.format(config['emb_dir'],
                                         config['dataset'],
                                         config['word_embedding'])

    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)

    config['vocab_dir'] = '{}{}/'.format(config['data_dir'],
                                         config['dataset'])

    config['beam_size'] = 3 if config['dec_mode'] == 'beam' else 1

    if config['device'] == 'x':
        if config['num_gpus'] > 1:
            config['device'] = 'cuda'
            config['multi_gpu_mode?'] = True
        else:
            raise StupidError("You don't have multiple GPUs stupid!")


get_dependent_params(config)
