import torch
import random
from utils import preprocess, metrics
from models.config import model_config as conf
from train_model import load_vocabulary, get_embedding_wts
from models import dae, vae


def translate(vocab, logits):
    """
    Converts model output logits to sentences
    logits -> (max_y_len, bs, vocab_size)

    """
    _, pred_tokens = torch.max(logits[1:], dim=2)
    pred_tokens = pred_tokens.permute(1, 0)

    generated = []
    for token_list in pred_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        generated.append(' '.join(sentence))
    return generated


def get_z(x, x_lens, model):
    encoder_dict = model._encode(x, x_lens)
    mu = encoder_dict['mu']
    log_sigma = encoder_dict['log_sigma']
    return model._reparameterize(mu, log_sigma)


def main():
    vocab = load_vocabulary()
    embedding_wts = get_embedding_wts(vocab)
    device = conf['device']
    if conf['model_code'] == 'dae':
        model = dae.AutoEncoder(conf, vocab, embedding_wts)
    elif conf['model_code'] == 'vae':
        model = vae.VariationalAutoEncoder(conf, vocab, embedding_wts)
    elif 'ved' in conf['model_code']:
        model = vae.BimodalVED(conf, vocab, embedding_wts)
    else:
        raise ValueError('Invalid model code: {}'.format(conf['model_code']))

    model.load_state_dict(
        torch.load(
            '{}{}'.format(conf['save_dir'], conf['pretrained_model']),
            map_location=device)['model'])

    with torch.no_grad():
        print('\n### Random Sampling ###:\n')
        random_sampled = translate(vocab, model._random_sample(1000))
        for s in random_sampled:
            print(s)

        _, val_pairs, _ = preprocess.prepare_data(conf)
        s1 = random.choice(val_pairs)[0]
        s2 = random.choice(val_pairs)[0]

        x1, x1_lens, _ = preprocess._btmcd(vocab, [(s1, s1)], conf)
        x2, x2_lens, _ = preprocess._btmcd(vocab, [(s2, s2)], conf)

        print('\n### Linear Interpolation ###:\n')
        print(s1)
        interpolated = translate(vocab, model._interpolate(get_z(x1, x1_lens, model),
                                                           get_z(x2, x2_lens, model), 50))
        for s in interpolated:
            print(s)
        print(s2)


if __name__ == '__main__':
    main()
