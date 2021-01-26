# the empowering backend
from torch import nn, optim

# for beautiful outputs somewhere
from pprint import pprint
from tensorboardX import SummaryWriter

# the real deal
from models.config import config
from utils import train_utils, model_utils, metrics


def main():
    # Initialize tensorboardX writer
    writer = SummaryWriter()
    pprint(config)

    # Load necessary data and hyperparams
    vocab, train_pairs, val_pairs, test_pairs, embedding_wts = \
        train_utils.load_data(config)

    batch_size, clip, device, num_gpus = config['batch_size'], config['clip'],\
        config['device'], config['num_gpus']

    # Instantiate model
    # pass vocab_size instead of embedding_wts below to randomly initialize
    # word-embeddings and train them along with the model
    epoch, model = train_utils.build_model(config,
                                           device=device,
                                           embedding_wts=embedding_wts)

    print(model)

    optimizer = optim.Adam(model.parameters(), config['lr'])
    if config['use_scheduler?']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1,
            gamma=config['gamma'],
            last_epoch=-1)

    criterion = nn.CrossEntropyLoss(ignore_index=config['PAD_TOKEN'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total {} trainable parameters'.format(num_params))

    print('Training started..')
    train_iter, val_iter, test_iter = 0, 0, 0
    for e in range(epoch, config['n_epochs']):
        # TRAIN
        _ = train_utils.train_epoch(model, criterion, optimizer,
                                    vocab, train_pairs, batch_size,
                                    clip, config, writer, e)

        # VALIDATE
        _, preds, y_val = train_utils.validate_epoch(model, criterion,
                                                     vocab, val_pairs,
                                                     batch_size, config,
                                                     writer, e)
        if config['use_scheduler?']:
            scheduler.step()
        scores = metrics.calculate_prf(preds, y_val)
        writer.add_scalars('metrics/val/', scores, e)
        metrics.plot_confusion_matrix(preds, y_val, sorted(config['classes']),
                                      e, config['dataset'],
                                      'bilstm-classifier-val')
        pprint(scores)
        model_utils.save_snapshot(model, e)

        # TEST every 5 epochs
        if e > 0 and (e+1) % 5 == 0:
            _, preds, y_test = train_utils.validate_epoch(model, criterion,
                                                          vocab, test_pairs,
                                                          batch_size, config,
                                                          writer, e)
            scores = metrics.calculate_prf(preds, y_test)
            writer.add_scalars('metrics/test/', scores, e)
            metrics.plot_confusion_matrix(preds, y_test,
                                          sorted(config['classes']),
                                          e, config['dataset'],
                                          'bilstm-classifier-test')
            pprint(scores)
    writer.close()


if __name__ == '__main__':
    main()
