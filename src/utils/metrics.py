import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate import bleu_score, meteor_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def calculate_all_metrics(hypothesis, references):
    return {
        'bleus': calculate_bleu_scores(hypothesis, references),
        'meteor': calculate_meteor(hypothesis, references)
        }


def calculate_bleu_scores(hypothesis, references):
    bleu_scores = {}
    for i in range(1, 5):
        w = 1.0 / i
        weights = [w] * i
        bleu_scores['bleu{}'.format(str(i))] = 100 * bleu_score.corpus_bleu(references, hypothesis, weights=weights)
    return bleu_scores


def calculate_meteor(hypothesis, references):
    return 100 * np.mean(
        [meteor_score.single_meteor_score(' '.join(r[0]), ' '.join(h))
            for (r, h) in zip(references, hypothesis)])


def calculate_prf(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance


def plot_confusion_matrix(predictions, targets, classes,
                          epoch, dataset, net_code, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    cm = confusion_matrix(targets, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = 100*np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v = cm[i, j]
        if normalize:
            v = np.round(v, decimals=1)
        plt.text(j, i, v,
                 horizontalalignment="center",
                 color="white" if v > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not os.path.exists('reports/figures/{}'.format(dataset)):
        os.mkdir('reports/figures/{}'.format(dataset))
    plt.savefig('reports/figures/{}/{}-cm_{}'.format(dataset, net_code, epoch))
    plt.close()
