import numpy as np
from nltk.translate import bleu_score, meteor_score


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
