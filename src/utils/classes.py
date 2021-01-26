"""
This module contains classes for different entities in the preprocessing like
Vocabulary, Dataset
"""


class Vocabulary(object):
    """Vocabulary class"""
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4  # count the special tokens above

    def add_sentence(self, sentence):
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.word2count[word] = 1
            self.index2word[self.size] = word
            self.size += 1
        else:
            self.word2count[word] += 1

    def sentence2index(self, sentence):
        indexes = []
        for w in sentence.split():
            try:
                indexes.append(self.word2index[w])
            except KeyError as e:  # handle OOV
                indexes.append(self.word2index['<UNK>'])
        return indexes

    def index2sentence(self, indexes):
        return [self.index2word[i] for i in indexes]

    def filter(self, min_freq):
        exclude_keys = {'<PAD>', '<SOS>', '<EOS>', '<UNK>'}
        # Reindex and filter
        _word2index = {}
        _word2index.update({'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3})
        index = 4
        for w, i in self.word2index.items():
            if w not in exclude_keys and self.word2count[w] > min_freq:
                _word2index[w] = index
                index += 1
        self.word2index = _word2index
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.word2count = {k: v for k, v in self.word2count.items() if k in self.word2index}
