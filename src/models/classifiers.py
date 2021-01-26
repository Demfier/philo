"""
This module contains all the classifiers
"""
import torch
from torch import nn
from torch.nn import functional as F

from .fusion_methods import AutoFusion, GanFusion


class BiModalEmotionClassifier(nn.Module):
    """
    Bimodal emotion classifier.
    NOTE: Although it uses both text and audio, the feature vector
    from audio is already calculated. Appropriate audio encoder will be
    added later.
    """
    def __init__(self, config, vocab_size=None, embedding_wts=None):
        """
        Initializes parameters for the bimodal classifier.
        Parameters:
        =================
        config (Dict):
            config dictionary with hyperparams. This would, by default,
            contain values from ./src/models/config.py
        embedding_wts (Dict) (optional):
            Dictionary with embedding wt. matrices of concerened languages.
            This only needs to passed if use of pretrained word-embeddings
            is desired. If not, pass vocab_size (See below)
        vocab_size (Int) (optional):
            Vocabulary size of the corpus. This only needs to be passed if
            randomly initialized word-embeddings are to be trained along
            with the model.
        """
        super(BiModalEmotionClassifier, self).__init__()
        self.config = config

        # init parameters for text
        self.embedding = {}
        self.embedding['en'] = \
            nn.Embedding.from_pretrained(embedding_wts['en'], freeze=True) \
            if embedding_wts \
            else nn.Embedding(vocab_size, config['embedding_dim'])

        # fix rnn type to bidirectional LSTM
        self.text_encoder = nn.LSTM(config['embedding_dim'],
                                    config['hidden_dim'],
                                    num_layers=config['enc_n_layers'],
                                    dropout=config['dropout'],
                                    batch_first=True,
                                    bidirectional=config['bidirectional'])

        self.use_attn = False if config['attn_model'] is None else True
        # input dimension is 2*hidden_dim when we are using a bidirectional
        # lstm cell, we concatenate hidden features from the two directions
        # before passing it through this output layer
        self.num_dirs = 2 if config['bidirectional'] else 1

        if self.use_attn:
            self.combined_ht = nn.Sequential(
                nn.Linear(2*self.num_dirs*self.config['hidden_dim'],
                          self.num_dirs*self.config['hidden_dim'],
                          bias=False),
                nn.ReLU()
                )

        # init audio parameters
        self.audio_encoder = nn.Sequential(
            nn.Linear(config['audio_dim'][0], 500),
            nn.ReLU(),
            nn.Linear(500, config['audio_dim'][1]))

        self.final_in_features = config['audio_dim'][1] +\
            self.num_dirs*config['hidden_dim']

        # declare the fusion layer parameters
        if config['fusion_type'] == 'auto':
            self.fusion_method = AutoFusion(config, self.final_in_features)
            # self.final_in_features += config['latent_dim']
        elif config['fusion_type'] == 'gan':
            self.fusion_method = GanFusion(config)
        else:
            self.fusion_method = None

        self.out = nn.Linear(config['latent_dim'], config['output_dim'])


    def fuse(self, z_dict, mode=None, stage=None):
        """
        Input:
        ======
        z_dict for AutoFusion and no fusion will have only one latent code:
        the concatenation of all z from all modalities

        z_dict for GanFusion will have the individual modalities z separated
        as we'll need to define a GanFusion module for every modality

        Returns:
        ========
        an output dictionary with logits and aux_loss
        """
        if self.fusion_method is None:
            return {'logits': self.out(z_dict['z']), 'aux_loss': None}
        elif isinstance(self.fusion_method, AutoFusion):
            fusion_output_dict =  self.fusion_method(z_dict['z'])
            z_fuse = fusion_output_dict['z']
            aux_loss = fusion_output_dict['loss']
            return {
                'logits': self.out(z_fuse),
                'aux_loss': aux_loss
            }
        elif isinstance(self.fusion_method, GanFusion):
            fusion_output_dict = self.fusion_method(z_dict['z'], mode, stage)
            z_fuse =
            return {
                'logits': self.out(z_fuse),
                'aux_loss': aux_loss
            }
        else:
            raise NotImplementedError


    def attn(self, hidden_state, encoder_outputs):
        """
        Implements global attention mechanism from Luong et al. Refer to
        the original paper to understand more:
        https://arxiv.org/pdf/1508.04025.pdf

        returns the context vector (attended encoder_outputs)
        Parameters:
        ==========
        hidden_state {bs, h} (torch.Tensor):
            hidden state for the final time step. Denoted by h_t in the paper
        encoder_outputs {bs, n, h} (torch.Tensor):
            hidden states for each time step t. Denoted by h_s in the paper

        Returns:
        ========
        context_vector {bs, h} (torch.Tensor):
            weighted average over all hidden states (encoder_outputs) based on
            the alignment vector a_t (soft_attn)
        """
        # denoted by score in the paper. we use the dot product attention
        # (bs, n, 1) => (bs, n, h)(bs, h, 1)
        attn_weights = torch.bmm(encoder_outputs, hidden_state.unsqueeze(2))
        # this is denoted by a_t (alignment vector) in the paper
        # NOTE: we haven't squeezed dim 2 in attn_weights as we'll to unsqueeze
        # it again when calculating the final context vector
        soft_attn = F.softmax(attn_weights, dim=1)
        # weighted average over all hidden states
        # (bs, h) => (bs, h, n)(bs, n, 1)
        context_vector = torch.bmm(encoder_outputs.permute(0, 2, 1),
                                   soft_attn).squeeze(2)
        return context_vector

    def forward(self, x, mode=None, stage=None):
        audio_features = self.audio_encoder(x['audio'])
        text_embeddings = self.embedding['en'](x['text'])
        # no need of packed embeddings for classification
        text_output, (text_hidden, _) = self.text_encoder(text_embeddings)
        if self.config['bidirectional']:
            # concatenate f/w and b/w hidden vectors
            text_hidden = torch.cat((text_hidden[0], text_hidden[1]), dim=1)

        if not self.use_attn:
            concat_z = torch.cat((text_hidden, audio_features), dim=-1)
        else:
            attended_vector = self.attn(text_hidden, text_output)
            new_ht = self.combined_ht(torch.cat(
                (attended_vector, text_hidden),
                dim=-1)
            )
            concat_z = torch.cat((new_ht, audio_features), dim=-1)

        return self.fuse({'z': concat_z}, mode, stage)

class AudioEmotionClassifier(nn.Module):
    """
    Audio-only emotion classifier
    It only uses features from audio modality for emotion classification
    """
    def __init__(self, config):
        """
        Initializes parameters for the bimodal classifier.
        Parameters:
        =================
        config (Dict):
            config dictionary with hyperparams. This would, by default,
            contain values from ./src/models/config.py
        """
        super(AudioEmotionClassifier, self).__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config['audio_dim'], 500),
            nn.ReLU(),
            nn.Linear(500, 300))
        self.out = nn.Linear(300, config['output_dim'])

    def forward(self, x):
        """
        Parameters:
        ===========
        x (Dict):
            Input dictionary containing data from the different modaltites
            For instance, x['audio'] contains input audio features
        IMPORTANT:
        ==========
        Shapes of all the tensors being used in this pass
        a) x['audio'].shape => [100, 8] (bs, audio_dim)
        """
        return {
            'logits': self.out(self.encoder(x['audio'])),
            'aux_loss': None
            }


class TextEmotionClassifier(BiModalEmotionClassifier):
    """
    Text emotion classifier.
    """
    def __init__(self, config, vocab_size=None, embedding_wts=None):
        """
        Initializes parameters for the text classifier.
        Parameters:
        =================
        config (Dict):
            config dictionary with hyperparams. This would, by default,
            contain values from ./src/models/config.py
        embedding_wts (Dict) (optional):
            Dictionary with embedding wt. matrices of concerened languages.
            This only needs to passed if use of pretrained word-embeddings
            is desired. If not, pass vocab_size (See below)
        vocab_size (Int) (optional):
            Vocabulary size of the corpus. This only needs to be passed if
            randomly initialized word-embeddings are to be trained along
            with the model.
        """
        super(TextEmotionClassifier, self).__init__(config, vocab_size,
                                                    embedding_wts)
        del self.audio_encoder
        self.out = nn.Linear(self.num_dirs*config['hidden_dim'],
                             config['output_dim'])

    def forward(self, x):
        """
        Parameters:
        ===========
        x (Dict):
            Input dictionary containing data from the different modaltites
            For instance, x['text'] contains input token sequence
        IMPORTANT:
        ==========
        Shapes of all the tensors being used in this pass:
        same as in the forward function of BimodalEmotionClassifier
        a) text_embeddings.shape => [100, 32, 300] (b, t, e)
        b) text_hidden.shape (before torch.cat) => [2*1, 100, 256] (2*1, b, h)
        c) text_hidden.shape (after torch.cat) => [100, 512] (b, h)
        c) self.out(text_hidden) (raw logits) => [100, 6] (b, output_dims)
        """
        text_embeddings = self.embedding['en'](x['text'])  # (bs, t, e)
        # ignore output as not using attention currently and cell state
        text_output, (text_hidden, _) = self.text_encoder(text_embeddings)
        if self.config['bidirectional']:
            # concatenate f/w and b/w hidden vectors
            text_hidden = torch.cat((text_hidden[0], text_hidden[1]), dim=1)
        if not self.use_attn:
            return {
                'logits': self.out(text_hidden),
                'aux_loss': None
                }

        attended_vector = self.attn(text_hidden, text_output)
        new_ht = self.combined_ht(torch.cat(
            (attended_vector, text_hidden),
            dim=-1)
        )
        return {
            'logits': self.out(new_ht),
            'aux_loss': None
            }
