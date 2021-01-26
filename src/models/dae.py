import torch
import random
from tqdm import tqdm
from torch import nn, optim
from models import attn_module
from .fusion_methods import GanFusion, AutoFusion


class AutoEncoder(nn.Module):
    """AutoEncoder model"""
    def __init__(self, config, vocab, embedding_wts):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding_wts = embedding_wts
        self.build_model()

    def build_model(self):
        self.unit = self.config['unit']
        self.device = self.config['device']
        self.sos_idx = self.config['SOS_TOKEN']
        self.pad_idx = self.config['PAD_TOKEN']
        # beam size is 1 by default
        self.beam_size = self.config['beam_size']
        self.hidden_dim = self.config['hidden_dim']
        self.latent_dim = self.config['latent_dim']
        self.embedding_dim = self.config['embedding_dim']
        self.bidirectional = self.config['bidirectional']
        self.enc_dropout = nn.Dropout(self.config['dropout'])
        self.dec_dropout = nn.Dropout(self.config['dropout'])

        if self.config['word_embedding']:
            self.embedding = nn.Embedding.from_pretrained(
                self.embedding_wts, freeze=self.config['freeze_embeddings?'])
        else:
            self.embedding = nn.Embedding(self.vocab.size, self.embedding_dim)

        if self.unit == 'lstm':
            self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                   self.config['enc_n_layers'],
                                   bidirectional=self.bidirectional)

            self.decoder = nn.LSTM(self.embedding_dim, self.latent_dim,
                                   self.config['dec_n_layers'])
        elif self.unit == 'gru':
            self.encoder = nn.GRU(self.embedding_dim, self.hidden_dim,
                                  self.config['enc_n_layers'],
                                  bidirectional=self.bidirectional)

            self.decoder = nn.GRU(self.embedding_dim, self.latent_dim,
                                  self.config['dec_n_layers'])
        else:
            self.encoder = nn.RNN(self.embedding_dim, self.hidden_dim,
                                  self.config['enc_n_layers'],
                                  bidirectional=self.bidirectional)

            self.decoder = nn.RNN(self.embedding_dim, self.latent_dim,
                                  self.config['dec_n_layers'])

        if self.config['attn_model']:
            self.attn = attn_module.Attn(self.config['attn_model'],
                                         self.hidden_dim)

        # All the projection layers
        self.pf = (2 if self.bidirectional else 1)  # project factor

        self.output2vocab = nn.Linear(self.latent_dim + self.embedding_dim,
                                      self.vocab.size)

        self.optimizer = optim.Adam(self.parameters(), self.config['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=15,
                                                   gamma=0.5)
        # Reconstruction loss
        self.rec_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        if config['fusion_type'] == 'gan':
            self.fusion = GanFusion(config)
        elif config['fusion_type'] == 'auto':
            self.fusion = AutoFusion(config, config['latent_dim']*3)
        else:
            self.fusion = None

    def _encode(self, x, x_lens):
        max_x_len, bs = x.shape
        # convert input sequence to embeddings
        embedded = self.enc_dropout(self.embedding(x))
        # embedded => (t x bs x embedding_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, x_lens,
                                                   enforce_sorted=False)
        # Forward pass through the encoder
        # outputs => (max_seq_len, bs, hidden_dim * self.pf)
        # h_n (& c_n) => (#layers * self.pf, bs, hidden_dim)
        outputs, hidden = self.encoder(packed)

        # outputs => (max_seq_len x bs x hidden_dim * self.pf)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Construct z from last time_step output
        if self.bidirectional:
            outputs = outputs.view(max_x_len, bs, self.pf, self.hidden_dim)
            # concatenate forward and backward encoder outputs
            outputs = outputs[:, :, 0, :] + outputs[:, :, 1, :]

            if self.unit == 'lstm':
                # sum forward and backward hidden states
                c_n = hidden[0].view(self.config['enc_n_layers'],
                                     self.pf, bs, self.hidden_dim)
                c_n = c_n[:, 0, :, :] + c_n[:, 1, :, :]

                h_n = hidden[1].view(self.config['enc_n_layers'],
                                     self.pf, bs, self.hidden_dim)
                h_n = h_n[:, 0, :, :] + h_n[:, 1, :, :]
                hidden = (c_n, h_n)
            else:
                # sum forward and backward hidden states
                hidden = hidden.view(self.config['enc_n_layers'],
                                     self.pf, bs, self.hidden_dim)
                hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        # outputs => (max_seq_len x bs x hidden_dim * self.pf)
        # hidden => (#enc_layers, bs, hidden * self.pf)
        return {'encoder_outputs': outputs, 'z': hidden}

    def _create_mask(self, tensor):
        return torch.ne(tensor, self.pad_idx)

    def _decode(self, z, y, infer, encoder_outputs=None, mask=None):
        """
        z -> (#enc_layers, batch_size x latent_dim)
        y -> (max_y_len, batch_size)

        encoder_outputs and mask will be used for attention mechanism
        TODO: Handle bidirectional decoders
        """
        max_y_len, bs = y.shape
        vocab_size = self.vocab.size

        if self.config['dec_mode'] == 'beam':
            # tensor to maintain k candidates
            # this tensor is intialized such that it has all vocab tokens as
            # candidates for the k beams but at each time step, we will
            # update the entire tensor with the best k candidates subsequences
            # at that time.
            # candidate_subseq => (max_y_len, bs*k*vocab)
            candidate_subseq = torch.arange(vocab_size).repeat(
                bs*self.beam_size).unsqueeze(0).repeat(max_y_len, 1).to(
                self.device)
            candidate_subseq[0] = torch.ones(
                vocab_size*bs*self.beam_size)*self.sos_idx
            # tensor to store decoder outputs
            decoder_outputs = (torch.ones(max_y_len, bs*self.beam_size,
                               vocab_size)*self.sos_idx).to(self.device)
        else:
            # tensor to store decoder outputs
            decoder_outputs = (torch.ones(max_y_len, bs*self.beam_size,
                               vocab_size)*self.sos_idx).to(self.device)

        # Reconstruct the hidden vector from z
        # if #dec_layers > #enc_layers, use currently obtained hidden
        #  to represent the last #enc_layers layers of decoder hidden
        if self.config['dec_n_layers'] > self.config['enc_n_layers']:
            raise \
                NotImplementedError("(dec_n_layers > enc_n_layers) Decoder " +
                                    "can't have more layers than the encoder")
        else:
            if self.unit == 'lstm':
                try:
                    # this will throw an error when z comes from an encoder
                    # other DAE's (VAE, in this case)
                    dec_hidden = (
                        z[0][-self.config['dec_n_layers']:, :, :],
                        z[1][-self.config['dec_n_layers']:, :, :]
                        )
                except Exception as e:
                    # consider h_n to be zero
                    dec_hidden = (
                        torch.zeros(self.config['dec_n_layers'], bs,
                                    self.latent_dim).to(self.device),
                        z[-self.config['dec_n_layers']:, :, :]
                        )
                # doesn't make a different for beam_size = 1
                dec_hidden = (
                    dec_hidden[0].repeat(1, self.beam_size, 1),
                    dec_hidden[1].repeat(1, self.beam_size, 1)
                    )
            else:
                dec_hidden = z[-self.config['dec_n_layers']:, :, :]
                dec_hidden = dec_hidden.repeat(1, self.beam_size, 1)

            # dec_hidden => (#dec_layers, bs*beam_size, latent_dim)

        # initial decoder input is <sos> token
        # ouptut -> (bs)
        output = y[0, :]
        # We maintain a beam of k responses instead of just one
        # Note that for modes other beam search, the below doesn't make a
        # difference
        # output -> (beam_size*bs)
        output = output.repeat(self.beam_size, 1).view(-1)

        # Start decoding process
        for t in range(1, max_y_len):
            # output -> (beam_size*bs, vocab_size)
            # dec_hidden -> (beam_size*bs, hidden * self.pf)
            output, dec_hidden = self._decode_token(output, dec_hidden, mask)
            decoder_outputs[t] = output
            do_tf = random.random() < self.config['tf_ratio']
            if infer or (not do_tf):
                if self.config['dec_mode'] == 'beam':
                    # split beam and bs dim from dec_outputs
                    gold_logits = decoder_outputs[:t+1].view(t+1, bs,
                                                             self.beam_size,
                                                             vocab_size)
                    # run a softmax on the last dimension to get scores
                    candidate_scores = torch.log_softmax(gold_logits, dim=-1)
                    # extract probabilities of the tokens and flatten logits
                    new_logits = new_logits[-1].squeeze(0).view(bs, -1)
                    # new_logits => (bs, beam_size*vocab_size)
                    # .topk returns scores and tokens of shape (bs, k)
                    # the modulo operator is required to get real token ids
                    # as its range would be beam_size*vocab_size otherwise
                    output = new_logits.topk(k=self.beam_size, dim=-1)[1].view(
                        -1) % self.vocab.size
                    # output => (bs*beam_size)
                    if y_specs is not None:
                        # replace all the tokens at time t with the guessed
                        # token for k beams
                        candidate_subseq[t] = output.repeat(vocab_size)
                else:
                    # output.max(1) -> (scores, tokens)
                    # doing a max along `dim=1` returns logit scores and
                    # token index for the most probable (max valued) token
                    # scores (& tokens) -> (bs)
                    output = output.max(dim=1)[1]  # greedy search
            elif do_tf:
                output = y[t].repeat(self.beam_size, 1).view(-1)
        return decoder_outputs

    def _decode_token(self, input, hidden, mask):
        """
        input -> (beam_size*bs)
        hidden -> (#dec_layers * self.pf x bs x hidden_dim)
                  (c_n is zero for lstm decoder)
        mask -> (bs x max_x_len)
            mask is used for attention
        """
        input = input.unsqueeze(0)
        # input -> (1, beam_size*bs)

        # embedded -> (1, beam_size*bs, embedding_dim)
        embedded = self.dec_dropout(self.embedding(input))
        # output -> (1, beam_size*bs, hidden_dim) (decoder is unidirectional)
        # h_n (and c_n) -> (1, beam_size*bs, hidden)
        output, hidden = self.decoder(embedded, hidden)
        output = self.output2vocab(torch.cat((output, embedded), dim=-1))
        # output -> (beam_size*bs, vocab_size)
        return output.squeeze(0), hidden

    def _attend(self, dec_output, enc_output):
        # TODO: Review for beam search compatibility
        # Get attention weights
        attn_weights = self.attn(dec_output, enc_output)
        # Get weighted sum
        context = attn_weights.bmm(enc_output.transpose(0, 1))
        # Concatenate weighted context vector and rnn output using Luong eq. 5
        dec_output = dec_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((dec_output, context), 1)
        concat_output = torch.tanh(self.attn.W_cat(concat_input))
        # Predict next word (Luong eq. 6)
        dec_output = self.output2vocab(concat_output)
        return dec_output, context

    def forward(self, x, x_lens, y=None, stage=None):
        infer = (y is None)
        if infer:  # model in val/test mode
            self.eval()
            y = torch.zeros(
                (self.config['MAX_LENGTH'], x['text'].shape[1])).long().fill_(
                 self.sos_idx).to(self.device)
        else:  # train mode
            self.train()
            self.optimizer.zero_grad()

        # z is the final forward and backward hidden state of all layers
        z_t = self._encode(x['text'], x_lens)['z']
        z_s = x['speech']
        z_v = x['video']

        # fusion (currently shows the best performing GAN-Fusion, AutoFusion
        # implementation is trivial, just remove GAN-specific operations
        fusion_dict = self.fusion({'z_t': z_t, 'z_s': z_s, 'z_v': z_v},
                                  mode='train' if not infer else 'infer',
                                  stage)

        z_fuse = self.fusion_dict['z']
        loss = self.fusion_dict['loss']

        # decoder_outputs -> (max_y_len, bs*beam_size, vocab_size)
        decoder_outputs = self._decode(z_fuse, y, infer)

        # loss calculation and backprop
        loss += self.rec_loss(
            decoder_outputs[1:].view(-1, decoder_outputs.shape[-1]),
            y.repeat(1, self.beam_size)[1:].view(-1))

        if not infer:
            loss.backward()
            # Clip gradients (wt. update) (very important)
            nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])
            self.optimizer.step()

            if 'disc' in stage:
                self.fusion.text_gan.discriminator.optimize(self.config['clip'])
                self.fusion.speech_gan.discriminator.optimize(self.config['clip'])
                self.fusion.video_gan.discriminator.optimize(self.config['clip'])
            else:
                self.fusion.text_gan.generator.optimize(self.config['clip'])
                self.fusion.speech_gan.generator.optimize(self.config['clip'])
                self.fusion.video_gan.generator.optimize(self.config['clip'])

        return {'pred_outputs': decoder_outputs, 'loss': loss}
