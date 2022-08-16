import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
import pprint
import math

class QAGANConfig:
    """ QAGAN configuration """
    num_datasets=6
    num_discriminator_layers=3
    dropout=0.1
    disc_true_lambda=0.5
    disc_fake_lambda=0.5
    kld_lambda=0.1
    discriminate_hidden_layers=False
    discriminate_cls=False
    discriminate_cls_sep=False
    use_discriminator=False
    sequence_len=384
    fake_discriminator_warmup_steps=1000
    true_discriminator_every_n_steps=1
    fake_discriminator_every_n_steps=2
    max_steps = 2500000
    anneal = True
    prediction_head = 'linear'
    constrain_hidden_repr = False

    def __repr__(self):
        return self.__str__()

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

        # make sure only one of discriminate_hidden_layers, discriminate_cls,
        # discriminate_cls_sep is true if use_discriminator is true
        if self.use_discriminator:
            assert sum([self.discriminate_hidden_layers, self.discriminate_cls, self.discriminate_cls_sep]) == 1

    # method for item assignment
    def __setitem__(self, key, value):
        setattr(self, key, value)

    # method for item access
    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return pprint.pformat(self.__dict__)

class QAGANPredictions:
    """ QAGAN predictions """

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.idx2key = {0:'loss', 
                        1:'start_logits', 
                        2:'end_logits', 
                        3:'loss_dict', 
                        4:'hidden_states',
                        5:'discriminator_predicted_class'}

    # method for item assignment
    def __setitem__(self, key, value):
        setattr(self, key, value)

    # method for item access
    def __getitem__(self, i):
        return getattr(self, self.idx2key[i])

    def __str__(self):
        return pprint.pformat(self.__dict__)

class DomainDiscriminator(nn.Module):
    """ Domain discriminator for discriminating 
        between different QA dataset domains 
    """

    def __init__(self, num_classes=3, input_size=768,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Encoder-Only Attention module """
    
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.out_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        # q, k, v: batch_size x len_q x hidden_size
        batch_size = q.size(0)
        len_q = q.size(1)
        len_k = k.size(1)
        # perform attention on every head
        head_q = self.q_linear(q).view(batch_size, len_q, self.num_heads, self.head_dim).transpose(1, 2)
        head_k = self.k_linear(k).view(batch_size, len_k, self.num_heads, self.head_dim).transpose(1, 2)
        head_v = self.v_linear(v).view(batch_size, len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # head_q, head_k, head_v: batch_size x num_heads x len_q x head_dim
        # compute attention score
        scores = torch.matmul(head_q, head_k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, len_k, len_q)
            scores = scores.masked_fill(mask, -1e9)
        # compute attention probability
        weights = F.softmax(scores, dim=-1)
        # apply dropout
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        # compute context vector
        contexts = torch.matmul(weights, head_v)
        # apply final linear layer
        contexts = contexts.transpose(1, 2).contiguous().view(batch_size, len_q, self.hidden_size)

        out = self.out_linear(contexts)
        return out

class TransformersEncoder(nn.Module):
    """ Transformers encoder layer """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(TransformersEncoder, self).__init__()
        self.attention = MultiHeadedSelfAttention(num_heads, hidden_size, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: batch_size x len_q x hidden_size
        # mask: batch_size x len_q x len_k
        # compute attention and residual connection
        x = self.attention(x, x, x, mask) + x
        # apply layer normalization
        x = self.norm1(x)
        # apply dropout
        x = self.dropout1(x)
        # apply feed forward and residual connection
        x = self.feed_forward(x) + x
        # apply layer normalization
        x = self.norm2(x)
        # apply dropout
        x = self.dropout2(x)
        return x
        
class TransformersDecoder(nn.Module):
    """ Transformers decoder layer """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(TransformersDecoder, self).__init__()
        self.self_attention = MultiHeadedSelfAttention(num_heads, hidden_size, dropout)
        self.cross_attention = MultiHeadedSelfAttention(num_heads, hidden_size, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_out, mask=None):
        # x: batch_size x len_q x hidden_size
        # encoder_out: batch_size x len_k x hidden_size
        # mask: batch_size x len_q x len_k
        # compute attention
        x = self.self_attention(x, x, x, mask)
        # add residual connection
        x = x + x
        # apply layer normalization
        x = self.norm1(x)
        # apply dropout
        x = self.dropout1(x)
        # compute attention and residual connection
        x = self.cross_attention(q=x, k=encoder_out, v=encoder_out, mask=mask) + x
        # apply layer normalization
        x = self.norm2(x)
        # apply dropout
        x = self.dropout2(x)
        # apply feed forward and residual connection
        x = self.feed_forward(x) + x
        # apply layer normalization
        x = self.norm3(x)
        # apply dropout
        x = self.dropout3(x)

        return x

class QAPredictionHead(nn.Module):
    """ Prediction head for generating start and end logits """

    def __init__(self, input_size, hidden_size=64, num_outputs=2):
        super(QAPredictionHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.linear = nn.Linear(input_size, hidden_size)
        self.qa_logits = nn.Linear(hidden_size, num_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.qa_logits.weight)
        nn.init.zeros_(self.qa_logits.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.qa_logits(x)
        return x

class QAConditionalPredictionHead(nn.Module):
    """ Conditional Linear Prediction Head """
    
    def __init__(self, input_size, hidden_size=64, logit_size=1):
        super(QAConditionalPredictionHead, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.qa_start_logit = nn.Linear(hidden_size, logit_size)
        self.qa_end_logit = nn.Linear(hidden_size+1, logit_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.qa_start_logit.weight)
        nn.init.zeros_(self.qa_start_logit.bias)
        nn.init.xavier_uniform_(self.qa_end_logit.weight)
        nn.init.zeros_(self.qa_end_logit.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_start = self.qa_start_logit(x)
        x_end = self.qa_end_logit(torch.cat([x, x_start], dim=-1))
        # concatenate start and end logits
        logits = torch.cat([x_start, x_end], dim=-1)
        return logits

class QAConditionalAttPredictionHead(nn.Module):
    """ Conditional prediction head for generating start and end logits.
        hidden state is passed through a the transformers encoder layer, which is 
        then used to generate start logit. Output of this encoder
        is then concatenated with the hidden state and passed through another
        transformers encoder to generate end logit. 
    """
    
    def __init__(self, hidden_size=64, logit_size=1):
        super(QAConditionalAttPredictionHead, self).__init__()
        self.hidden_size = hidden_size
        self.attn_start = TransformersEncoder(num_heads=8, hidden_size=hidden_size)
        self.attn_end = TransformersEncoder(num_heads=8, hidden_size=2*hidden_size)
        self.qa_start_logit = nn.Linear(hidden_size, logit_size)
        self.qa_end_logit = nn.Linear(2*hidden_size, logit_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.qa_start_logit.weight)
        nn.init.zeros_(self.qa_start_logit.bias)
        nn.init.xavier_uniform_(self.qa_end_logit.weight)
        nn.init.zeros_(self.qa_end_logit.bias)

    def forward(self, x, mask=None):
        x_att_start = self.attn_start(x, mask=mask)
        x_start_logit = self.qa_start_logit(x_att_start)
        x = torch.cat([x, x_att_start], dim=-1)
        x_att_end = self.attn_end(x, mask=mask)
        x_end_logit = self.qa_end_logit(x_att_end)
        # concatenate start and end logits
        logits = torch.cat([x_start_logit, x_end_logit], dim=-1)
        return logits

class QAConditionalTransformersPredictionHead(nn.Module):
    """ Conditional prediction head for generating start and end logits.
        hidden state is passed through a transformers encoder, output of 
        which (hidden states) is then fed to the transformers decoder. Decoder
        is then used to generate start logits hidden vector, this is then
        used along with the encoder hidden vector to generate end logits
        hidden vector. Both start and end logits vector are then fed through an
        additional MLP to generate start and end logits.
    """

    def __init__(self, hidden_size=64, logit_size=1):
        super(QAConditionalTransformersPredictionHead, self).__init__()
        self.encoder = TransformersEncoder(num_heads=8, hidden_size=hidden_size)
        self.decoder = TransformersDecoder(num_heads=8, hidden_size=hidden_size)
        self.qa_start_logit = nn.Linear(hidden_size, logit_size)
        self.qa_end_logit = nn.Linear(hidden_size, logit_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.qa_start_logit.weight)
        nn.init.zeros_(self.qa_start_logit.bias)
        nn.init.xavier_uniform_(self.qa_end_logit.weight)
        nn.init.zeros_(self.qa_end_logit.bias)

    def forward(self, x, mask=None):
        x_encoder = self.encoder(x, mask=mask)
        # use zero query vector for decoding start logits 
        x_start_logit = self.decoder(x, x_encoder, mask=mask)
        start_logit = self.qa_start_logit(x_start_logit)
        # use decoder hidden state and input vector for decoding end logits
        x_end_logit = self.decoder(x+x_start_logit, x_encoder, mask=mask)
        end_logit = self.qa_end_logit(x_end_logit)
        # concatenate start and end logits
        logits = torch.cat([start_logit, end_logit], dim=-1)
        return logits
        
class QAGAN(nn.Module):
    """ QAGAN model """

    def __init__(self, config):
        super(QAGAN, self).__init__()
        self.config = config
        self.backbone = config.backbone
        self.tokenizer = config.tokenizer
        self.sequence_len = config.sequence_len
        self.hidden_size = self.backbone.config.hidden_size
        self.anneal = config.anneal
        self.max_steps = config.max_steps
        self.disc_step = 0

        # define prediction head
        if config.prediction_head == 'linear':
            self.qa_outputs = QAPredictionHead(self.hidden_size)
        elif config.prediction_head == 'conditional_linear':
            self.qa_outputs = QAConditionalPredictionHead(self.hidden_size)
        elif config.prediction_head == 'conditional_attention':
            self.qa_outputs = QAConditionalAttPredictionHead(self.hidden_size)
        elif config.prediction_head == 'conditional_transformers':
            self.qa_outputs = QAConditionalTransformersPredictionHead(
                                input_size=self.sequence_len, hidden_size=self.hidden_size)
        else:
            raise ValueError('Invalid prediction head type')

        # define input size
        input_size = self.hidden_size
        if config.discriminate_cls:
            input_size = self.hidden_size
        elif config.discriminate_cls_sep:
            input_size = self.hidden_size * 2
        elif config.discriminate_hidden_layers:
            input_size = self.hidden_size * config.sequence_len

        # idefine discriminator
        self.discriminator = DomainDiscriminator(num_classes=config.num_datasets,
                                                 input_size=input_size,
                                                 hidden_size=self.hidden_size,
                                                 num_layers=config.num_discriminator_layers,
                                                 dropout=config.dropout)
        # other hyperparameters
        self.num_classes = config.num_datasets
        self.disc_true_lambda = config.disc_true_lambda
        self.disc_fake_lambda = config.disc_fake_lambda
        self.discriminate_cls_sep = config.discriminate_cls_sep
        self.sep_id = self.tokenizer.sep_token_id

    # only for prediction
    def forward(self, input_ids, attention_mask,
                start_positions=None, end_positions=None, 
                labels=None, discriminator=False,
                return_hidden_states=False):

        # forward pass to get logits predictions and loss
        def forward_qa(input_ids, attention_mask,
                       start_positions=None, end_positions=None):
            sequence_output = self.backbone(input_ids, attention_mask).last_hidden_state
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            # set loss to zero
            qa_loss = torch.tensor(0.0, dtype=torch.float, device=input_ids.device)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            ignored_index = start_logits.size(1)
            if (start_positions is not None) and (end_positions is not None):
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                # compute negative log-likelihood loss
                loss_fct = nn.NLLLoss(ignore_index=ignored_index)
                start_log_softmax = nn.LogSoftmax(dim=1)(start_logits)
                end_log_softmax = nn.LogSoftmax(dim=1)(end_logits)
                start_loss = loss_fct(start_log_softmax, start_positions)
                end_loss = loss_fct(end_log_softmax, end_positions)
                qa_loss = (start_loss + end_loss) / 2.0

            return qa_loss, start_logits, end_logits, sequence_output

        # get logits predictions and loss
        qa_loss, start_logits, end_logits, sequence_output = \
            forward_qa(input_ids, attention_mask, start_positions, end_positions)

        # set up loss dictionary
        loss_dict = {'NLL': qa_loss.item()}
        total_loss = qa_loss

        # constrain hidden representation to conform
        # to a normal distribution
        if self.config.constrain_hidden_repr:
            normal_dist = self.sample_gaussian(sequence_output[:, 0].size()).to(sequence_output.device)
            normal_prob = torch.sigmoid(normal_dist)
            seq_prob = F.log_softmax(sequence_output[:, 0], dim=-1)
            # kld_loss = self.kl_div(sequence_output, normal_dist)
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            kld_loss = self.config.kld_lambda * kl_criterion(seq_prob, normal_prob) / normal_prob.size(1)
            total_loss += kld_loss
            loss_dict['KLD'] = kld_loss.item()

        # discriminator
        if self.config.use_discriminator and \
           (start_positions is not None) and \
           (end_positions is not None):
            # train discriminator with true labels
            if discriminator:
                # train discriminator to predict domain
                disc_loss = self.forward_discriminator_true(
                                        input_ids, sequence_output, labels)
                # add discriminator loss
                if self.disc_step % self.config.true_discriminator_every_n_steps == 0:
                    total_loss += disc_loss
                loss_dict['disc_loss_true'] = disc_loss.item()
                self.disc_step += 1
            else:
                # train LM to fool discriminator
                disc_loss = self.forward_discriminator_fake(
                                        input_ids, sequence_output)
                # add discriminator loss
                if self.disc_step >= self.config.fake_discriminator_warmup_steps and \
                   self.disc_step % self.config.fake_discriminator_every_n_steps == 0:
                    total_loss += disc_loss
                loss_dict['disc_loss_fake'] = disc_loss.item()

        if return_hidden_states:
            return QAGANPredictions(loss=total_loss, 
                                    start_logits=start_logits,
                                    end_logits=end_logits,
                                    loss_dict=loss_dict,
                                    hidden_states=sequence_output.detach())
        else:
            return QAGANPredictions(loss=total_loss, 
                                    start_logits=start_logits,
                                    end_logits=end_logits,
                                    loss_dict=loss_dict)

    def forward_discriminator_true(self, input_ids, sequence_output, labels):
        B, T, H = sequence_output.shape
        # do not update the language model
        with torch.no_grad():
            cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
            if self.config.discriminate_cls_sep:
                sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
                hidden = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
            elif self.config.discriminate_hidden_layers:
                hidden = sequence_output.view(B, -1)  # [b, T*d]
            else:
                hidden = cls_embedding
        log_prob = self.discriminator(hidden.detach().view(B, -1))
        criterion = nn.NLLLoss(reduction='mean')
        loss = self.disc_true_lambda * criterion(log_prob, labels)

        return loss

    def forward_discriminator_fake(self, input_ids, sequence_output, loss='nll'):
        B, T, H = sequence_output.shape
        cls_embedding = sequence_output[:, 0] # [b, d] : [CLS] representation
        if self.config.discriminate_cls_sep:
            sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
            hidden = torch.cat([cls_embedding, sep_embedding], dim=1)
        elif self.config.discriminate_hidden_layers:
            hidden = sequence_output.view(B, -1)
        else:
            hidden = sequence_output[:, 0]  
        log_prob = self.discriminator(hidden)

        if loss == 'nll':
            # set random targets
            targets = torch.randint(0, self.num_classes, (B,), dtype=torch.long, device=input_ids.device)
            criterion = nn.NLLLoss(reduction='mean')
            anneal_rate = self.anneal_tanh()
            loss = self.disc_fake_lambda * criterion(log_prob, targets)
            if self.anneal:
                loss = loss * anneal_rate

            return loss

        elif loss == 'kld':
            targets = torch.ones_like(log_prob) * (1 / self.num_classes)
            # As with NLLLoss, the input given is expected to contain log-probabilities
            # and is not restricted to a 2D Tensor. The targets are given as probabilities
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            anneal_rate = self.anneal_tanh()
            kld_loss = self.disc_fake_lambda * kl_criterion(log_prob, targets)
            if self.anneal:
                kld_loss = kld_loss * anneal_rate

            return kld_loss

        else:
            raise ValueError("Unknown loss type: {}".format(loss))

    def sample_gaussian(self, size):
        # sample from N(0, I)
        z = torch.normal(mean=0.0, std=1.0, size=size)
        return z

    def kl_div(self, p, q):
        """
        Computes the KL divergence between two categorical distributions
        Args:
            p tensor: (batch, dim): p distribution
            q tensor: (batch, dim): q distribution under KL(p || q)
        Return:
            kl: tensor: (batch,) kl between each sample
        """
        log_p = torch.log(p, dim=-1)
        log_q = torch.log(q, dim=-1)
        kl = p * (log_p - log_q)
        kl = kl.sum(dim=-1)
        return kl

    def log_normal(self, x, m, v):
        """
        Computes the elem-wise log probability of a Gaussian and then sum over the
        last dim. Basically we're assuming all dims are batch dims except for the
        last dim.
        Args:
            x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
            m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
            v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance
        Return:
            log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
                each sample. Note that the summation dimension is not kept
        """
        sigma = torch.sqrt(v)
        log_prob_element_wise = -((x - m).pow(2)/(2 * sigma.pow(2))) - 0.5 * np.log(2 * np.pi) - torch.log(sigma)
        log_prob = log_prob_element_wise.sum(dim=-1)

        return log_prob

    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution
        Args:
            h: tensor: (batch, ..., dim, ...): Arbitrary tensor
            dim: int: (): Dimension along which to split the tensor for mean and
                variance
        Returns:
            m: tensor: (batch, ..., dim / 2, ...): Mean
            v: tensor: (batch, ..., dim / 2, ...): Variance
        """
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def get_sep_embedding(self, input_ids, sequence_output):
        batch_size = input_ids.size(0)
        sep_idx = (input_ids == self.sep_id).sum(1)
        sep_embedding = sequence_output[torch.arange(batch_size), sep_idx]
        return sep_embedding

    # save pretrained method
    def save_pretrained(self, path):
        ckpt_path = os.path.join(path, 'qagan.pt')
        # create directory if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        state_dict = self.state_dict()
        torch.save(state_dict, ckpt_path)

    # load pretrained method
    def from_pretrained(self, path):
        ckpt_path = os.path.join(path, 'qagan.pt')
        # load state dict if exist
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path)
            self.load_state_dict(state_dict)
            print("loaded pretrained model from {}".format(path))
        else:
            print("checkpoint '{}' does not exist".format(path))
        return self

    # tanh annealing function
    # this function ramps up from 0 to 1 in tanh(x) between the steps: fake_discriminator_warmup_steps and self.config.max_steps)
    def anneal_tanh(self):
        return ((math.tanh((2.0 * ((self.disc_step * 2.0 - 
                                   ((self.config.max_steps + self.config.fake_discriminator_warmup_steps))
                                   )
                                  ) / (self.config.max_steps - self.config.fake_discriminator_warmup_steps))
                          ) + 1.0) / 2.0)
