import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
import pprint

class QAGANConfig:
    """ QAGAN configuration """
    num_datasets=3
    num_layers=3
    dropout=0.1
    dis_lambda=0.5
    concat=False
    use_discriminator=False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return pprint.pformat(self.__dict__)

class QAGANPredictions:
    """ QAGAN predictions """

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.idx2key = {0:'loss', 1:'start_logits', 2:'end_logits', 3:'loss_dict'}

    def __getitem__(self, i):
        return getattr(self, self.idx2key[i])

class DomainDiscriminator(nn.Module):
    """ Domain discriminator for discriminating 
        between different QA dataset domains 
    """

    def __init__(self, num_classes=3, input_size=768 * 2,
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

class QAGAN(nn.Module):
    """ QAGAN model """

    def __init__(self, config):
        super(QAGAN, self).__init__()
        self.config = config
        self.backbone = config.backbone
        self.tokenizer = config.tokenizer
        self.hidden_size = self.backbone.config.hidden_size

        # define prediction head
        self.qa_outputs = QAPredictionHead(self.hidden_size)
        # define discriminator
        input_size = self.hidden_size if not config.concat else 2 * self.hidden_size
        self.discriminator = DomainDiscriminator(num_classes=config.num_datasets,
                                                 input_size=input_size,
                                                 hidden_size=self.hidden_size,
                                                 num_layers=config.num_layers,
                                                 dropout=config.dropout)
        # other hyperparameters
        self.num_classes = config.num_datasets
        self.dis_lambda = config.dis_lambda
        self.concat = config.concat
        self.sep_id = self.tokenizer.sep_token_id

    # only for prediction
    def forward(self, input_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None):

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
        # discriminator
        if self.config.use_discriminator and \
           (start_positions is not None) and (end_positions is not None):
            # train discriminator to predict domain
            disc_true = self.forward_discriminator_true(
                                    input_ids, sequence_output, labels)

            # train LM to fool discriminator
            disc_fake = self.forward_discriminator_fake(
                                    input_ids, sequence_output)

            # add discriminator loss
            total_loss += self.dis_lambda * (disc_true + disc_fake)
            loss_dict['disc_true_loss'] = disc_true.item()
            loss_dict['disc_fake_loss'] = disc_fake.item()

        return QAGANPredictions(loss=total_loss, 
                                start_logits=start_logits,
                                end_logits=end_logits,
                                loss_dict=loss_dict)

    def forward_discriminator_true(self, input_ids, sequence_output, labels):
        with torch.no_grad():
            cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
            if self.concat:
                sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
                hidden = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
            else:
                hidden = cls_embedding
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = self.dis_lambda * criterion(log_prob, labels)

        return loss

    def forward_discriminator_fake(self, input_ids, sequence_output):
        cls_embedding = sequence_output[:, 0] # [b, d] : [CLS] representation
        if self.concat:
            sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
            hidden = torch.cat([cls_embedding, sep_embedding], dim=1)
        else:
            hidden = sequence_output[:, 0]  
        log_prob = self.discriminator(hidden)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor. The targets are given as probabilities
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        kld_loss = self.dis_lambda * kl_criterion(log_prob, targets)

        return kld_loss

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