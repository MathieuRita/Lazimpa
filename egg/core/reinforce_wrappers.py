# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
import numpy as np


from .transformer import TransformerEncoder, TransformerDecoder
from .rnn import RnnEncoder, RnnEncoderImpatient
from .util import find_lengths


class ReinforceWrapper(nn.Module):
    """
    Reinforce Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs. During training, the wrapper
    transforms them into a tuple of (sample from the multinomial, log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """
    def __init__(self, agent):
        super(ReinforceWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy


class ReinforceDeterministicWrapper(nn.Module):
    """
    Simple wrapper that makes a deterministic agent (without sampling) compatible with Reinforce-based game, by
    adding zero log-probability and entropy values to the output. No sampling is run on top of the wrapped agent,
    it is passed as is.
    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceDeterministicWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4, 3])
    >>> (log_prob == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    """
    def __init__(self, agent):
        super(ReinforceDeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)

        return out, torch.zeros(1).to(out.device), torch.zeros(1).to(out.device)


class SymbolGameReinforce(nn.Module):
    """
    A single-symbol Sender/Receiver game implemented with Reinforce.
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0):
        """
        :param sender: Sender agent. On forward, returns a tuple of (message, log-prob of the message, entropy).
        :param receiver: Receiver agent. On forward, accepts a message and the dedicated receiver input. Returns
            a tuple of (output, log-probs, entropy).
        :param loss: The loss function that accepts:
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs the end-to-end loss. Can be non-differentiable; if it is differentiable, this will be leveraged
        :param sender_entropy_coeff: The entropy regularization coefficient for Sender
        :param receiver_entropy_coeff: The entropy regularizatino coefficient for Receiver
        """
        super(SymbolGameReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.sender_entropy_coeff = sender_entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, sender_input, labels, receiver_input=None):
        message, sender_log_prob, sender_entropy = self.sender(sender_input)
        receiver_output, receiver_log_prob, receiver_entropy = self.receiver(message, receiver_input)

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        policy_loss = ((loss.detach() - self.mean_baseline) * (sender_log_prob + receiver_log_prob)).mean()
        entropy_loss = -(sender_entropy.mean() * self.sender_entropy_coeff + receiver_entropy.mean() * self.receiver_entropy_coeff)

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        for k, v in rest_info.items():
            if hasattr(v, 'mean'):
                rest_info[k] = v.mean().item()

        rest_info['baseline'] = self.mean_baseline
        rest_info['loss'] = loss.mean().item()
        rest_info['sender_entropy'] = sender_entropy.mean()
        rest_info['receiver_entropy'] = receiver_entropy.mean()

        return full_loss, rest_info


class RnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, num_layers=1, cell='rnn', force_eos=True):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """
        super(RnnSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        prev_hidden = [self.agent(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            # ATTENTION ENLEVER LAJOUT
            #if step==0:
            #    step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)-1000*torch.cat((torch.zeros((h_t.size(0),1)),torch.ones((h_t.size(0),int(self.vocab_size/2))),torch.zeros((h_t.size(0),int(self.vocab_size/2)))),dim=1).to("cuda")
            #else:
            #    step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)-1000*torch.cat((torch.zeros((h_t.size(0),1)),torch.zeros((h_t.size(0),int(self.vocab_size/2))),torch.ones((h_t.size(0),int(self.vocab_size/2)))),dim=1).to("cuda")
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy


class RnnReceiverReinforce(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game. The wrapper logic feeds the message into the cell
    and calls the wrapped agent on the hidden state vector for the step that either corresponds to the EOS input to the
    input that reaches the maximal length of the sequence.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverReinforce, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        sample, logits, entropy = self.agent(encoded, input)

        return sample, logits, entropy

class RnnReceiverCompositionality(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game with several attributes (for compositionality experiments).
    RnnReceiverCompositionality is equivalent to RnnReceiverReinforce but treated each attribute independently.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size,max_len,n_attributes, n_values, cell='rnn', num_layers=1):
        super(RnnReceiverCompositionality, self).__init__()
        self.agent = agent
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)
        self.hidden_to_output = nn.Linear(hidden_size, n_attributes*n_values)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        logits = F.log_softmax(self.hidden_to_output(encoded).reshape(encoded.size(0),self.n_attributes,self.n_values), dim=2)
        #entropy=-torch.exp(logits)*logits
        entropy=[]
        slogits= []
        for i in range(logits.size(1)):
          distr = Categorical(logits=logits[:,i,:])
          entropy.append(distr.entropy())
          x = distr.sample()
          slogits.append(distr.log_prob(x))

        entropy = torch.stack(entropy).permute(1, 0)
        slogits = torch.stack(slogits).permute(1, 0)

        return logits, slogits, entropy

class RnnReceiverDeterministic(nn.Module):
    """
    Reinforce Wrapper for a deterministic Receiver in variable-length message game. The wrapper logic feeds the message
    into the cell and calls the wrapped agent with the hidden state that either corresponds to the end-of-sequence
    term or to the end of the sequence. The wrapper extends it with zero-valued log-prob and entropy tensors so that
    the agent becomes compatible with the SenderReceiverRnnReinforce game.

    As the wrapped agent does not sample, it has to be trained via regular back-propagation. This requires that both the
    the agent's output and  loss function and the wrapped agent are differentiable.

    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, embed_dim=10, hidden_size=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        agent_output = self.agent(encoded, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy

class RnnReceiverImpatient(nn.Module):
    """
    Impatient Listener.
    The wrapper logic feeds the message into the cell and calls the wrapped agent.
    The wrapped agent has to returns the intermediate hidden states for every position.
    All the hidden states are mapped to a categorical distribution with a single
    Linear layer (hidden_to_ouput) followed by a softmax.
    Thess categorical probabilities (step_logits) will then be used to compute the Impatient loss function.
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size,max_len,n_features, cell='rnn', num_layers=1):
        super(RnnReceiverImpatient, self).__init__()

        self.max_len = max_len
        self.hidden_to_output = nn.Linear(hidden_size, n_features)
        self.encoder = RnnEncoderImpatient(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)

        sequence = []
        logits = []
        entropy = []

        for step in range(encoded.size(0)):
            h_t=encoded[step,:,:]
            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample() # Sampling useless ?
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))
            sequence.append(step_logits)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        return sequence, logits, entropy

class RnnReceiverImpatientCompositionality(nn.Module):

    """
    RnnReceiverImpatientCompositionality is an adaptation of RnnReceiverImpatientCompositionality
    for inputs with several attributes (compositionality experiments).
    Each attribute is treated independently.
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size,max_len,n_attributes, n_values, cell='rnn', num_layers=1):
        super(RnnReceiverImpatientCompositionality, self).__init__()

        self.max_len = max_len
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.hidden_to_output = nn.Linear(hidden_size, n_attributes*n_values)
        self.encoder = RnnEncoderImpatient(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):

        encoded = self.encoder(message)

        sequence = []
        slogits = []
        entropy = []

        for step in range(encoded.size(0)):

            h_t=encoded[step,:,:]
            step_logits = F.log_softmax(self.hidden_to_output(h_t).reshape(h_t.size(0),self.n_attributes,self.n_values), dim=2)
            distr = Categorical(logits=step_logits)

            sequence.append(step_logits)

            entropy_step=[]
            slogits_step=[]

            for i in range(step_logits.size(1)):
              distr = Categorical(logits=step_logits[:,i,:])
              entropy_step.append(distr.entropy())
              x = distr.sample()
              slogits_step.append(distr.log_prob(x))

            entropy_step = torch.stack(entropy_step).permute(1, 0)
            slogits_step = torch.stack(slogits_step).permute(1, 0)

            entropy.append(entropy_step)
            slogits.append(slogits_step)

        sequence = torch.stack(sequence).permute(1,0,2,3)
        entropy = torch.stack(entropy).permute(1,0,2)
        slogits= torch.stack(slogits).permute(1,0,2)
        #logits = torch.stack(logits).permute(1, 0)
        #entropy = torch.stack(entropy).permute(1, 0)

        return sequence, slogits, entropy

#class RnnReceiverImpatient2(nn.Module):

#    """
#    Impatient listener
#    """

#    def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, n_features, num_layers=1, cell='rnn'):
#        """
#        :param agent: the agent to be wrapped
#        :param vocab_size: the communication vocabulary size
#        :param embed_dim: the size of the embedding used to embed the output symbols
#        :param hidden_size: the RNN cell's hidden state size
#        :param max_len: maximal length of the output messages
#        :param cell: type of the cell used (rnn, gru, lstm)
#        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
#        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
#        """
#        super(RnnReceiverImpatient, self).__init__()
#        self.agent = agent
#
#        self.max_len = max_len
#
#        self.hidden_to_output = [nn.Linear(hidden_size, n_features).to("cuda")]*self.max_len
#        self.embedding = nn.Embedding(vocab_size, embed_dim)
#        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
#        self.embed_dim = embed_dim
#        self.vocab_size = vocab_size
#        self.num_layers = num_layers
#        self.cells = None

#        cell = cell.lower()
#        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}
#
#        if cell not in cell_types:
#            raise ValueError(f"Unknown RNN Cell: {cell}")

#        cell_type = cell_types[cell]
#        self.cells = nn.ModuleList([
#            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
#            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

#        self.reset_parameters()

#    def reset_parameters(self):
#        nn.init.normal_(self.sos_embedding, 0.0, 0.01)
#
#    def forward(self, message, input, message_lengths=0):
#        from_sym_to_onehot=torch.eye(self.vocab_size,self.vocab_size).to("cuda")
#        prev_hidden = [self.agent(from_sym_to_onehot[message[:,0]],input)]
#        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

#        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

#        input = torch.stack([self.sos_embedding] * from_sym_to_onehot[message[0]].size(0))

#        sequence = []
#        logits = []
#        entropy = []

#        for step in range(self.max_len):
#            input = torch.stack([self.sos_embedding] * from_sym_to_onehot[message[:,step]].size(0))
#            for i, layer in enumerate(self.cells):
#                if isinstance(layer, nn.LSTMCell):
#                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
#                    prev_c[i] = c_t
#                else:
#                    h_t = layer(input, prev_hidden[i])
#                prev_hidden[i] = h_t
#                input = h_t

#            step_logits = F.log_softmax(self.hidden_to_output[step](h_t), dim=1)
#            distr = Categorical(logits=step_logits)
#            entropy.append(distr.entropy())

#            if self.training:
#                x = distr.sample()
#            else:
#                x = step_logits.argmax(dim=1)
#            logits.append(distr.log_prob(x))
#            sequence.append(step_logits)

#        sequence = torch.stack(sequence).permute(1, 0, 2)
#        logits = torch.stack(logits).permute(1, 0)
#        entropy = torch.stack(entropy).permute(1, 0)

#        return sequence, logits, entropy



class SenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param reg: apply the regularization scheduling (Lazy Speaker)
        """
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)

        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        if self.reg:

          sc=rest["acc"].sum()/rest["acc"].size(0)

          # Pour n_features=100
          self.length_cost= sc**(45) / 5

          #self.length_cost= sc**(45) / 10
          #if sc>0.99:
              #self.length_cost=(sc-0.99)*100 +0.01
          #else:
              #self.length_cost=0.
          #if sc>0.995:
              #self.length_cost+=0.01
              #if self.length_cost==0.3:
            #      self.length_cost-=0.01
              #print(self.length_cost)

          #if sc<0.98:
              #self.length_cost=0.


        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class SenderImpatientReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/ Impatient Receiver game with training done via Reinforce.
    It is equivalent to SenderReceiverRnnReinforce but takes into account the intermediate predictions of Impatient Listener:
    - the Impatient loss is used
    - tensor shapes are adapted for variance reduction.

    When reg is set to True, the regularization scheduling is applied (Lazy Speaker).
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param reg: apply the regularization scheduling (Lazy Speaker)
        """
        super(SenderImpatientReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.reg=reg

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)

        # If impatient 1
        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        """ NOISE VERSION

        # Randomly takes a position
        rand_length=np.random.randint(0,message.size(1))

        # Loss by output
        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output[:,rand_length,:], labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r[:,rand_length])

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r[:,rand_length])
        """

        #Loss
        loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r.mean(1))

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r.mean(1))

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r.mean(1)

        if self.reg:
            sc=0.

            for i in range(message_lengths.size(0)):
              sc+=crible_acc[i,message_lengths[i]-1]
            sc/=message_lengths.size(0)

            # Regularization scheduling paper
            #self.length_cost= sc**(45) / 10

            # Pour n_features=100
            self.length_cost= sc**(45) / 5

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class CompositionalitySenderReceiverRnnReinforce(nn.Module):

    """
    Adaptation of SenderReceiverRnnReinforce to inputs with several attributes.
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,n_attributes,n_values,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        """
        super(CompositionalitySenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.reg=reg
        self.n_attributes=n_attributes
        self.n_values=n_values

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):

        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)

        # Noisy channel
        noise_level=0.
        noise_map=torch.from_numpy(1*(np.random.rand(message.size(0),message.size(1))<noise_level)).to("cuda")
        noise=torch.from_numpy(np.random.randint(1,self.sender.vocab_size,size=(message.size(0),message.size(1)))).to("cuda") # random symbols

        message_noise=message*(1-noise_map) + noise_map* noise

        # Receiver normal
        receiver_output_all_att, log_prob_r_all_att, entropy_r_all_att = self.receiver(message_noise, receiver_input, message_lengths)
        #dim=[batch_size,n_att,n_val]

        # reg
        sc=0.

        loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output_all_att, labels,self.n_attributes,self.n_values)

        #if self.reg:
        #     for i in range(message_lengths.size(0)):
        #      sc+=crible_acc[i,message_lengths[i]-1]


        log_prob_r=log_prob_r_all_att.mean(1)
        entropy_r=entropy_r_all_att.mean(1)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        #if self.reg:
        #    sc/=message_lengths.size(0)

        #    if sc>0.98:
        #    	self.length_cost+=0.1
        #    else:
        #    	self.length_cost=0.
            #self.length_cost= sc**(60) / 2

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class CompositionalitySenderImpatientReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,n_attributes,n_values,att_weights,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        """
        super(CompositionalitySenderImpatientReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.reg=reg
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.att_weights=att_weights

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):

        #print(sender_input[:,11:-1])
        message, log_prob_s, entropy_s = self.sender(torch.floor(sender_input))
        message_lengths = find_lengths(message)

        # If impatient 1
        receiver_output_all_att, log_prob_r_all_att, entropy_r_all_att = self.receiver(message, receiver_input, message_lengths)

        # reg
        sc=0.

        # Version de base
        #loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output_all_att, labels,self.n_attributes,self.n_values,self.att_weights)

        # Take into account the fact that an attribute is not sampled
        loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output_all_att, labels,self.n_attributes,self.n_values,self.att_weights)


        if self.reg:
            for i in range(message_lengths.size(0)):
              sc+=crible_acc[i,message_lengths[i]-1]


        log_prob_r=log_prob_r_all_att.mean(1).mean(1)
        entropy_r=entropy_r_all_att.mean(1).mean(1)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        if self.reg:
            sc/=message_lengths.size(0)

            if sc>0.9 and sc<0.99:
                self.length_cost=0.
            if sc>0.99:
                self.length_cost+=0.01
            #if sc<0.9:
            #   self.length_cost=-0.1
            #self.length_cost= sc**(60) / 2

        length_loss = message_lengths.float() * self.length_cost

        # Penalty redundancy
        #counts_unigram=((message[:,1:]-message[:,:-1])==0).sum(axis=1).sum(axis=0)
        #unigram_loss = self.unigram_penalty*counts_unigram

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class TransformerReceiverDeterministic(nn.Module):
    def __init__(self, agent, vocab_size, max_len, embed_dim, num_heads, hidden_size, num_layers, positional_emb=True,
                causal=True):
        super(TransformerReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                          max_len=max_len,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          num_layers=num_layers,
                                          hidden_size=hidden_size,
                                          positional_embedding=positional_emb,
                                          causal=causal)

    def forward(self, message, input=None, lengths=None):
        if lengths is None:
            lengths = find_lengths(message)

        transformed = self.encoder(message, lengths)
        agent_output = self.agent(transformed, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy


class TransformerSenderReinforce(nn.Module):
    def __init__(self, agent, vocab_size, embed_dim, max_len, num_layers, num_heads, hidden_size,
                 generate_style='standard', causal=True, force_eos=True):
        """
        :param agent: the agent to be wrapped, returns the "encoder" state vector, which is the unrolled into a message
        :param vocab_size: vocab size of the message
        :param embed_dim: embedding dimensions
        :param max_len: maximal length of the message (including <eos>)
        :param num_layers: number of transformer layers
        :param num_heads: number of attention heads
        :param hidden_size: size of the FFN layers
        :param causal: whether embedding of a particular symbol should only depend on the symbols to the left
        :param generate_style: Two alternatives: 'standard' and 'in-place'. Suppose we are generating 4th symbol,
            after three symbols [s1 s2 s3] were generated.
            Then,
            'standard': [s1 s2 s3] -> embeddings [[e1] [e2] [e3]] -> (s4 = argmax(linear(e3)))
            'in-place': [s1 s2 s3] -> [s1 s2 s3 <need-symbol>] -> embeddings [[e1] [e2] [e3] [e4]] -> (s4 = argmax(linear(e4)))
        :param force_eos: <eos> added to the end of each sequence
        """
        super(TransformerSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos
        assert generate_style in ['standard', 'in-place']
        self.generate_style = generate_style
        self.causal = causal

        self.max_len = max_len

        if force_eos:
            self.max_len -= 1

        self.transformer = TransformerDecoder(embed_dim=embed_dim,
                                              max_len=max_len, num_layers=num_layers,
                                              num_heads=num_heads, hidden_size=hidden_size)

        self.embedding_to_vocab = nn.Linear(embed_dim, vocab_size)

        self.special_symbol_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        self.embed_scale = math.sqrt(embed_dim)

    def generate_standard(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = self.special_symbol_embedding.expand(batch_size, -1).unsqueeze(1).to(device)
        input = special_symbol

        for step in range(self.max_len):
            if self.causal:
                attn_mask = torch.triu(torch.ones(step+1, step+1).byte(), diagonal=1).to(device)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
            else:
                attn_mask = None
            output = self.transformer(embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask)
            step_logits = F.log_softmax(self.embedding_to_vocab(output[:, -1, :]), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            input = torch.cat([input, new_embedding.unsqueeze(dim=1)], dim=1)

        return sequence, logits, entropy

    def generate_inplace(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = self.special_symbol_embedding.expand(batch_size, -1).unsqueeze(1).to(encoder_state.device)
        output = []
        for step in range(self.max_len):
            input = torch.cat(output + [special_symbol], dim=1)
            if self.causal:
                attn_mask = torch.triu(torch.ones(step+1, step+1).byte(), diagonal=1).to(device)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
            else:
                attn_mask = None

            embedded = self.transformer(embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask)
            step_logits = F.log_softmax(self.embedding_to_vocab(embedded[:, -1, :]), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            output.append(new_embedding.unsqueeze(dim=1))

        return sequence, logits, entropy

    def forward(self, x):
        encoder_state = self.agent(x)

        if self.generate_style == 'standard':
            sequence, logits, entropy = self.generate_standard(encoder_state)
        elif self.generate_style == 'in-place':
            sequence, logits, entropy = self.generate_inplace(encoder_state)
        else:
            assert False, 'Unknown generate style'

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy
