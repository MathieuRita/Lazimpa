# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.channel.features import OneHotLoader, UniformLoader
from egg.zoo.channel.archs import Sender, Receiver
from egg.core.util import dump_sender_receiver_test
from egg.core.util import dump_impose_message
from egg.core.util import dump_test_position
from egg.core.util import dump_test_position_impatient
from egg.core.reinforce_wrappers import RnnReceiverImpatient
from egg.core.reinforce_wrappers import SenderImpatientReceiverRnnReinforce
from egg.core.util import dump_sender_receiver_impatient


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--dim_dataset', type=int, default=10240,
                        help='Dim of constructing the data (default: 10240)')
    parser.add_argument('--force_eos', type=int, default=0,
                        help='Force EOS at the end of the messages (default: 0)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--receiver_num_heads', type=int, default=8,
                        help='Number of attention heads for Transformer Receiver (default: 8)')
    parser.add_argument('--sender_num_heads', type=int, default=8,
                        help='Number of self-attention heads for Transformer Sender (default: 8)')
    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--causal_sender', default=False, action='store_true')
    parser.add_argument('--causal_receiver', default=False, action='store_true')

    parser.add_argument('--sender_generate_style', type=str, default='in-place', choices=['standard', 'in-place'],
                        help='How the next symbol is generated within the TransformerDecoder (default: in-place)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: rnn)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument('--probs', type=str, default='uniform',
                        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="Penalty for the message length, each symbol would before <EOS> would be "
                             "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")
    parser.add_argument('--early_stopping_thr', type=float, default=0.9999,
                        help="Early stopping threshold on accuracy (default: 0.9999)")

    parser.add_argument('--receiver_weights',type=str ,default="receiver_weights.pth",
                        help="Weights of the receiver agent")
    parser.add_argument('--sender_weights',type=str ,default="sender_weights.pth",
                        help="Weights of the sender agent")
    parser.add_argument('--save_dir',type=str ,default="analysis/",
                        help="Directory to save the results of the analysis")
    parser.add_argument('--impatient', type=bool, default=False,
                        help="Impatient listener")
    parser.add_argument('--unigram_pen', type=float, default=0.0,
                        help="Add a penalty for redundancy")

    args = core.init(parser, params)

    return args

def dump(game, n_features, device, gs_mode):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        core.dump_sender_receiver(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        #print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    return acc, messages

def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}

def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1

    if opts.probs == 'uniform':
        probs = np.ones(opts.n_features)
    elif opts.probs == 'powerlaw':
        probs = 1 / np.arange(1, opts.n_features+1, dtype=np.float32)
    else:
        probs = np.array([float(x) for x in opts.probs.split(',')], dtype=np.float32)
    probs /= probs.sum()

    train_loader = OneHotLoader(n_features=opts.n_features, batch_size=opts.batch_size,
                                batches_per_epoch=opts.batches_per_epoch, probs=probs)

    # single batches with 1s on the diag
    test_loader = UniformLoader(opts.n_features)

    if opts.sender_cell == 'transformer':
        sender = Sender(n_features=opts.n_features, n_hidden=opts.sender_embedding)
        sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size,
                                                 embed_dim=opts.sender_embedding, max_len=opts.max_len,
                                                 num_layers=opts.sender_num_layers, num_heads=opts.sender_num_heads,
                                                 hidden_size=opts.sender_hidden,
                                                 force_eos=opts.force_eos,
                                                 generate_style=opts.sender_generate_style,
                                                 causal=opts.causal_sender)
    else:
        sender = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)

        sender = core.RnnSenderReinforce(sender,
                                   opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                   cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                   force_eos=force_eos)
    if opts.receiver_cell == 'transformer':
        receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_embedding)
        receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size, opts.max_len,
                                                         opts.receiver_embedding, opts.receiver_num_heads, opts.receiver_hidden,
                                                         opts.receiver_num_layers, causal=opts.causal_receiver)
    else:

        receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)

        if not opts.impatient:
          receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
          receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding,
                                                 opts.receiver_hidden, cell=opts.receiver_cell,
                                                 num_layers=opts.receiver_num_layers)
        else:
          receiver = Receiver(n_features=opts.receiver_hidden, n_hidden=opts.vocab_size)
          # If impatient 1
          receiver = RnnReceiverImpatient(receiver, opts.vocab_size, opts.receiver_embedding,
                                            opts.receiver_hidden, cell=opts.receiver_cell,
                                            num_layers=opts.receiver_num_layers, max_len=opts.max_len, n_features=opts.n_features)

    sender.load_state_dict(torch.load(opts.sender_weights,map_location=torch.device('cpu')))
    receiver.load_state_dict(torch.load(opts.receiver_weights,map_location=torch.device('cpu')))

    if not opts.impatient:
        game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                           length_cost=opts.length_cost,unigram_penalty=opts.unigram_pen)
    else:
        game = SenderImpatientReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                           length_cost=opts.length_cost,unigram_penalty=opts.unigram_pen)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])



    # Debut test position

    position_sieve=np.zeros((opts.n_features,opts.max_len))

    for position in range(opts.max_len):

        dataset = [[torch.eye(opts.n_features).to(device), None]]

        if opts.impatient:
            sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
                dump_test_position_impatient(trainer.game,
                                    dataset,
                                    position=position,
                                    voc_size=opts.vocab_size,
                                    gs=False,
                                    device=device,
                                    variable_length=True)
        else:
            sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
                dump_test_position(trainer.game,
                                    dataset,
                                    position=position,
                                    voc_size=opts.vocab_size,
                                    gs=False,
                                    device=device,
                                    variable_length=True)

        acc_pos=[]

        for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
            input_symbol = sender_input.argmax()
            output_symbol = receiver_output.argmax()
            acc = (input_symbol == output_symbol).float().item()
            acc_pos.append(acc)

        acc_pos=np.array(acc_pos)

        position_sieve[:,position]=acc_pos

    # Put -1 for position after message_length
    _, messages = dump(trainer.game, opts.n_features, device, False)

    # Convert messages to numpy array
    messages_np=[]
    for x in messages:
        x = x.cpu().numpy()
        messages_np.append(x)

    for i in range(len(messages_np)):
        # Message i
        message_i=messages_np[i]
        id_0=np.where(message_i==0)[0]

        if id_0.shape[0]>0:
          for j in range(id_0[0]+1,opts.max_len):
              position_sieve[i,j]=-1


    np.save("analysis/position_sieve.npy",position_sieve)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
