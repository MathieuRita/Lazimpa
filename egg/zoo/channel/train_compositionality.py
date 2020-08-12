# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import itertools
import torch.utils.data
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import egg.core as core
#from scipy.stats import entropy
from egg.core import EarlyStopperAccuracy
from egg.zoo.channel.features import OneHotLoader, UniformLoader, OneHotLoaderCompositionality, TestLoaderCompositionality
from egg.zoo.channel.archs import Sender, Receiver
from egg.core.reinforce_wrappers import RnnReceiverImpatient, RnnReceiverImpatientCompositionality, RnnReceiverCompositionality
from egg.core.reinforce_wrappers import SenderImpatientReceiverRnnReinforce, CompositionalitySenderImpatientReceiverRnnReinforce, CompositionalitySenderReceiverRnnReinforce
from egg.core.util import dump_sender_receiver_impatient, dump_sender_receiver_impatient_compositionality, dump_sender_receiver_compositionality

from egg.core.trainers import CompoTrainer


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

    # AJOUT
    parser.add_argument('--dir_save', type=str, default="expe_1",
                        help="Directory in which we will save the information")
    parser.add_argument('--unigram_pen', type=float, default=0.0,
                        help="Add a penalty for redundancy")
    parser.add_argument('--impatient', type=bool, default=False,
                        help="Impatient listener")
    parser.add_argument('--print_message', type=bool, default=False,
                        help='Print message ?')
    parser.add_argument('--reg', type=bool, default=False,
                        help='Add regularization ?')

    # Compositionality
    parser.add_argument('--n_attributes', type=int, default=3,
                        help='Number of attributes (default: 2)')
    parser.add_argument('--n_values', type=int, default=3,
                        help='Number of values by attribute')
    parser.add_argument('--att_weights', type=list, default=[1,1,1],
                        help='Weights of each attribute')
    parser.add_argument('--probs_attributes', type=str, default="uniform",
                        help='Sampling prob for each att')

    args = core.init(parser, params)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}

def loss_impatient(sender_input, _message, message_length, _receiver_input, receiver_output, _labels):

    to_onehot=torch.eye(_message.size(1)).to("cuda")
    to_onehot=torch.cat((to_onehot,torch.zeros((1,_message.size(1))).to("cuda")),0)
    len_mask=[]
    for i in range(message_length.size(0)):
      len_mask.append(to_onehot[message_length[i]])
    len_mask=torch.stack(len_mask,dim=0)

    coef=(1/message_length.to(float)).repeat(_message.size(1),1).transpose(1,0)
    coef2=coef*torch.arange(_message.size(1),0,-1).repeat(_message.size(0),1).to("cuda")

    len_mask=torch.cumsum(len_mask,dim=1)
    len_mask=torch.ones(len_mask.size()).to("cuda").add_(-len_mask)

    len_mask.mul_((coef2))
    len_mask.mul_((1/len_mask.sum(1)).repeat((_message.size(1),1)).transpose(1,0))

    crible_acc=torch.zeros(size=_message.size()).to("cuda")
    crible_loss=torch.zeros(size=_message.size()).to("cuda")

    for i in range(receiver_output.size(1)):
      crible_acc[:,i].add_((receiver_output[:,i,:].argmax(dim=1) == sender_input.argmax(dim=1)).detach().float())
      crible_loss[:,i].add_(F.cross_entropy(receiver_output[:,i,:], sender_input.argmax(dim=1), reduction="none"))

    acc=crible_acc*len_mask
    loss=crible_loss*len_mask

    acc = acc.sum(1)
    loss= loss.sum(1)

    return loss, {'acc': acc}, crible_acc

def loss_compositionality(sender_input, _message, message_length, _receiver_input, receiver_output, _labels,n_attributes,n_values):

    loss=0.

    sender_input=sender_input.reshape(sender_input.size(0),n_attributes,n_values)

    crible_acc=(receiver_output.argmax(dim=2)==sender_input.argmax(2)).detach().float().mean(1)

    for j in range(receiver_output.size(1)):
        #K=10*(1/(j+1))
        K=sender_input[:,j,:].max(dim=1).values
        loss+=K*F.cross_entropy(receiver_output[:,j,:], sender_input[:,j,:].argmax(dim=1), reduction="none")

    return loss, {'acc': crible_acc}, crible_acc

def loss_impatient_compositionality(sender_input, _message, message_length, _receiver_input, receiver_output, _labels,n_attributes,n_values,att_weights):

    to_onehot=torch.eye(_message.size(1)).to("cuda")
    to_onehot=torch.cat((to_onehot,torch.zeros((1,_message.size(1))).to("cuda")),0)
    len_mask=[]
    for i in range(message_length.size(0)):
      len_mask.append(to_onehot[message_length[i]])
    len_mask=torch.stack(len_mask,dim=0)

    coef=(1/message_length.to(float)).repeat(_message.size(1),1).transpose(1,0)
    #coef2=coef*torch.arange(_message.size(1),0,-1).repeat(_message.size(0),1).to("cuda")
    coef2=coef
    # NEW LOSS
    #Mlen=n_attributes*torch.ones(message_length.size()).to("cuda")
    #coef=(1/Mlen).repeat(_message.size(1),1).transpose(1,0)
    #coef2=coef

    len_mask=torch.cumsum(len_mask,dim=1)
    len_mask=torch.ones(len_mask.size()).to("cuda").add_(-len_mask)

    #len_mask.mul_((coef2)) # A priori avec la normalisation apres, etape useless
    len_mask.mul_((1/len_mask.sum(1)).repeat((_message.size(1),1)).transpose(1,0))

    crible_acc=torch.zeros(size=_message.size()).to("cuda")
    crible_loss=torch.zeros(size=_message.size()).to("cuda")

    for i in range(receiver_output.size(1)):
      ro=receiver_output[:,i,:].reshape(receiver_output.size(0),n_attributes,n_values)
      si=sender_input.reshape(sender_input.size(0),n_attributes,n_values)

      crible_acc[:,i].add_((ro.argmax(dim=2)==si.argmax(2)).detach().float().sum(1)/n_attributes)

      #crible_loss[:,i].add_(F.cross_entropy(receiver_output[:,i,:], sender_input.argmax(dim=1), reduction="none"))
      for j in range(ro.size(1)):
        #K=att_weights[j]
        K=si[:,j,:].max(dim=1).values
        crible_loss[:,i].add_(K*F.cross_entropy(ro[:,j,:], si[:,j,:].argmax(dim=1), reduction="none")/n_attributes)

    acc=crible_acc*len_mask
    loss=crible_loss*len_mask

    acc = acc.sum(1)
    loss= loss.sum(1)

    return loss, {'acc': acc}, crible_acc

def dump(game, n_features, device, gs_mode, epoch):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        core.dump_sender_receiver(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%50==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    #print(f'Mean accuracy wrt uniform distribution is {unif_acc}')
    #print(f'Mean accuracy wrt powerlaw distribution is {powerlaw_acc}')
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    return acc_vec, messages

def dump_impatient(game, n_features, device, gs_mode,epoch):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        dump_sender_receiver_impatient(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%25==0 or epoch>300:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    #print(f'Mean accuracy wrt uniform distribution is {unif_acc}')
    #print(f'Mean accuracy wrt powerlaw distribution is {powerlaw_acc}')
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    return acc_vec, messages

def dump_compositionality(game, n_attributes, n_values, device, gs_mode,epoch):
    # tiny "dataset"
    one_hots = torch.eye(n_values)

    val=np.arange(n_values)
    combination=list(itertools.product(val,repeat=n_attributes))

    dataset=[]

    for i in range(len(combination)):
      new_input=torch.zeros(0)
      for j in combination[i]:
        new_input=torch.cat((new_input,one_hots[j]))
      dataset.append(new_input)


    # ATTENTION ICI AJOUT JUSTE DES COMBINAISONS POUR DEUX ATTRIBUTES
    #dataset.append(torch.cat((torch.zeros(n_values),torch.zeros(n_values))))
    #combination.append((-1,-1))

    #for j in range(n_values):
    #  new_input=torch.cat((torch.zeros(n_values),one_hots[j]))
    #  dataset.append(new_input)
    #  combination.append((-1,j))

    #for j in range(n_values):
    #  new_input=torch.cat((one_hots[j],torch.zeros(n_values)))
    #  dataset.append(new_input)
    #  combination.append((j,-1))

    dataset=torch.stack(dataset)

    dataset=[[dataset,None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        dump_sender_receiver_compositionality(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    acc_vec=np.zeros(((n_values**n_attributes), n_attributes))

    for i in range(len(receiver_outputs)):
      message=messages[i]
      correct=True
      if i<n_values**n_attributes:
          for j in range(len(list(combination[i]))):
            if receiver_outputs[i][j]==list(combination[i])[j]:
              unif_acc+=1
              acc_vec[i,j]=1
      print(f'input: {",".join([str(x) for x in combination[i]])} -> message: {",".join([str(x.item()) for x in message])} -> output: {",".join([str(x) for x in receiver_outputs[i]])}', flush=True)

    unif_acc /= (n_values**n_attributes) * n_attributes

    print(json.dumps({'unif': unif_acc}))

    return acc_vec, messages

def dump_impatient_compositionality(game, n_attributes, n_values, device, gs_mode,epoch):
    # tiny "dataset"
    one_hots = torch.eye(n_values)

    val=np.arange(n_values)
    combination=list(itertools.product(val,repeat=n_attributes))

    dataset=[]

    for i in range(len(combination)):
      new_input=torch.zeros(0)
      for j in combination[i]:
        new_input=torch.cat((new_input,one_hots[j]))
      dataset.append(new_input)

    dataset=torch.stack(dataset)

    dataset=[[dataset,None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        dump_sender_receiver_impatient_compositionality(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    acc_vec=np.zeros(((n_values**n_attributes), n_attributes))

    for i in range(len(receiver_outputs)):
      message=messages[i]
      correct=True
      for j in range(len(list(combination[i]))):
        if receiver_outputs[i][j]==list(combination[i])[j]:
          unif_acc+=1
          acc_vec[i,j]=1
      if epoch%5==0:
          print(f'input: {",".join([str(x) for x in combination[i]])} -> message: {",".join([str(x.item()) for x in message])} -> output: {",".join([str(x) for x in receiver_outputs[i]])}', flush=True)

    unif_acc /= (n_values**n_attributes) * n_attributes

    print(json.dumps({'unif': unif_acc}))

    return acc_vec, messages

def main(params):
    print(torch.cuda.is_available())
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1

    # Distribution of the inputs
    if opts.probs=="uniform":
        probs=[]
        probs_by_att = np.ones(opts.n_values)
        probs_by_att /= probs_by_att.sum()
        for i in range(opts.n_attributes):
            probs.append(probs_by_att)

    if opts.probs=="entropy_test":
        probs=[]
        for i in range(opts.n_attributes):
            probs_by_att = np.ones(opts.n_values)
            probs_by_att[0]=1+(1*i)
            probs_by_att /= probs_by_att.sum()
            probs.append(probs_by_att)

    if opts.probs_attributes=="uniform":
        probs_attributes=[1]*opts.n_attributes

    if opts.probs_attributes=="uniform_indep":
        probs_attributes=[]
        probs_attributes=[0.2]*opts.n_attributes

    if opts.probs_attributes=="echelon":
        probs_attributes=[]
        for i in range(opts.n_attributes):
            #probs_attributes.append(1.-(0.2)*i)
            #probs_attributes.append(0.7+0.3/(i+1))
            probs_attributes=[1.,0.95,0.9,0.85]

    print("Probability by attribute is:",probs_attributes)

    train_loader = OneHotLoaderCompositionality(n_values=opts.n_values, n_attributes=opts.n_attributes, batch_size=opts.batch_size*opts.n_attributes,
                                                batches_per_epoch=opts.batches_per_epoch, probs=probs, probs_attributes=probs_attributes)

    # single batches with 1s on the diag
    test_loader = TestLoaderCompositionality(n_values=opts.n_values,n_attributes=opts.n_attributes)

    ### SENDER ###

    sender = Sender(n_features=opts.n_attributes*opts.n_values, n_hidden=opts.sender_hidden)

    sender = core.RnnSenderReinforce(sender,opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                   cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                   force_eos=force_eos)


    ### RECEIVER ###

    receiver = Receiver(n_features=opts.n_values, n_hidden=opts.receiver_hidden)

    if not opts.impatient:
        receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
        receiver = RnnReceiverCompositionality(receiver, opts.vocab_size, opts.receiver_embedding,
                                            opts.receiver_hidden, cell=opts.receiver_cell,
                                            num_layers=opts.receiver_num_layers, max_len=opts.max_len, n_attributes=opts.n_attributes, n_values=opts.n_values)
    else:
        receiver = Receiver(n_features=opts.receiver_hidden, n_hidden=opts.vocab_size)
        # If impatient 1
        receiver = RnnReceiverImpatientCompositionality(receiver, opts.vocab_size, opts.receiver_embedding,
                                            opts.receiver_hidden, cell=opts.receiver_cell,
                                            num_layers=opts.receiver_num_layers, max_len=opts.max_len, n_attributes=opts.n_attributes, n_values=opts.n_values)


    if not opts.impatient:
        game = CompositionalitySenderReceiverRnnReinforce(sender, receiver, loss_compositionality, sender_entropy_coeff=opts.sender_entropy_coeff,
                                           n_attributes=opts.n_attributes,n_values=opts.n_values,receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                           length_cost=opts.length_cost,unigram_penalty=opts.unigram_pen,reg=opts.reg)
    else:
        game = CompositionalitySenderImpatientReceiverRnnReinforce(sender, receiver, loss_impatient_compositionality, sender_entropy_coeff=opts.sender_entropy_coeff,
                                           n_attributes=opts.n_attributes,n_values=opts.n_values,att_weights=opts.att_weights,receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                           length_cost=opts.length_cost,unigram_penalty=opts.unigram_pen,reg=opts.reg)

    optimizer = core.build_optimizer(game.parameters())

    trainer = CompoTrainer(n_attributes=opts.n_attributes,n_values=opts.n_values,game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

    curr_accs=[0]*7

    game.att_weights=[1]*(game.n_attributes)

    for epoch in range(int(opts.n_epochs)):

        print("Epoch: "+str(epoch))

        #if epoch%100==0:
        #  trainer.optimizer.defaults["lr"]/=2


        trainer.train(n_epochs=1)
        if opts.checkpoint_dir:
            trainer.save_checkpoint(name=f'{opts.name}_vocab{opts.vocab_size}_rs{opts.random_seed}_lr{opts.lr}_shid{opts.sender_hidden}_rhid{opts.receiver_hidden}_sentr{opts.sender_entropy_coeff}_reg{opts.length_cost}_max_len{opts.max_len}')

        if not opts.impatient:
            acc_vec,messages=dump_compositionality(trainer.game, opts.n_attributes, opts.n_values, device, False,epoch)
        else:
            acc_vec,messages=dump_impatient_compositionality(trainer.game, opts.n_attributes, opts.n_values, device, False,epoch)

        print(acc_vec.mean(0))
        #print(trainer.optimizer.defaults["lr"])


        # ADDITION TO SAVE MESSAGES
        all_messages=[]
        for x in messages:
            x = x.cpu().numpy()
            all_messages.append(x)
        all_messages = np.asarray(all_messages)

        if epoch%50==0:
            torch.save(sender.state_dict(), opts.dir_save+"/sender/sender_weights"+str(epoch)+".pth")
            torch.save(receiver.state_dict(), opts.dir_save+"/receiver/receiver_weights"+str(epoch)+".pth")

        np.save(opts.dir_save+'/messages/messages_'+str((epoch))+'.npy', all_messages)
        np.save(opts.dir_save+'/accuracy/accuracy_'+str((epoch))+'.npy', acc_vec)
        print(acc_vec.T)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
