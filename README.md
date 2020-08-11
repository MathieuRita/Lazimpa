# LazImpa

This repository gathers the code used for all the experiments of the following paper:

- *“LazImpa”: Lazy and Impatient neural agents learn to communicate efficiently* (under review)

The code is an extension of EGG toolkit (https://github.com/facebookresearch/EGG) presented in *EGG: a toolkit for research on Emergence of lanGuage in Games*, Eugene Kharitonov, Rahma Chaabouni, Diane Bouchacourt, Marco Baroni. EMNLP 2019.

## Presentation

`LazImpa` repository implements a Speaker/Listener game where agents have to cooperatly communicate in order to win the game.

## Run the code

### Command lines

1. First clone the repository:
`git clone https://github.com/MathieuRita/LE_test.git`

`mv "./LE_test/egg" "./egg"`

2. Create a directory in which all the useful data will be saved (you have to respect the following hierarchy):

`mkdir dir_save`

`mkdir dir_save/sender`

`mkdir dir_save/receiver`

`mkdir dir_save/messages`

`mkdir dir_save/accuracy`


3. Train agents:

`python -m egg.zoo.channel.train --dir_save=dir_save \
                                                                  --vocab_size=40 \
                                                                  --max_len=30 \
                                                                  --impatient=True \
                                                                  --reg=True \
                                                                  --n_features=1000 \
                                                                  --print_message=False \
                                                                  --random_seed=7 \
                                                                  --probs="powerlaw" \
                                                                  --n_epoch=2501 \
                                                                  --batch_size=512 \
                                                                  --length_cost=0. \
                                                                  --sender_cell="lstm" \
                                                                  --receiver_cell="lstm" \
                                                                  --sender_hidden=250 \
                                                                  --receiver_hidden=600 \
                                                                  --receiver_embedding=100 \
                                                                  --sender_embedding=10 \
                                                                  --batches_per_epoch=100 \
                                                                  --lr=0.001 \
                                                                  --sender_entropy_coeff=2. \
                                                                  --sender_num_layers=1 \
                                                                  --receiver_num_layers=1 \
                                                                  --early_stopping_thr=0.99 `

4. Test agents:

**TO DO**

### H-parameters description

H-params can be divided in 3 classes: experiment H-params, architecture H-params, optimization H-params, backup H-params. Here is a description of the main H-parameters:

1. Experiment H-params:
- `vocab_size`: size of vocabulary in the communication channel (default=40)
- `max_len`: maximum length of the message in the communication channel (default=30)
- `n_features`:
- `probs`: frequency distribution of the inputs (default=`powerlaw`)

2. Architecture H-params:
- `impatient`: if set to `True`, the Listener is made Impatient
- `reg`: if set to `True`, the Speaker is made Lazy
- `sender_cell`:
- `sender_hidden`:
- `sender_embedding`:
- `sender_num_layers`:
- `receiver_cell`:
- `receiver_hidden`:
- `receiver_embedding`:
- `receiver_num_layers`:

3. Optimization H-params:
- `n_epochs`:
- `lr`:
- `sender_entropy_coeff`

## Paper results

## Reproductibility

## How to cite ?
