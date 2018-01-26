# Sequences to label (s2l) - Entailment Classification
A PyTorch library for recognizing entailment between two sentences
### Requirements
Python 3.5+,
PyTorch 0.3.0,
CUDA 7.5+, 
tqdm

## How to run

### Get the data
Default training uses the stanford SNLI dataset, you can download this dataset [here.](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) Put the snli_1.0 folder in the s2l repository.

You will also need to download the Stanford Glove word embeddings from [here.](http://nlp.stanford.edu/data/glove.6B.zip) Currently s2l only works with 300 dimensional word embeddings. 

### Run training
With defaults:
```
python main.py --cuda --expr_name=test_model --embed=/path/to/glove.6B.300d.txt
```
As of now, this package requires `--cuda`

#### Load a previous model
`--resume:` path to the desired checkpoint file. Example -> `--resume=exps/test_model/checkpoint.pth.tar`


## Hyperparameters

#### Network


`--h_size:` size of LSTM hidden layer

`--lstm_layers:` number of layers in LSTM

`--mlp_d:` size of linear layer following LSTM

#### Vocabulary
`--vocab:` default is vocabulary.txt, here you can create your own vocab

`--labels:` the entailment labels are saved in labels.txt, change for different labeling

`--embed:` path to glove.6B.300d.txt, as of now will only run with this glove embedding 

