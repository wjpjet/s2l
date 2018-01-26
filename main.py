"""Run Sequence to label learning"""
import os
import argparse
import torch
import utils
from model import s2l_Net
from train import train, evaluate
from vocab import Vocabulary

def main():
    parser = argparse.ArgumentParser(description='Run Sequence to label entailment learning')

    # Seeds
    parser.add_argument('--seed', default=10001, type=int,
                        help='Random seed')

    # Network hyperparameters
    parser.add_argument('--h_size', default=int(128), type=int)
    parser.add_argument('--mlp_d', default=int(256), type=int)
    parser.add_argument('--lstm_layers', default=int(1), type=int)

    # Training hyperparameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--epochs', default='50', type=int)	
    parser.add_argument('--start_epoch', default='0', type=int)

    # To resume training, path to training
    parser.add_argument('--resume', type=str)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--output', default='exps/', type=str)
    parser.add_argument('--expr_name', required=True, type=str)

    # For using cuda
    parser.add_argument('--cuda', action='store_true')

    # For loading vocab, labels, and word embeddings
    parser.add_argument('--vocab', default='vocabulary.txt', type=str)
    parser.add_argument('--labels', default='labels.txt', type=str)
    parser.add_argument('--embed', required=True, type=str)

    # Load training data
    parser.add_argument('--data', default='snli_1.0/snli_1.0_train.jsonl', type=str)
    parser.add_argument('--net', default=None, type=str)

    args = parser.parse_args()
    
    if args.dev:
        args.output = 'devs/'

    args.output = os.path.join(args.output, args.expr_name)
    utils.Config(vars(args)).dump(os.path.join(args.output, 'configs.txt'))
    return args


if __name__ == '__main__':
    args = main()

    torch.backends.cudnn.benckmark = True
    utils.set_all_seeds(args.seed)

    # Init data from SNLI or other data source
    data, itos = utils.obtain_data(args.data, args.vocab) 

    # Build Vocabulary
    print("Loading vocabulary and word embeddings...")
    vocab = Vocabulary(args.labels, itos)
    vocab.set_word_embedding(args.embed, args.vocab)
    numpy_data = vocab.process_data(data)
    print("Embedding shape: ")
    print(vocab.embeddings.shape)

    # Build network
    net = s2l_Net(h_size=args.h_size, 
                    v_size=vocab.embeddings.shape[0], 
                    embed_d=vocab.embeddings.shape[1], 
                    mlp_d=args.mlp_d, 
                    num_classes=vocab.num_classes, 
                    lstm_layers=args.lstm_layers
                    )
    net.embedding.weight.data = torch.from_numpy(vocab.embeddings)
    net.embedding.weight.requires_grad = False
    net.display()
    net.cuda()

    #evaluator = lambda logger: evaluate(net, logger, args.dev)
    print("Initiating Training")
    train(net,
          args.cuda,
          args.start_epoch,
          args.epochs,
          args.batch_size,
          numpy_data,
          vocab,
          args.resume,
          args.output)
