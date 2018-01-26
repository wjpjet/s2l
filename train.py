import os
import time
import torch
import torch.nn
import utils
import random
import numpy as np
from logger import Logger
from tqdm import tqdm
from torch.autograd import Variable
from torch import optim
from torch.nn import CrossEntropyLoss

def save_checkpoint(state, output_dir, filename='checkpoint.pth.tar'):
    torch.save(state, output_dir + "/" + filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')

# Forward and backward pass with batch_size samples
def forward_backward_pass(net, data, vocab, optim, batch_size, cuda, criterion):
    # Convert minibatch to numpy minibatch
    s1, s2, y, lengths = utils.convert_to_numpy(data, vocab)
    
    # Convert numpy to Tensor
    s1_tensor = torch.from_numpy(s1).type(torch.LongTensor)
    s2_tensor = torch.from_numpy(s2).type(torch.LongTensor)
    target_tensor = torch.from_numpy(y).type(torch.LongTensor)
    
    s1 = Variable(s1_tensor)
    s2 = Variable(s2_tensor)
    target = Variable(target_tensor)
    
    if cuda:
        s1 = s1.cuda()
        s2 = s2.cuda()
        target = target.cuda()
    
    optim.zero_grad()
    output = net.forward(s1,s2,lengths) 
    loss = criterion(output, target)

    # Backprogogate and update optimizer
    loss.backward()
    optim.step()
    return loss

def train(net,
            cuda,
            start_epoch,
            epochs,
            batch_size,
            numpy_data,
            vocab,
            resume,
            output_dir):

    logger = Logger(os.path.join(output_dir, 'train_log.txt'))
    
    # Adam optimizer TODO: add optimizer selection
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

    # If loading from checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}', rerun with correct checkpoint".format(resume))
            exit(0)

    criterion = CrossEntropyLoss()
    net.train()

    # Set cuda:
    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    for epoch in range(start_epoch, epochs):
        print("Epoch {0}/{1}: {2}%".format(epoch, epochs, 100*float(epoch)/epochs))

	    # Shuffle for new set of mini batches
        random.shuffle(numpy_data)

        # Loop dataset in chunks of size batch_size
        for start, end in tqdm(utils.batch_index_gen(batch_size, len(numpy_data))):
            loss = forward_backward_pass(net, numpy_data[start:end], vocab, optimizer, batch_size, cuda, criterion)

        # Save model each epoch
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
             }, output_dir)

# Evaluation mdoel
def evaluate(net, logger,  mode='dev'):
    file_name = 'snli_1.0/snli_1.0_dev.jsonl' if mode == 'dev' else 'snli_1.0/snli_1.0_test.jsonl'
    
    dev_data, _ = obtain_data(file_name)
    dev_n_data = vocab.process_data(dev_data)
    
    print("Length of data: {}".format(len(dev_n_data)))
    
    eval_batch_size = 1024
    net.eval()

    total = len(dev_n_data)
    hit = 0
    correct = 0
    
    # Batch dev eval
    for start, end in batch_index_gen(eval_batch_size, len(dev_n_data)):

        s1, s2, y, lengths = convert_to_numpy(dev_n_data[start:end])

        s1_tensor = torch.from_numpy(s1).type(torch.LongTensor)
        s2_tensor = torch.from_numpy(s2).type(torch.LongTensor)
        target_tensor = torch.from_numpy(y).type(torch.LongTensor)

        s1 = Variable(s1_tensor, volatile=True)
        s2 = Variable(s2_tensor, volatile=True)
        target = Variable(target_tensor, volatile=True)

        if cuda:
            s1 = s1.cuda()
            s2 = s2.cuda()
            target = target.cuda()

        output = net.forward(s1,s2,lengths)
        loss = criterion(output, target)
    
        #print("output size: {}".format(output.size()))
        #print("target size: {}".format(target.size()))
        pred = output.data.max(1)[1] # get the index of the max log-probability
        #print(pred[:5])
        #print(output[:])
        correct += pred.eq(target.data).cpu().sum()

    return correct / float(total)
