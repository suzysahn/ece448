# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss functions
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/
        self.input = nn.Linear(in_size, 128)
        self.hidden = nn.Linear(128, 128)
        self.output = nn.Linear(128, out_size)


    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        # return self.net.parameters()
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
         # return torch.zeros(x.shape[0], 3)
        x = F.selu(self.input(x))
        x = F.selu(self.hidden(x))
        return self.output(x) 

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        L = self.loss_fn(x, y)
        L.backward()
        return L


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, out_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """

    # train_set (7500, 784)
    # train_lables (7500)
    # dev_set (2400, 784)
    # n_iter = 10
    # batch_size 100
    
    # @param lrate: The learning rate for the model.
    # @param loss_fn: The loss function
    # @param in_size: Dimension of input
    # @param out_size: Dimension of output
    # The network should have the following architecture (in terms of hidden units):
    # in_size -> 128 ->  out_size    

    # Build model
    lrate = 1e-4
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn += 

    # normalizing data
    mean = train_set.mean()
    std = train_set.std()
    train_set = (train_set - mean) / std
    dev_set = (dev_set - mean) / std

    in_size = len(train_set[0])
    out_size = 5
    model = NeuralNet(lrate, loss_fn, in_size, out_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.9)     

    # https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/

    batch = 0
    losses = []
    running_loss = 0
    model.train()
    for inputs, labels in zip(train_set, train_labels):
        optimizer.zero_grad()
        outputs = model.forward(inputs).expand(1, out_size)
        labels = labels.expand(1)
        loss = model.step(outputs, labels)
        optimizer.step()
        running_loss += loss.item()

        batch += 1
        if batch % batch_size == 0:    # print every 100 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #     (epoch + 1, i + 1, running_loss / 2000))
            losses.append(running_loss / batch_size)
            running_loss = 0.0           
        
        if batch >= n_iter * batch_size:
            break

    yhats = np.zeros(len(dev_set))
    model.eval()
    for i, inputs in enumerate(dev_set):
        outputs = model(inputs).expand(1, out_size)
        _, preds_tensor = torch.max(outputs, 1)
        yhats[i] = preds_tensor.item()
    
    # torch.save(model, "./net_p2.model")

    return losses, yhats, model
