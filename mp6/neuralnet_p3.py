# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
    
class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        """
        1) DO NOT change the name of self.encoder & self.decoder
        2) Both of them need to be subclass of torch.nn.Module and callable, like
           output = self.encoder(input)
        """
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.y = None
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  8, kernel_size=8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=8),
            nn.ReLU(True))
        
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16, 8, kernel_size=8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,  1, kernel_size=8),
            nn.ReLU(True))
        
    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        # return self.net.parameters()
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return xhat: an (N, out_size) torch tensor of output from the network
        """
        self.y = x
        x = x.reshape(1, 1, 28, 28)       
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(1, 28 * 28)
        return x

    def step(self, x):
        # x [100, 784]
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        L = self.loss_fn(x, self.y)
        L.backward()
        return L

def fit(train_set,dev_set,n_iter,batch_size=100):
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

    start = time.time()    
    
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)
    
    # Build model
    lrate = 1e-4
    loss_fn = nn.MSELoss()
    in_size = len(train_set[0])
    out_size = 5
    model = NeuralNet(lrate, loss_fn, in_size, out_size)

    # Loss and optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.9)     
    optimizer = torch.optim.Adam(model.parameters(), lrate)

    # https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
    
    ## training part 
    model.train()
    batch = 0
    losses = []
    running_loss = 0
    for i, inputs in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = model.step(outputs)
        optimizer.step()

        batch += 1
        if batch % batch_size == 0:
            losses.append(running_loss / batch_size)
            running_loss = 0.0           
        
        if batch >= n_iter * batch_size:
            break
        
    ## evaluation part 
    model.eval()
    yhats = torch.Tensor(len(dev_set), len(dev_set[0]))
    # yhats = np.zeros(len(dev_set))    
    for i, inputs in enumerate(testloader):
        outputs = model.forward(inputs)
        yhats[i] = outputs
 
    print("Total = %s sec" % (time.time() - start))   
 
    return losses, yhats.detach().numpy(), model
