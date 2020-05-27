import gym
import numpy as np
import torch
from torch import nn

import utils
from policies import QPolicy


def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    # pass
    hiddensize = 50 # 24
    model = nn.Sequential(
        nn.Linear(statesize, hiddensize),
        nn.LeakyReLU(),
        nn.Linear(hiddensize, hiddensize * 2),
        nn.LeakyReLU(),
        nn.Linear(hiddensize * 2, actionsize)
    )
    return model


class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate 
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.from_numpy(state).type(torch.FloatTensor)
            qvals = self.model(states)
        return qvals.numpy()

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        qvals = self.model(torch.Tensor(state))
        if done:
            qvals[action] = reward
        else: 
            # Update network weights using the last step only
            qvals_next = self.model(torch.Tensor(next_state))
            qvals[action] = reward + self.gamma * torch.max(qvals_next).item()

        # Update network weights            
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, qvals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def save(self, outpath):
        """
        saves the model at the specified outpath
        """        
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    # Out hyperparameters
    args.model = 'models/dqn.model'
    args.episodes = 150
    args.epsilon = 0.3
    args.epsilon_decay_factor = 0.99
    args.lr = 0.001
    args.gamma = 0.90
    
    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)
    utils.qlearn(env, policy, args)
    torch.save(policy.model, args.model)
    
    # From here, take from mp7.py
    # Environment (a Markov Decision Process model)
    # Q Model
    
    model = utils.loadmodel(args.model, env, statesize, actionsize)
    print("Model: {}".format(model))

    # Rollout
    _, rewards = utils.rollout(env, model, args.episodes, args.epsilon, render=True)

    # Report
    #Evaluate total rewards for MountainCar environment
    score = np.array([np.array(rewards) > -200.0]).sum()
    print('Score: ' + str(score) + '/' + str(args.episodes))
    