import copy
import math
from typing import List, Any
import sys

import torch
import numpy as np
from mabwiser.utils import Arm, Constants

from golem.core.log import default_log

import warnings
warnings.filterwarnings("ignore")


class NeuralMAB:
    def __init__(self, arms: List[Arm],
                 seed: int = Constants.default_seed,
                 n_jobs: int = 1):
        self.arms = arms
        self.seed = seed
        self.n_jobs = n_jobs
        self.log = default_log('NeuralMAB')
        # to track when GNN needs to be updated
        self.iter = 0
        self._initial_fit(context_size=500)

    def _initial_fit(self, context_size: int):

        # params for GNN
        self._beta = 0.02
        self._lambd = 1
        self._lr = 0.0001
        self._H_q = 10
        self._interT = 1000
        self._hidden_dim = [1000, 1000]
        hid_dim_lst = self._hidden_dim
        dim_second_last = self._hidden_dim[-1] * 2

        dim_for_init = [context_size + len(self.arms)] + hid_dim_lst + [1]
        self.W0, total_dim = self._initialization(dim_for_init)
        self.LAMBDA = self._lambd * torch.eye(dim_second_last, dtype=torch.double)
        self.bb = torch.zeros(self.LAMBDA.size()[0], dtype=torch.double).view(-1, 1)

        theta = np.random.randn(dim_second_last, 1) / np.sqrt(dim_second_last)
        self.theta = torch.from_numpy(theta)

        self.THETA_action = torch.tensor([])
        self.CONTEXT_action = torch.tensor([])
        self.REWARD_action = torch.tensor([])
        self.result_neuralucb = []
        self.W = copy.deepcopy(self.W0)
        self.summ = 0

    def partial_fit(self, decisions: List[Any], contexts: List[List[Any]], rewards: List[float]):
        for decision, context, reward in zip(decisions, contexts, rewards):
            # first, calculate estimated value for different actions
            ucb = []
            bphi = []
            for a in range(0, len(self.arms)):
                temp = self._transfer(context, a, len(self.arms))
                bphi.append(temp)
                feat = self._feature_extractor(temp, self.W)
                ucb.append(torch.mm(self.theta.view(1, -1), feat) + self._beta * self._UCB(self.LAMBDA, feat))

            a_choose = decision

            self.summ += (max(ucb) - reward)
            self.result_neuralucb.append(self.summ)

            # finally update W by doing TRAIN_SE
            if np.mod(self.iter, self._H_q) == 0:
                CONTEXT_action = bphi[a_choose]
                REWARD_action = torch.tensor([reward], dtype=torch.double)
            else:
                CONTEXT_action = torch.cat((self.CONTEXT_action, bphi[a_choose]), 1)
                REWARD_action = torch.cat((self.REWARD_action, torch.tensor([reward], dtype=torch.double)), 0)

            # update LAMBDA and bb
            self.LAMBDA += torch.mm(self._feature_extractor(bphi[a_choose], self.W),
                                    self._feature_extractor(bphi[a_choose], self.W).t())
            self.bb += reward * self._feature_extractor(bphi[a_choose], self.W)
            theta, LU = torch.solve(self.bb, self.LAMBDA)

            if np.mod(self.iter, self._H_q) == 0:
                THETA_action = theta.view(-1, 1)
            else:
                THETA_action = torch.cat((self.THETA_action, theta.view(-1, 1)), 1)

            if np.mod(self.iter, self._H_q) == self._H_q - 1:
                self.log.info(f'Current regret: {self.summ}')
                self.W = self._train_with_shallow_exploration(CONTEXT_action, REWARD_action, self.W0,
                                                              self._interT, self._lr, THETA_action, self._H_q)
        self.iter += 1

    def predict(self, context: Any) -> int:
        """ Predicts which arm to pull to get maximum reward. """
        ucb = []
        bphi = []
        for a in range(0, len(self.arms)):
            temp = self._transfer(context, a, len(self.arms))
            bphi.append(temp)
            feat = self._feature_extractor(temp, self.W)
            ucb.append(torch.mm(self.theta.view(1, -1), feat) + self._beta * self._UCB(self.LAMBDA, feat))

        if self.iter < 3 * len(self.arms):
            a_choose = self.iter % len(self.arms)
        else:
            a_choose = ucb.index(max(ucb))
        return a_choose

    def predict_expectations(self, context: Any) -> List[float]:
        """ Returns expected reward for each arm. """
        ucb = []
        bphi = []
        for a in range(0, len(self.arms)):
            temp = self._transfer(context, a, len(self.arms))
            bphi.append(temp)
            feat = self._feature_extractor(temp, self.W)
            ucb.append(torch.mm(self.theta.view(1, -1), feat) + self._beta * self._UCB(self.LAMBDA, feat))
        return ucb

    @staticmethod
    def _initialization(dim):
        """ Initialization.
        dim consists of (d1, d2,...), where dl = 1 (placeholder, deprecated). """
        w = []
        total_dim = 0
        for i in range(0, len(dim) - 1):
            if i < len(dim) - 2:
                temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i + 1])
                temp = np.kron(np.eye(2, dtype=int), temp)
                temp = torch.from_numpy(temp)
                w.append(temp)
                total_dim += dim[i + 1] * dim[i] * 4
            else:
                temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i])
                temp = np.kron([[1, -1]], temp)
                temp = torch.from_numpy(temp)
                w.append(temp)
                total_dim += dim[i + 1] * dim[i] * 2

        return w, total_dim

    @staticmethod
    def _feature_extractor(x, W):
        """ Functions feature extractor.
        x is the input, dimension is d; W is the list of parameter matrices. """
        depth = len(W)
        output = x
        for i in range(0, depth - 1):
            output = torch.mm(W[i], output)
            output = output.clamp(min=0)

        output = output * math.sqrt(W[depth - 1].size()[1])
        return output

    def _gradient_loss(self, X, Y, W, THETA):
        """ Return a list of grad, satisfying that W[i] = W[i] - grad[i] ##for single context x. """
        depth = len(W)
        num_sample = Y.shape[0]
        loss = []
        grad = []
        relu = []
        output = X
        loss.append(output)
        for i in range(0, depth - 1):
            output = torch.mm(W[i], output)
            relu.append(output)
            output = output.clamp(min=0)
            loss.append(output)

        THETA_t = torch.transpose(THETA, 0, 1).view(num_sample, 1, -1)
        output_t = torch.transpose(output, 0, 1).view(num_sample, -1, 1)
        output = torch.bmm(THETA_t, output_t).squeeze().view(1, -1)

        loss.append(output)
        ####
        feat = self._feature_extractor(X, W)
        feat_t = torch.transpose(feat, 0, 1).view(num_sample, -1, 1)
        output_t = torch.bmm(THETA_t, feat_t).squeeze().view(1, -1)

        #### backward gradient propagation
        back = output_t - Y
        back = back.double()
        grad_t = torch.mm(back, loss[depth - 1].t())
        grad.append(grad_t)

        for i in range(1, depth):
            back = torch.mm(W[depth - i].t(), back)
            back[relu[depth - i - 1] < 0] = 0
            grad_t = torch.mm(back, loss[depth - i - 1].t())
            grad.append(grad_t)
        ####
        grad1 = []
        for i in range(0, depth):
            grad1.append(grad[depth - 1 - i] * math.sqrt(W[depth - 1].size()[1]) / len(X[0, :]))

        if (grad1[0] != grad1[0]).any():
            print('nan found')
            sys.exit('nan found')
        return grad1

    def _loss(self, X, Y, W, THETA):
        #### total loss
        num_sample = len(X[0, :])
        output = self._feature_extractor(X, W)
        THETA_t = torch.transpose(THETA, 0, 1).view(num_sample, 1, -1)
        output_t = torch.transpose(output, 0, 1).view(num_sample, -1, 1)
        output_y = torch.bmm(THETA_t, output_t).squeeze().view(1, -1)

        summ = (Y - output_y).pow(2).sum() / num_sample
        return summ

    def _train_with_shallow_exploration(self, X, Y, W_start, T, et, THETA, H):
        """ Gd-based model training with shallow exploration
        Dataset X, label Y. """
        W = copy.deepcopy(W_start)
        num_sample = H
        X = X[:, -H:]
        Y = Y[-H:]
        THETA = THETA[:, -H:]

        prev_loss = 1000000
        prev_loss_1k = 1000000
        for i in range(0, T):
            grad = self._gradient_loss(X, Y, W, THETA)
            if (grad[0] != grad[0]).any():
                print('nan found')
            for j in range(0, len(W) - 1):
                W[j] = W[j] - et * grad[j]

            curr_loss = self._loss(X, Y, W, THETA)
            if i % 100 == 0:
                print('------', curr_loss)
                if curr_loss > prev_loss_1k:
                    et = et * 0.1
                    print('lr/10 to', et)

                prev_loss_1k = curr_loss

            # early stopping
            if abs(curr_loss - prev_loss) < 1e-6:
                break
            prev_loss = curr_loss
        return W

    @staticmethod
    def _UCB(A, phi):
        """ Ucb term. """
        try:
            tmp, LU = torch.solve(phi, A)
        except:
            tmp = torch.Tensor(np.linalg.solve(A, phi))

        return torch.sqrt(torch.mm(torch.transpose(phi, 0, 1).double(), tmp.double()))

    @staticmethod
    def _transfer(c, a, arm_size):
        """
        Transfer an array context + action to new context with dimension 2*(__context__ + __armsize__).
        """
        action = np.zeros(arm_size)
        action[a] = 1
        c_final = np.append(c, action)
        c_final = torch.from_numpy(c_final)
        c_final = c_final.view((len(c_final), 1))
        c_final = c_final.repeat(2, 1)
        return c_final
