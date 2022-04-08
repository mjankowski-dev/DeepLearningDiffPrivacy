import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Normalization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm # gives progress bar when loading

import scipy.stats as sc
import time
from sklearn.preprocessing import StandardScaler
import random


class moment_accountant():
    def __init__(self, seed, params: dict, deltaFixed=False, epsFixed=False, debug=False, **kwargs):

        # moment accountant hyperparameters
        self.mixN = 1000  # samples for mixture of moment arrays
        self.debug = debug
        self.seed = seed

        # constants
        self.maxOrder = params["maxOrder"]  # maximum moment order
        self.lambd = np.arange(1, self.maxOrder + 1)
        # self.lambd = np.array([0])
        self.lambdaN = len(self.lambd)  # number of lambdas to check
        print(f"Maxorder = {self.maxOrder}, with order array:")
        print(self.lambd)
        self.sigma = params["sigma"]
        self.q = params["q"]
        self.T = params["T"]

        # booleans
        self.deltaFixed = deltaFixed
        self.epsFixed = epsFixed

        if self.deltaFixed:
            if self.epsFixed:
                raise Exception("Choose ONLY epsilon or delta as fixed")
            # in case delta is held fixed
            print("keeping delta fixed")
            self.delta = kwargs["delta"]
            self.th_epsilon = kwargs["th_epsilon"]
        elif self.epsFixed:
            # in case epsilon is held fixed
            print("keeping epsilon fixed")
            self.epsilon = kwargs["epsilon"]
            self.th_delta = kwargs["th_delta"]
        else:
            raise Exception("Choose EITHER epsilon or delta as fixed")

        # self.e1_mu0, self.e1_mu, self.e2_mu0, self.e2_mu = self._setup_mixNormNP() # obtain random sample arrays NUMPY
        self.e1_mu0, self.e1_mu, self.e2_mu0, self.e2_mu = self._setup_mixNormTF()  # obtain random sample arrays TENSORFLOW
        self.alpha = self._compute_moment(self.lambd)

        # initializations
        self.alphaSum = 0  # moment
        self.lambdArgmin = []
        self.iterations = 0
        self.deltaList = []
        self.epsList = []
        # =================================
        print("moment accountant setup complete")

    def _setup_mixNormTF(self):
        '''
        Mixture of gaussians by tensorflow, so no assumptions used
        https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Mixture
        '''

        '''
        CHECKED - WORKING CORRECTLY
        '''
        # setup normal dis & mixture of normals arrays
        np.random.seed(self.seed)  # set seed
        mu_0 = np.random.normal(0, self.sigma, (self.mixN))
        # analytical mean & var for mix

        # setup gaussian mixture
        dismix_mu = tfp.distributions.Mixture(cat=tfp.distributions.Categorical(probs=[1. - self.q, self.q]),
                                              components=[tfp.distributions.Normal(loc=0., scale=self.sigma),
                                                          tfp.distributions.Normal(loc=1., scale=self.sigma)])
        mu = dismix_mu.sample(sample_shape=(self.mixN), seed=self.seed).numpy()

        # find pdf values for z's
        e1_mu0 = sc.norm.pdf(mu_0, loc=0, scale=self.sigma)
        e1_mu = dismix_mu.prob(mu_0).numpy()

        e2_mu0 = sc.norm.pdf(mu, loc=0, scale=self.sigma)
        e2_mu = dismix_mu.prob(mu).numpy()

        return e1_mu0, e1_mu, e2_mu0, e2_mu

    def _compute_moment(self, lambd: np.array):
        '''
        CHECKED - WORKING CORRECTLY
        '''
        # computes unbiased expectation for E1 & E2, then the moment alpha
        lambd = np.broadcast_to(np.expand_dims(lambd, -1), (self.lambdaN, self.mixN))  # broadcast
        # E1 = 1/self.mixN*np.sum(np.transpose(np.power(np.transpose(self.mu_0/self.mu),lambd)), axis = 0)
        # E2 = 1/self.mixN*np.sum(np.transpose(np.power(np.transpose(self.mu/self.mu_0),lambd)), axis = 0)
        E1 = np.nanmean(np.transpose(np.power(np.transpose(self.e1_mu0 / self.e1_mu), lambd)), axis=0)
        '''
        note that due to setup E1 will always be < 1 since denom > num always
        '''
        E2 = np.nanmean(np.transpose(np.power(np.transpose(self.e2_mu / self.e2_mu0), lambd)), axis=0)
        '''
        note that due to setup E2 will always be > 1 since denom < num always
        '''
        # alpha = np.log(np.maximum(E1,E2))
        alpha = np.log(np.maximum(E1, E2))
        return alpha

    def compute_deltaEps(self):
        # tail bound
        # alpha = self._compute_moment(self.lambd) + self.alpha # note that this is the log moment!
        alpha = self.alphaSum + self.alpha  # note that this is the log moment!
        self.alphaSum = alpha  # update moment
        if self.epsFixed:
            # epsilon is kept fixed, compute delta
            epsilon = self.epsilon
            delta = np.min(np.exp(alpha - self.lambd * epsilon))
            # TODO remove inf or nan <- does not seem necessary
            if self.debug:
                ind = np.argmin(np.exp(alpha - self.lambd * epsilon))
                self.lambdArgmin.append(self.lambd[ind])
        if self.deltaFixed:
            # delta is kept fixed, compute epsilon
            delta = self.delta
            epsilon = (alpha - np.log(delta)) / self.lambd
            if self.debug:
                ind = np.argmin(epsilon)
                self.lambdArgmin.append(self.lambd[ind])
            epsilon = np.min(epsilon)
            # TODO remove inf or nan <- does not seem necessary
        self.epsList.append(epsilon)
        self.deltaList.append(delta)
        return delta, epsilon

    def check_thresholds(self, delta: float, epsilon: float):
        go = True
        if self.epsFixed:
            if self.th_delta < delta:
                # delta threshold exceeded
                go = False
        if self.deltaFixed:
            if self.th_epsilon < epsilon:
                # epsilon threshold exceeded
                go = False
        return go

    def plot_traces(self):
        if len(self.epsList) == 0:
            raise Exception("Apply iterations on accountant instance before calling this function")
        elif not self.debug:
            raise Exception("Debug was set to false, thus not all relevant data was collected")
        else:
            # gather data
            epsilon = np.array(self.epsList)
            delta = np.array(self.deltaList)
            lambdas = np.array(self.lambdArgmin)
            print(f"Delta fixed = {self.deltaFixed}| Last delta = {delta[-1]}")
            print(f"Epsilon fixed = {self.epsFixed}| Last epsilon = {epsilon[-1]}")
            print("Fixed parameter will not be plotted \n NOTE: Iteration arrays are returned")
            # plotting
            iterations = np.arange(0, len(epsilon))
            plt.figure()
            if self.deltaFixed:
                plt.plot(iterations, epsilon, label='epsilon')
            else:
                plt.plot(iterations, delta, label='delta')
            # plt.legend(loc='upper left')
            plt.title("epsilon or delta over iterations")

            plt.figure()
            plt.plot(iterations, lambdas)
            plt.title("Lambda chosen over iterations")

        return delta, epsilon, lambdas

