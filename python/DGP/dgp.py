# Copyright 2018 Manon Kok and Arno Soliln
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gpflow
import numpy as np
import tensorflow as tf
#import scipy.io as spio
from math import gamma
from functools import reduce

from gpflow.decors import params_as_tensors
from gpflow.decors import name_scope

from gpflow import settings
float_type = settings.dtypes.float_type
from .matrix_structures import DiagMat

from .domain import gp_domain

class DGP(gpflow.models.GPModel):
    """
    Domain GP.
    """
    def __init__(self, X, Y, ms, domain, kern, lik, Xtest):
        mf = gpflow.mean_functions.Zero()
        gpflow.models.GPModel.__init__(self, X, Y, kern=kern, likelihood=lik(), mean_function=mf)
        self.domain = domain
        self.ms = ms
        self.trainingKuf = self.domain.eigenfun(X).transpose()
        self.testKuf = self.domain.eigenfun(Xtest).transpose()
        self.predictFlag = 1
    
        assert self.Y.shape[1] == 1
        self.num_data = X.shape[0]
        self.num_input = X.shape[1]
        self.num_latent = Y.shape[1]

        self.q_mu = gpflow.params.Parameter(np.zeros((ms, 1)))
        pos = gpflow.transforms.positive
        self.q_sqrt = gpflow.params.Parameter(np.ones(ms), pos)

    @name_scope('KL')
    @params_as_tensors
    def build_KL(self):
        """
        We're working in a 'whitened' representation, so this is the KL between
        q(u) and N(0, 1)
        """
        Kuu = self.make_Kuu(self.kern, self.num_input)
        Kim = Kuu.solve(self.q_mu)
        KL = 0.5*tf.squeeze(tf.matmul(tf.transpose(Kim), self.q_mu))  # Mahalanobis term
        KL += 0.5 * Kuu.trace_KiX(tf.diag(tf.square(tf.reshape(self.q_sqrt, [-1]))))
        KL += -0.5*tf.cast(tf.size(self.q_mu), float_type)  # Constant term.
        KL += -0.5*tf.reduce_sum(tf.log(tf.square(self.q_sqrt)))  # Log det Q
        KL += 0.5*Kuu.logdet()  # Log det P
        return KL

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        # compute the mean and variance of the latent function
        self.predictFlag = 0
        f_mu, f_var = self._build_predict(self.X, full_cov=False)
        self.predictFlag = 1

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, self.Y)
        return tf.reduce_sum(E_lik) - self.build_KL()  

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, X, full_cov=False):    
        Kuf = self.make_Kuf(X)
        Kuu = self.make_Kuu(self.kern, self.num_input)
        KiKuf = Kuu.solve(Kuf)

        mu = tf.matmul(tf.transpose(KiKuf), self.q_mu)
        tmp1 = tf.expand_dims(self.q_sqrt, 1) * KiKuf
        if full_cov:
            # Kff
            var = self.kern.K(X)
            var = var + tf.matmul(tf.transpose(tmp1), tmp1)  # Projected variance Kfu Ki S Ki Kuf
            var = var - tf.matmul(tf.transpose(Kuf), KiKuf)  # Qff
            var = tf.expand_dims(var, 2)

        else:
            var = self.kern.Kdiag(X)  # Kff
            var = var + tf.reduce_sum(tf.square(tmp1), 0)  # Projected variance Kfu Ki [A + WWT] Ki Kuf
            var = var - tf.reduce_sum(Kuf * KiKuf, 0)  # Qff
            var = tf.reshape(var, (-1, 1))
        return mu, var
    
    #@params_as_tensors
    def make_Kuf(self, X):
        """
        Make a representation of the Kuf matrices.
        """
        if self.predictFlag == 1:
            phi = self.testKuf
        else:
            phi = self.trainingKuf
        return phi
    
    def make_Kuu(self, kern, input_dim):
        """
        Make a representation of the Kuu matrices.
        """
        # Extract eigenvalues
        eigenValues = tf.constant(self.domain.eigenval())
        
        # Compute prior: magnSigma2*sqrt(2*pi)^d*lengthScale^d*exp(-w.^2*lengthScale^2/2)
        # Note that in Matlab the function is called on sqrt(lambda)
        # In this implementation, the square is instead omitted and the 
        # function works directly on lambda
        if isinstance(kern, gpflow.kernels.RBF):
            omega = kern.variance * \
                np.power(2.*np.pi, input_dim/2.) * \
                tf.pow(kern.lengthscales,input_dim) *\
                tf.exp(-eigenValues * tf.square(kern.lengthscales)/2.)    
        elif isinstance(kern, gpflow.kernels.Matern12) or \
            isinstance(kern, gpflow.kernels.Matern32) or \
            isinstance(kern, gpflow.kernels.Matern52):
            # magnSigma2 * (2^d * pi^(d/2) * gamma(nu + d/2) * (2*nu)^nu / (gamma(nu) * lengthScale^(2*nu))) * ...
            # (2*nu/lengthScale^2 + w.^2).^(-nu-d/2)
            if isinstance(kern, gpflow.kernels.Matern12):
                nu = 1./2.
            elif isinstance(kern, gpflow.kernels.Matern32):   
                nu = 3./2.
            else:
                nu = 5./2.
            tmp = 2.*nu / tf.pow(kern.lengthscales,2.) + eigenValues
            omega = kern.variance * \
                np.power(2., input_dim) * \
                np.power(np.pi, input_dim/2.) * \
                gamma(nu+input_dim/2.) * \
                np.power(2.*nu, nu) / \
                gamma(nu) / \
                tf.pow(kern.lengthscales,2.*nu) * \
                tf.pow(tmp, -nu-input_dim/2.)
        else:
            raise NotImplementedError
            
        return DiagMat(1. / omega)  
