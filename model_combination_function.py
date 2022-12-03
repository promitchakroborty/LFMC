import math
import copy
import numpy as np
import scipy.integrate as integrate

from sklearn.gaussian_process import GaussianProcessRegressor as GPR_algo
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as Noise


class model_handler:

    def __init__(self, ndim, nmodels, lf_list, hf, surr_build, sel_strat, fail_thresh, train_data, train_val, ntrain,
                 closecheck, random_state):

        # Initializing Surrogate Build
        self.surr_build = surr_build        # Mode 'avg' or 'sel'
        self.sel_strat = sel_strat          # S for Stochastic, D for Deterministic
        self.random_state = random_state
        if self.random_state is None:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(self.random_state)

        # initializing
        self.ndim = ndim
        self.nmodels = nmodels
        self.ntrain = ntrain
        self.lf_list = lf_list
        self.hf = hf
        self.fail_thresh = fail_thresh
        self.train_data = train_data
        self.train_val = train_val
        self.closecheck = closecheck

        self.lf_model_nparams = np.zeros(nmodels)
        self.AICc = np.zeros(nmodels)
        self.delta = np.zeros(nmodels)
        self.global_model_probs = np.zeros(nmodels)
        self.model_probs = np.zeros(nmodels)
        self.corrections = np.zeros(nmodels)
        self.std_devs = np.zeros(nmodels)
        self.surrogate_response_value = 0.0
        self.u = 0.0

        # Training GPs and calculating AIC
        kernel = C(constant_value_bounds=(1e-5, 1e5))*RBF(length_scale=np.ones(ndim), length_scale_bounds=(1e-5, 1e5))\
                 + Noise(noise_level=1, noise_level_bounds=(1e-5, 1e5))
        nparams = 1+ndim+1
        self.K = []
        self.kernels = []
        for j in range(nmodels):
            self.K.append(GPR_algo(kernel=kernel, n_restarts_optimizer=20, random_state=69))
            self.K[j].fit(train_data[j], train_val[j])
            self.kernels.append(copy.deepcopy(self.K[j].kernel_))
            self.lf_model_nparams[j] = nparams
            self.AICc[j] = (-2 * self.K[j].log_marginal_likelihood_value_) + (2 * self.lf_model_nparams[j]) \
                           + ((2 * self.lf_model_nparams[j] * (self.lf_model_nparams[j] + 1)) /
                              (ntrain[j] - self.lf_model_nparams[j] - 1))

        # Calculating model probabilities
        for j in range(nmodels):
            self.delta[j] = self.AICc[j] - np.min(self.AICc)
            self.global_model_probs[j] = np.exp(-0.5 * self.delta[j])

        probnorm_global = np.sum(self.global_model_probs)
        for j in range(nmodels):
            self.global_model_probs[j] = self.global_model_probs[j] / probnorm_global

        if surr_build == 'sel':
            if self.sel_strat == 'D':
                self.sel_idx = int(-1)
            elif self.sel_strat == 'S':
                self.sel_idx = int(-1)
            else:
                raise ValueError('Nonexistent surrogate build')
            self.modelval = np.zeros(1)
            self.responses = np.zeros(1)
        elif surr_build == 'avg':
            self.modelval = np.zeros(nmodels)
            self.responses = np.zeros(nmodels)
            self.sel_idx = int(-1)
        else:
            self.sel_idx = int(-1)

    @staticmethod
    def probability_integrand(z, mu, sig, idx, N):
        res = 1.0
        for i in range(N):
            if i == idx:
                res = res * (1 / (sig[i] * ((2 * np.pi) ** 0.5))) * (np.exp(-0.5 * (((z - mu[i]) / sig[i]) ** 2))
                                                                     + np.exp(-0.5 * (((z + mu[i]) / sig[i]) ** 2)))
            else:
                res = res * (1 - (0.5 * (math.erf((z - mu[i]) / (sig[i] * (2 ** 0.5)))
                                         + math.erf((z + mu[i]) / (sig[i] * (2 ** 0.5))))))
        return res

    def surrogate_parameter_evaluator(self, pt):
        # Calculate model predictions
        for model_idx in range(self.nmodels):
            self.corrections[model_idx], self.std_devs[model_idx] =\
                self.K[model_idx].predict(np.atleast_2d(pt), return_std=True)

        l_lim = 0
        r_lim = math.inf

        # Calculate model probabilities
        self.model_probs = np.zeros(self.nmodels)
        for model_idx in range(self.nmodels):
            l_lim = np.max([0, np.abs(self.corrections[model_idx]) - (5 * self.std_devs[model_idx])])
            r_lim = np.abs(self.corrections[model_idx]) + (5 * self.std_devs[model_idx])
            f = lambda z: self.probability_integrand(z=z, mu=self.corrections, sig=self.std_devs, idx=model_idx,
                                                     N=self.nmodels)
            [self.model_probs[model_idx], throwaway] = integrate.quad(f, l_lim, r_lim)
        probnorm = np.sum(self.model_probs)
        self.model_probs = self.model_probs / probnorm

        # Select model(s)
        if self.random_state is None:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(int(self.random_state + self.ntrain[self.sel_idx]))
        if self.surr_build == 'sel':
            if self.sel_strat == 'D':
                self.sel_idx = int(np.nonzero(self.model_probs == np.max(self.model_probs))[0][0])
                # int(np.where(self.model_probs == np.max(self.model_probs))[0])
            elif self.sel_strat == 'S':
                self.sel_idx = int(rng.choice(np.arange(self.nmodels), p=self.model_probs, size=1))
            else:
                raise ValueError('Nonexistent surrogate build')
        elif self.surr_build == 'avg':
            self.sel_idx = int(-1)
        else:
            self.sel_idx = int(-1)

        return self.model_probs

    def corrected_model_evaluator(self, pt):
        # Evaluate model(s)
        if self.surr_build == 'avg':
            for model_idx in range(self.nmodels):
                self.modelval[model_idx] = self.lf_list[model_idx](pt)
                self.responses[model_idx] = self.modelval[model_idx] + self.corrections[model_idx]
        elif self.surr_build == 'sel':
            self.modelval[0] = self.lf_list[self.sel_idx](pt)
            self.responses[0] = self.modelval[0] + self.corrections[self.sel_idx]
        else:
            raise ValueError('Nonexistent surrogate build')
        return self.responses, self.std_devs

    def surrogate(self):
        self.surrogate_response_value = 0.0
        if self.surr_build == 'avg':
            for model_idx in range(self.nmodels):
                self.surrogate_response_value += self.model_probs[model_idx]*self.responses[model_idx]
        elif self.surr_build == 'sel':
            self.surrogate_response_value = self.responses[0]
        else:
            raise ValueError('Nonexistent surrogate build')
        return self.surrogate_response_value

    def learning_function(self, thresh):
        self.u = 0
        if self.surr_build == 'avg':
            sig = 0
            for model_idx in range(self.nmodels):
                # Averaging u-values
                # self.u += self.model_probs[model_idx] *\
                #           (abs(self.responses[model_idx] - thresh) / abs(self.std_devs[model_idx]))
                sig += (self.model_probs[model_idx] * self.std_devs[model_idx])**2
            self.u = ((abs(self.surrogate_response_value - thresh)) / abs(sig**0.5))
        elif self.surr_build == 'sel':
            self.u = ((abs(self.responses[0] - thresh)) / abs(self.std_devs[self.sel_idx]))
        else:
            raise ValueError('Nonexistent surrogate build')
        return self.u

    def retrainer(self, pt, hfval):
        if self.surr_build == 'avg':
            for model_idx in range(self.nmodels):
                checkflag = 0
                for checkcounter in range(int(self.ntrain[model_idx])):
                    marker = 0
                    for dimcounter in range(self.ndim):
                        if abs(self.train_data[model_idx][checkcounter, dimcounter] - pt[dimcounter]) <= self.closecheck:
                            marker += 1
                    if marker == self.ndim:
                        checkflag = 1
                        break
                    else:
                        checkflag = 0
                if checkflag == 0:
                    self.train_data[model_idx] = np.append(self.train_data[model_idx], np.copy([pt]), axis=0)
                    corr = hfval - self.modelval[model_idx]
                    self.ntrain[model_idx] += 1
                    self.train_val[model_idx] = np.append(self.train_val[model_idx], np.copy([corr]), axis=0)
                    self.K[model_idx] = GPR_algo(kernel=self.kernels[model_idx], n_restarts_optimizer=20,
                                                 random_state=69)
                    self.K[model_idx].fit(self.train_data[model_idx], self.train_val[model_idx])
                    self.kernels[model_idx] = copy.deepcopy(self.K[model_idx].kernel_)
                    self.AICc[model_idx] = (-2 * self.K[model_idx].log_marginal_likelihood_value_) +\
                                           (2 * self.lf_model_nparams[model_idx]) +\
                                           ((2 * self.lf_model_nparams[model_idx] *
                                             (self.lf_model_nparams[model_idx] + 1)) /
                                            (self.ntrain[model_idx] - self.lf_model_nparams[model_idx] - 1))
            for j in range(self.nmodels):
                self.delta[j] = self.AICc[j] - np.min(self.AICc)
                self.global_model_probs[j] = np.exp(-0.5 * self.delta[j])
            probnorm_global = np.sum(self.global_model_probs)
            for j in range(self.nmodels):
                self.global_model_probs[j] = self.global_model_probs[j] / probnorm_global
        elif self.surr_build == 'sel':
            if self.random_state is None:
                rng = np.random.default_rng(None)
            else:
                rng = np.random.default_rng(int(self.random_state + self.ntrain[self.sel_idx]))
            checkflag = 0
            for checkcounter in range(int(self.ntrain[self.sel_idx])):
                marker = 0
                for dimcounter in range(self.ndim):
                    if abs(self.train_data[self.sel_idx][checkcounter, dimcounter] - pt[dimcounter]) <= self.closecheck:
                        marker += 1
                if marker == self.ndim:
                    checkflag = 1
                    break
                else:
                    checkflag = 0
            if checkflag == 0:
                self.train_data[self.sel_idx] = np.append(self.train_data[self.sel_idx], np.copy([pt]), axis=0)
                corr = hfval - self.modelval[0]
                self.ntrain[self.sel_idx] += 1
                self.train_val[self.sel_idx] = np.append(self.train_val[self.sel_idx], np.copy([corr]), axis=0)
                self.K[self.sel_idx] = GPR_algo(kernel=self.kernels[self.sel_idx], n_restarts_optimizer=20,
                                                random_state=69)
                self.K[self.sel_idx].fit(self.train_data[self.sel_idx], self.train_val[self.sel_idx])
                self.kernels[self.sel_idx] = copy.deepcopy(self.K[self.sel_idx].kernel_)
                self.AICc[self.sel_idx] = (-2 * self.K[self.sel_idx].log_marginal_likelihood_value_) +\
                                          (2 * self.lf_model_nparams[self.sel_idx]) +\
                                          ((2 * self.lf_model_nparams[self.sel_idx] *
                                            (self.lf_model_nparams[self.sel_idx] + 1)) /
                                           (self.ntrain[self.sel_idx] - self.lf_model_nparams[self.sel_idx] - 1))
            for j in range(self.nmodels):
                self.delta[j] = self.AICc[j] - np.min(self.AICc)
                self.global_model_probs[j] = np.exp(-0.5 * self.delta[j])
            probnorm_global = np.sum(self.global_model_probs)
            for j in range(self.nmodels):
                self.global_model_probs[j] = self.global_model_probs[j] / probnorm_global
        else:
            raise ValueError('Nonexistent surrogate build')
        return