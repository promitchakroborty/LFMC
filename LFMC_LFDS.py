import time

import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

from sklearn.metrics import r2_score as r2
from scipy.interpolate import griddata

from UQpy.sampling.mcmc import Stretch

import model_defs as model_defs
import model_combination_function as mcf

random_seed = None

# Defining input distribution
x_dist = stats.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1.0, 0.0], [0.0, 1.0]]))
target_logpdf = lambda x: x_dist.logpdf(x)

std_norm = stats.norm(loc=0.0, scale=1.0)

# Problem thresholds
# in_thresh = 5.0
fail_thresh = 0.0    # model_defs.hfeval([in_thresh, in_thresh])
u_thresh = 2.0

# Crude Monte Carlo pf calculation
target_pf_data = x_dist.rvs(size=(10**6), random_state=random_seed)
target_pf = 0
for i in range((10**6)):
    if fail_thresh - model_defs.hfeval(target_pf_data[i]) <= 0:
        target_pf += 1
print('Target pf = ' + str(target_pf/(10**6)))
print('Failure threshold = ' + str(fail_thresh))

# Model Definitions and initializations
nmodels = 4
hf_model = model_defs.hfeval
lf_models = [model_defs.lf1eval, model_defs.lf2eval, model_defs.lf3eval, model_defs.lf4eval]
surr_build = 'sel'
sel_strat = 'D'
lotp_for_pf = True

algo_start_time = time.time()

# Initial/Training sample
init_size = 20
init_sample = x_dist.rvs(size=init_size, random_state=random_seed)

# Initializing MCMC Parameters and output variables
nsamples = 20000
ndim = 2
nchains = int(0.1*nsamples)
nspc = int(nsamples/nchains)
print(nchains)
samples = np.zeros((nspc, nchains, ndim))
res_us = -1*np.ones((nspc, nchains))
response = np.zeros((nspc, nchains))
point_fail = np.zeros((nspc, nchains))
response_thresh_arr = np.zeros(init_size)
all_samples = []
all_response = []
all_us = []
all_point_fail = []
sel_sample_plot = [np.zeros((1, ndim))]*int(nmodels+1)
sel_idx_storer = nmodels+2

hf_count = init_size
pf = 1.0
inter_pf = 0.0
COV = 0.0
delf = 0.0

cumhf = np.array([0])
hfindex = np.array([0])

cumlf = np.zeros((1, nmodels))
lf_count = np.zeros((1, nmodels))

noise_arr = np.zeros((1, nmodels))
AIC_prob_arr = np.zeros((1, nmodels))
AIC_arr = np.zeros((1, nmodels))

store_thresh = []
store_fail = []
store_delf = []

subcount = 0

closecheck = 10 ^ -9
inter_thresh = -1*math.inf

# Initial training data
train_part = np.zeros((1, nmodels))
train_size = np.zeros(nmodels)
train_data = []
train_val = []
train_rej = np.zeros((1, nmodels), dtype=int)
rejtemp = np.zeros((1, nmodels), dtype=int)
for j in range(nmodels):
    train_data += [np.zeros((init_size, ndim))]
    train_val += [np.zeros(init_size)]

for i in range(init_size):
    samples[0, i, :] = init_sample[i]
    response[0, i] = hf_model(samples[0, i, :])
    res_us[0, i] = math.inf
    for j in range(nmodels):
        train_data[j][i, :] = samples[0, i, :]
        train_val[j][i] = response[0, i] - lf_models[j](samples[0, i, :])
        train_size[j] += 1
    response_thresh_arr[i] = response[0, i]

inter_thresh = np.minimum(np.percentile(response_thresh_arr, 90), fail_thresh)

# Creating handler object to train GPs and calculate stuff
handler_obj = mcf.model_handler(ndim=ndim, nmodels=nmodels, lf_list=lf_models, hf=hf_model, surr_build=surr_build,
                                sel_strat=sel_strat, fail_thresh=fail_thresh, train_data=train_data,
                                train_val=train_val, ntrain=train_size, closecheck=closecheck, random_state=random_seed)

for m in range(nmodels):
    noise_arr[0, m] = float(copy.deepcopy(handler_obj.K[m].kernel_.get_params()['k2__noise_level']))
    AIC_prob_arr[0, m] = float(copy.deepcopy(handler_obj.global_model_probs[m]))
    AIC_arr[0, m] = float(copy.deepcopy(handler_obj.AICc[m]))

model_prob_arr = np.copy(handler_obj.model_probs).reshape((1, nmodels))

train_part = np.append(train_part, [np.copy(handler_obj.ntrain)], axis=0)

for j in range(nmodels):
    print('Kriging model ' + str(j) + ' retrain ' + str(handler_obj.ntrain[j] - init_size) + ' parameter = '
          + str(handler_obj.K[j].kernel_))
print(handler_obj.model_probs)

# First subset

# MCMC Step
for j in range(nspc):
    for i in range(nchains):

        print('Subset ' + str(subcount) + ', Sample ' + str([j, i]))

        if j == 0 and i < init_size:
            continue

        # Generating sample and calculating surrogate responses
        samples[j, i, :] = x_dist.rvs(size=1)  #, random_state=random_seed+j+(i*nchains)).reshape((1, -1))
        handler_obj.surrogate_parameter_evaluator(pt=samples[j, i, :])

        model_res, model_stds = handler_obj.corrected_model_evaluator(pt=samples[j, i, :])
        response[j, i] = handler_obj.surrogate()

        cumlf[0, handler_obj.sel_idx] += 1
        sel_idx_storer = int(handler_obj.sel_idx)

        # Checking surrogate sufficiency
        u = handler_obj.learning_function(thresh=inter_thresh)
        res_us[j, i] = u

        # HF call and retraining for insufficient surrogate
        if res_us[j, i] < u_thresh:  # u < u_thresh:
            response[j, i] = hf_model(samples[j, i, :])
            res_us[j, i] = math.inf
            hf_count += 1
            sel_idx_storer = int(nmodels)

            cumlf[0, handler_obj.sel_idx] -= 1

            cumhf = np.append(cumhf, hf_count)
            hfindex = np.append(hfindex, ((j * nchains) + i))

            start_time = time.time()
            handler_obj.retrainer(pt=samples[j, i, :], hfval=response[j, i])

            noise_arr = np.append(noise_arr, np.zeros((1, nmodels)), axis=0)
            AIC_prob_arr = np.append(AIC_prob_arr, np.zeros((1, nmodels)), axis=0)
            AIC_arr = np.append(AIC_arr, np.zeros((1, nmodels)), axis=0)
            for m in range(nmodels):
                noise_arr[-1, m] = float(copy.deepcopy(handler_obj.K[m].kernel_.get_params()['k2__noise_level']))
                AIC_prob_arr[-1, m] = float(copy.deepcopy(handler_obj.global_model_probs[m]))
                AIC_arr[-1, m] = float(copy.deepcopy(handler_obj.AICc[m]))

            for mi in range(nmodels):
                print('Kriging model ' + str(mi) + ' retrain ' + str(handler_obj.ntrain[mi] - init_size) +
                      ' parameter = ' + str(handler_obj.K[mi].kernel_))
            print(handler_obj.model_probs, np.sum(handler_obj.model_probs))
            print('Train time = ' + str(time.time() - start_time))

        sel_sample_plot[sel_idx_storer] = np.append(sel_sample_plot[sel_idx_storer], [np.copy(samples[j, i, :])],
                                                    axis=0)

        model_prob_arr = np.append(model_prob_arr, [np.copy(handler_obj.model_probs)], axis=0)

        response_thresh_arr = np.append(response_thresh_arr, np.copy([response[j, i]]), axis=0)
        inter_thresh = np.minimum(np.percentile(response_thresh_arr, 90), fail_thresh)

        lf_count = np.append(lf_count, np.copy(cumlf), axis=0)

for j in range(nspc):
    for i in range(nchains):
        if inter_thresh - response[j, i] <= 0.0:
            if lotp_for_pf:
                point_fail[j, i] = std_norm.cdf(res_us[j, i])
            else:
                point_fail[j, i] = 1
        else:
            if lotp_for_pf:
                point_fail[j, i] = std_norm.cdf(-1*res_us[j, i])
            else:
                point_fail[j, i] = 0
fail_count = np.sum(point_fail)
inter_pf = (fail_count/nsamples)
store_fail += [inter_pf]
delf = ((1 - inter_pf)/(inter_pf*nsamples))**0.5
store_delf += [delf]

# Storing data in global arrays and updating subset counters
prev_thresh = inter_thresh
all_samples += [samples.copy()]
all_response += [response.copy()]
all_us += [res_us.copy()]
all_point_fail += [point_fail.copy()]
train_part = np.append(train_part, [np.copy(handler_obj.ntrain)], axis=0)

subcount += 1

# Calculating subset pf and seeding next subset
response_thresh_arr = np.zeros(nchains)
samples = np.zeros((nspc, nchains, ndim))
res_us = -1*np.ones((nspc, nchains))
response = np.zeros((nspc, nchains))
point_fail = np.zeros((nspc, nchains))

seed_count = nchains
for j in range(nspc):
    for i in range(nchains):
        if inter_thresh - all_response[-1][j, i] <= 0.0:
            if seed_count > 0:
                temps = all_samples[-1][j, i, :]
                tempr = all_response[-1][j, i]
                tempu = all_us[-1][j, i]
                tempp = all_point_fail[-1][j, i]
                samples[0, seed_count - 1, :] = temps
                response[0, seed_count - 1] = tempr
                res_us[0, seed_count - 1] = tempu
                point_fail[0, seed_count - 1] = tempp
                response_thresh_arr[seed_count - 1] = tempr
                seed_count -= 1

inter_thresh = np.minimum(np.percentile(response_thresh_arr, 90), fail_thresh)

pf = pf*inter_pf
COV += delf**2
print('\nFail for thresh = ' + str(fail_count / nsamples))
print('Intermediate Failure Threshold = ' + str(prev_thresh))
store_thresh += [prev_thresh]

# Subset Simulation
while prev_thresh < fail_thresh:
    print('\n\nIntermediate Failure Threshold = ' + str(prev_thresh))

    # MCMC step
    for j in range(nspc - 1):

        # Generating samples_raw
        step = Stretch(log_pdf_target=target_logpdf, seed=samples[j, :, :].tolist(), nsamples=3 * nchains)  #, random_state=random_seed+j)
        samples[j + 1, :, :] = step.samples[2 * nchains:]

        for i in range(nchains):

            print('Subset ' + str(subcount) + ', Sample ' + str([j, i]))

            # Calculating surrogate responses
            handler_obj.surrogate_parameter_evaluator(pt=samples[j + 1, i, :])

            model_res, model_stds = handler_obj.corrected_model_evaluator(pt=samples[j + 1, i, :])
            response[j + 1, i] = handler_obj.surrogate()

            cumlf[0, handler_obj.sel_idx] += 1
            sel_idx_storer = int(handler_obj.sel_idx)

            # Checking surrogate sufficiency
            u = handler_obj.learning_function(thresh=inter_thresh)
            res_us[j + 1, i] = u

            rejf = 0

            # HF call and retraining for insufficient surrogate
            if res_us[j + 1, i] < u_thresh:  # u < u_thresh:
                response[j + 1, i] = hf_model(samples[j + 1, i, :])
                res_us[j + 1, i] = math.inf
                hf_count += 1

                cumlf[0, handler_obj.sel_idx] -= 1
                sel_idx_storer = int(nmodels)

                cumhf = np.append(cumhf, hf_count)
                hfindex = np.append(hfindex, ((subcount * nsamples) + (j * nchains) + i))

                start_time = time.time()
                handler_obj.retrainer(pt=samples[j + 1, i, :], hfval=response[j + 1, i])

                noise_arr = np.append(noise_arr, np.zeros((1, nmodels)), axis=0)
                AIC_prob_arr = np.append(AIC_prob_arr, np.zeros((1, nmodels)), axis=0)
                AIC_arr = np.append(AIC_arr, np.zeros((1, nmodels)), axis=0)
                for m in range(nmodels):
                    noise_arr[-1, m] = float(copy.deepcopy(handler_obj.K[m].kernel_.get_params()['k2__noise_level']))
                    AIC_prob_arr[-1, m] = float(copy.deepcopy(handler_obj.global_model_probs[m]))
                    AIC_arr[-1, m] = float(copy.deepcopy(handler_obj.AICc[m]))

                for mi in range(nmodels):
                    print('Kriging model ' + str(mi) + ' retrain ' + str(handler_obj.ntrain[mi] - init_size) +
                          ' parameter = ' + str(handler_obj.K[mi].kernel_))
                    rejtemp[0, mi] = int(handler_obj.ntrain[mi] - 1)
                print(handler_obj.model_probs, np.sum(handler_obj.model_probs))
                print('Train time = ' + str(time.time() - start_time))
                print([j, i])
                rejf = 1

            # Checking for subset threshold
            if prev_thresh - response[j + 1, i] > 0:
                samples[j + 1, i, :] = np.copy(samples[j, i, :])
                response[j + 1, i] = np.copy(response[j, i])
                res_us[j + 1, i] = np.copy(res_us[j, i])
                if rejf == 1:
                    train_rej = np.append(train_rej, np.copy(rejtemp), axis=0)
            else:
                sel_sample_plot[sel_idx_storer] = np.append(sel_sample_plot[sel_idx_storer],
                                                            [np.copy(samples[j + 1, i, :])], axis=0)

            model_prob_arr = np.append(model_prob_arr, [np.copy(handler_obj.model_probs)], axis=0)

            response_thresh_arr = np.append(response_thresh_arr, np.copy([response[j + 1, i]]), axis=0)
            inter_thresh = np.minimum(np.percentile(response_thresh_arr, 90), fail_thresh)

            lf_count = np.append(lf_count, np.copy(cumlf), axis=0)

    for j in range(nspc):
        for i in range(nchains):
            if inter_thresh - response[j, i] <= 0.0:
                if lotp_for_pf:
                    point_fail[j, i] = std_norm.cdf(res_us[j, i])
                else:
                    point_fail[j, i] = 1
            else:
                if lotp_for_pf:
                    point_fail[j, i] = std_norm.cdf(-1 * res_us[j, i])
                else:
                    point_fail[j, i] = 0
    fail_count = np.sum(point_fail)
    inter_pf = (fail_count / nsamples)
    store_fail += [inter_pf]
    gamma = 0
    for lam in range(1, nspc - 1):
        R0 = inter_pf * (1 - inter_pf)
        Rs = 0
        for l in range(1, nchains + 1):
            for m in range(1, nspc + 1 - lam):
                Rs += point_fail[m - 1, l - 1] * point_fail[m - 1 + lam, l - 1]
        Rs = (Rs / (nsamples - (lam * nchains))) - (inter_pf ** 2)
        gamma += (2 * (1 - (lam / nspc)) * (Rs / R0))
    delf = (((1 - inter_pf) / (inter_pf * nsamples)) * (1 + gamma)) ** 0.5
    store_delf += [delf]

    # Storing data in global arrays and updating subset counters
    prev_thresh = inter_thresh
    all_samples += [samples.copy()]
    all_response += [response.copy()]
    all_us += [res_us.copy()]
    all_point_fail += [point_fail.copy()]
    train_part = np.append(train_part, [np.copy(handler_obj.ntrain)], axis=0)

    subcount += 1

    # Calculating subset pf and seeding next subset
    response_thresh_arr = np.zeros(nchains)
    samples = np.zeros((nspc, nchains, ndim))
    res_us = -1 * np.ones((nspc, nchains))
    response = np.zeros((nspc, nchains))
    point_fail = np.zeros((nspc, nchains))

    seed_count = nchains
    for j in range(nspc):
        for i in range(nchains):
            if inter_thresh - all_response[-1][j, i] <= 0.0:
                if seed_count > 0:
                    temps = all_samples[-1][j, i, :]
                    tempr = all_response[-1][j, i]
                    tempu = all_us[-1][j, i]
                    tempp = all_point_fail[-1][j, i]
                    samples[0, seed_count - 1, :] = temps
                    response[0, seed_count - 1] = tempr
                    res_us[0, seed_count - 1] = tempu
                    point_fail[0, seed_count - 1] = tempp
                    response_thresh_arr[seed_count - 1] = tempr
                    seed_count -= 1

    inter_thresh = np.minimum(np.percentile(response_thresh_arr, 90), fail_thresh)

    pf = pf * inter_pf
    COV += delf ** 2
    print('\nFail for thresh = ' + str(fail_count / nsamples))
    print('Intermediate Failure Threshold = ' + str(prev_thresh))
    store_thresh += [prev_thresh]

COV = COV**0.5

algo_end_time = time.time()

# Final outputs
print('\n\n\nLFDS')
print('Target pf = ' + str(target_pf/(10**6)))
print('Estimated pf = ' + str(pf))
print('COV of pf = ' + str(COV))
print('Total HF calls = ' + str(hf_count))
print('Total algorithm run time = ' + str(algo_end_time-algo_start_time))
print('Total LF model calls = ' + str(cumlf))
print('Total Retrains = ' + str(handler_obj.ntrain - init_size*np.ones_like(handler_obj.ntrain)))

hfplotter = np.zeros((subcount, nsamples))
resplotter = np.zeros((subcount, nsamples))
hfr2 = np.zeros(subcount*nsamples)
resr2 = np.zeros(subcount*nsamples)
samps = np.zeros(((subcount*nsamples), ndim))
idx = 0
for idx_sub in range(subcount):
    index = 0
    for idx_nchains in range(nchains):
        for idx_nspc in range(nspc):
            samps[idx, :] = np.copy(all_samples[idx_sub][idx_nspc, idx_nchains])
            resplotter[idx_sub, index] = all_response[idx_sub][idx_nspc, idx_nchains]
            resr2[idx] = resplotter[idx_sub, index]
            hfplotter[idx_sub, index] = hf_model(all_samples[idx_sub][idx_nspc, idx_nchains, :])
            hfr2[idx] = hfplotter[idx_sub, index]
            idx += 1
            index += 1
r2_val = r2(hfr2, resr2)

print('Estimated R^2 score = ' + str(r2_val))

print('Thresholds = ' + str(store_thresh))
print('Intermediate Failures = ' + str(store_fail))
print('Intermediate COVs = ' + str(store_delf))

print(samples.shape)
print('Final model probabilities = ' + str(handler_obj.model_probs))

print('Max absolute difference between HF model and surrogate responses = ' + str(np.max(abs(hfplotter-resplotter))))
print('Max relative difference between HF model and surrogate responses = ' +
      str(np.max(abs((hfplotter-resplotter)/hfplotter))))
print('Avg absolute difference between HF model and surrogate responses = ' + str(np.mean(abs(hfplotter-resplotter))))
print('Avg relative difference between HF model and surrogate responses = ' +
      str(np.mean(abs((hfplotter-resplotter)/hfplotter))))
print('Std dev absolute difference between HF model and surrogate responses = ' + str(np.std(hfplotter-resplotter)))
print('Std dev relative difference between HF model and surrogate responses = ' +
      str(np.std((hfplotter-resplotter)/hfplotter)))

print('Final GP Parameters = ' + str(handler_obj.kernels))

flim = 5
fstep = 0.025
fpts = int((2*flim)/fstep)
x, y = np.mgrid[-1*flim:flim:fstep, -1*flim:flim:fstep]
pos = np.dstack((x, y))
posval = np.zeros((fpts, fpts))
z_hf = griddata(samps, hfr2, (x, y))
z_surrogate = griddata(samps, resr2, (x, y))
for px in range(fpts):
    for py in range(fpts):
        posval[px, py] = hf_model(pos[px, py, :])
cmap = cm.get_cmap('tab10')

# All plots
plt.figure()
failsurface_exact = plt.contour(x, y, posval, levels=[fail_thresh], linestyles='solid', linewidths=2.0, colors=[cmap(0)])
failsurface_pred = plt.contour(x, y, z_surrogate, levels=[fail_thresh], linestyles='dashed', linewidths=2.0, colors=[cmap(1)])
plt.title('Failure Surface: Solid - Exact, Dashed - Predicted')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('Failure surface LFDS.pdf')
plt.close()

markers=["+", "x", "|", "_"]

plt.figure()
for idx_model in range(nmodels):
    for idx_sub in range(subcount):
        plt.scatter\
            (handler_obj.train_data[idx_model][int(train_part[idx_sub+1, idx_model]):int(train_part[idx_sub+2, idx_model]), 0],
             handler_obj.train_data[idx_model][int(train_part[idx_sub+1, idx_model]):int(train_part[idx_sub+2, idx_model]), 1],
             marker=markers[idx_model], color=cmap(idx_sub),
             label=('Model ' + str(idx_model) + ': Training samples from subset ' + str(idx_sub+1)))
    plt.scatter \
        (handler_obj.train_data[idx_model][train_rej[1:, idx_model], 0],
         handler_obj.train_data[idx_model][train_rej[1:, idx_model], 1],
         marker='s', color=cmap(6), label='Rejected training samples')
    plt.scatter \
        (handler_obj.train_data[idx_model][int(train_part[0, idx_model]):int(train_part[1, idx_model]), 0],
         handler_obj.train_data[idx_model][int(train_part[0, idx_model]):int(train_part[1, idx_model]), 1],
         marker=".", color=cmap(7), label='Initial training samples')
plt.title('Training Data Points')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Training Data LFDS.pdf')
plt.close()

for idx_model in range(nmodels):
    plt.figure()
    for idx_sub in range(subcount):
        plt.scatter\
            (handler_obj.train_data[idx_model][int(train_part[idx_sub+1, idx_model]):int(train_part[idx_sub+2, idx_model]), 0],
             handler_obj.train_data[idx_model][int(train_part[idx_sub+1, idx_model]):int(train_part[idx_sub+2, idx_model]), 1],
             label=('Training samples_raw from subset ' + str(idx_sub+1)))
    plt.scatter \
        (handler_obj.train_data[idx_model][int(train_part[0, idx_model]):int(train_part[1, idx_model]), 0],
         handler_obj.train_data[idx_model][int(train_part[0, idx_model]):int(train_part[1, idx_model]), 1],
         label='Initial training samples_raw')
    plt.scatter \
        (handler_obj.train_data[idx_model][train_rej[1:, idx_model], 0],
         handler_obj.train_data[idx_model][train_rej[1:, idx_model], 1],
         marker='s', color='k', label='Rejected training samples_raw')
    plt.title('Training data for model ' + str(idx_model+1))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('Model ' + str(idx_model+1) + ' train LFDS.pdf')
    plt.close()

plt.figure()
for idx_sub in range(subcount):
    plt.scatter(all_samples[idx_sub][:, :, 0], all_samples[idx_sub][:, :, 1],
                label=('Samples from subset ' + str(idx_sub+1)))
plt.title('Sample points')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.savefig('Samples LFDS.pdf')
plt.close()

plt.figure()
for idx_model in range(nmodels):
    plt.scatter(sel_sample_plot[idx_model][1:, 0], sel_sample_plot[idx_model][1:, 1],
                label=('Samples from model ' + str(idx_model+1)))
    # plt.show()
plt.scatter(sel_sample_plot[nmodels][1:, 0], sel_sample_plot[nmodels][1:, 1], label='Samples from HF')
plt.title('Sample points by selected model')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.savefig('Selected Model LFDS.pdf')
plt.close()

plt.figure()
plt.plot([-1.5*fail_thresh, 1.5*fail_thresh], [-1.5*fail_thresh, 1.5*fail_thresh], 'r')
for idx_sub in range(subcount):
    plt.scatter(hfplotter[idx_sub, :], resplotter[idx_sub, :], label=('Samples from subset ' + str(idx_sub+1)))
plt.title('Comparison of HF model response and Surrogate model response')
plt.xlabel('HF model response')
plt.ylabel('Surrogate response')
plt.legend()
plt.savefig('HF vs Surrogate LFDS.pdf')
plt.close()

plt.figure()
plt.plot(hfindex, cumhf)
plt.title('Cumulative HF Model calls as a function of sample index')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative HF Calls')
plt.savefig('Cumulative HF LFDS.pdf')
plt.close()

sampleindices = np.arange(subcount*nsamples)
plt.figure()
plt.plot(sampleindices, resplotter.flatten())
plt.title('Surrogate Response')
plt.xlabel('Sample Index')
plt.ylabel('Surrogate response')
plt.savefig('Surrogate LFDS.pdf')
plt.close()

plt.figure()
for idx_model in range(nmodels):
    plt.plot(model_prob_arr[:, idx_model], label=('Model ' + str(idx_model+1)))
plt.title('Evolution of Model Probabilities')
plt.xlabel('Sample Index')
plt.ylabel('Probability')
plt.legend()
plt.savefig('Model probabilities LFDS.pdf')
plt.close()


plt.figure()
for idx_model in range(nmodels):
    plt.plot(lf_count[:, idx_model], label=('Model ' + str(idx_model+1)))
plt.title('Cumulative LF Model calls as a function of sample index')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative LF Calls')
plt.legend()
plt.savefig('Cumulative LF calls LFDS.pdf')
plt.close()

for idx_model in range(nmodels):
    plt.figure()
    plt.plot(noise_arr[:, idx_model])
    plt.title('Evolution of noise variance for Model ' + str(idx_model))
    plt.xlabel('Training Sample Index')
    plt.ylabel('Learned Noise Variance')
    plt.savefig('Noise Variance Evolution for Model ' + str(idx_model) + ' LFDS.pdf')
    plt.close()

plt.figure()
for idx_model in range(nmodels):
    plt.plot(AIC_prob_arr[:, idx_model], label=('Model ' + str(idx_model+1)))
plt.title('Evolution of AIC Probabilities')
plt.xlabel('Training Sample Index')
plt.ylabel('AIC Probability')
plt.legend()
plt.savefig('Evolution of AIC Probabilities LFDS.pdf')
plt.close()

for idx_model in range(nmodels):
    plt.figure()
    plt.plot(AIC_arr[:, idx_model])
    plt.title('Evolution of AICc for Model ' + str(idx_model))
    plt.xlabel('Training Sample Index')
    plt.ylabel('AICc')
    plt.savefig('AICc Evolution for Model ' + str(idx_model) + ' LFDS.pdf')
    plt.close()
