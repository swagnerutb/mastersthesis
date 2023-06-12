import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
real_type = tf.float32

import time

##### Import local ######
from neural_approx import NeuralApproximator
from models.multi_heston_algopy import multi_Heston_algopy
#########################


def rmse(target,
         prediction):
  
  target = target.flatten()
  prediction = prediction.flatten()
  
  if len(target) != len(prediction):
    raise Exception("RMSE: Target and prediction are not the same length")
  
  return np.linalg.norm(np.subtract(target, prediction)) / np.sqrt(len(target))
  

def test(differential, #boolean
         generatorQ, #model
         generatorP, #model
         sizes,
         simul_seed=None,
         test_seed=None,
         weight_seed=None,
         eps_PCA = 1e-12,
         eps_diff_PCA = 1e-4,
         eps_diff_fixed_dim = None,
         learning_rate_shift = 1,
         full_info=False,
         tau_call_obs=[],
         K_factors=[]):

  ##### Simulation #####
  time_start = time.time()
  print("===== Simulating: training, validation, test sets =====")
  print(f"----- Q model at {time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]} -----")
  ##### Q model #####
  x_train, y_train, dydx_train = generatorQ.simulateSet(max(sizes),
                                                differential = differential,
                                                seed = simul_seed,
                                                full_info = full_info,
                                                tau_call_obs = tau_call_obs,
                                                K_factors = K_factors)
  
  

  print(f"----- P model at {time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]} -----")
  ##### P model #####
  x_test, x_spot_test, x_vol_test, _ = generatorP.simulateSet(max(sizes),
                                                differential = False,
                                                seed = test_seed,
                                                get_X1_only = True,
                                                full_info = full_info,
                                                tau_call_obs = tau_call_obs,
                                                K_factors = K_factors)

  y_test = generatorQ.valueSet(x_spot_test, x_vol_test, differential=differential)
  y_testP = generatorP.valueSet(x_spot_test, x_vol_test, differential=differential)

  print(f"----- Simulation step finished at {time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]} -----")
  time_fin = time.time()
  print(f"\nTime to generate: {np.round(time_fin - time_start, decimals=0)} sec\n")
  
  ##### Neural approximator #####
  print("===== Initializing neural approximator =====")
  x_train_mod = x_train
  x_test_mod = x_test
  dydx_train_mod = dydx_train
  
  regressor = NeuralApproximator(x_raw = x_train_mod,
                                 y_raw = y_train,
                                 dydx_raw = dydx_train_mod,
                                 eps_PCA = eps_PCA,
                                 eps_diff_PCA = eps_diff_PCA,
                                 eps_diff_fixed_dim = eps_diff_fixed_dim)
  
  pred_vals = {}

  size = sizes[0]

  regressor.prepare(m = size,
                    differential = differential,
                    weight_seed = weight_seed)

  t0 = time.time()
  
  regressor.train("std training")
  pred_vals[("standard", size)] = regressor.predict_values(x_test_mod) #predictions

  t1 = time.time()
  training_time = t1-t0
  print(f"\n===== TRAINING TIME: {np.round(training_time,4)} =====")
  
  prePCA_dim = regressor.x_raw.shape[1]
  postPCA_dim = regressor.x.shape[1]
  
  return x_test, y_test, pred_vals, y_testP, x_train, y_train, training_time, prePCA_dim, postPCA_dim


def plotData(differential,
             predictions,
             x_vals,
             x_axisname,
             y_axisname,
             targets,
             sizes,
             compute_rmse = False,
             weights = None,
             y_extra = None,
             y_extra_label = None,
             x_extra = None):
  
  n_rows = len(sizes)
  n_cols = 1

  fig, ax = plt.subplots(n_rows, n_cols, squeeze = False)
  
  if not differential:
      ax[0,0].set_title('standard')
  else:
      ax[0,0].set_title('differential')
  
  for i, size in enumerate(sizes):
      for j, regType in enumerate(['standard']):

          if compute_rmse:
              errors = 100 * (predictions[(regType, size)] - targets)
              if weights is not None:
                  errors /= weights
              rmse = np.sqrt((errors * errors).mean(axis=0))
              t = f'rmse {np.round(rmse,2)}'
          else:
              t = x_axisname

          ax[i,j].set_xlabel(t)
          ax[i,j].set_ylabel(y_axisname)
          
          if y_extra is not None and x_extra is not None:
            ax[i,j].plot(x_extra, y_extra, 'g.', label = y_extra_label)
          elif y_extra is not None:
            ax[i,j].plot(x_vals, y_extra, 'g.', label = y_extra_label)
          
          ax[i,j].plot(x_vals, targets, 'r.', label = 'Targets')
          
          ax[i,j].plot(x_vals, predictions[(regType, size)], 'c.', label = 'Predictions', alpha=0.6)
          
          ax[i,j].legend(prop={'size':8})

  plt.tight_layout()
  plt.subplots_adjust(top=0.9)
  plt.title('data')
  plt.show()
  

def plotDensity(differential,
                predictions,
                x_vals,
                x_axisname,
                y_axisname,
                targets,
                sizes,
                compute_rmse = False,
                weights = None,
                y_extra = None,
                y_extra_label = None,
                title = None):
  
  data = pd.DataFrame(data = {'x_vals':x_vals.flatten(),
                              'predictions':predictions.flatten(),
                              'targets':targets.flatten()})
  
  if y_extra is not None:
    data[y_extra_label] = y_extra.flatten()
    y_labels = ['predictions', 'targets', y_extra_label]
    colours = ['tab:blue', 'tab:orange', 'tab:green']
  else:
    y_labels = ['predictions', 'targets']
    colours = ['tab:blue', 'tab:orange']
  
  for y_lab in y_labels:
    data[data[y_lab] < 0] = 0
  
  fig, ax = plt.subplots()
  mean = {}
  quantile95 = {}
  quantile975 = {}
  for col, c in zip(y_labels, colours):
    mean[col] = data[col].mean()
    quantile95[col] = data[col].quantile(0.95)
    quantile975[col] = data[col].quantile(0.975)
    sns.kdeplot(data = data[col], label = col, ax = ax, shade = True, color=c, clip=[0,None])
    plt.axvline(mean[col], label = f'E[exposure] ({col})', color=c, linestyle='dashed')
    plt.axvline(quantile95[col], label = f'95th perc ({col})', color=c, linestyle='dotted')
    plt.axvline(quantile975[col], label = f'97.5th perc ({col})', color=c, linestyle='dashdot')
  
  deviation_mean = (mean['predictions'] - mean['targets'])/mean['targets']
  deviation_quantile95 = (quantile95['predictions'] - quantile95['targets'])/quantile95['targets']
  deviation_quantile975 = (quantile975['predictions'] - quantile975['targets'])/quantile975['targets']
  
  if title is not None:
    plt.title(title)
  ax.legend()
  ax.set_xlabel('Exposure')
  plt.tight_layout()
  plt.title('data')
  plt.show()


def plotdata_3d(x_test, #test data, shape = (m, 2)
                y_test,  #test data
                pred_vals, #predicted values
                x_train, #training data
                y_train, #training data
                size, #m
                zlim = None,
                show_training_data = False,
                y_extra = None,
                y_extra_label = None):
  
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  
  ax.scatter3D(x_test[:,0], x_test[:,1], y_test, label='validation')
  ax.scatter3D(x_test[:,0], x_test[:,1], pred_vals[("standard", size)], label='predicted')
  if show_training_data:
    ax.scatter3D(x_train[:,0], x_train[:,1], y_train, label='training')
  if y_extra is not None:
    ax.scatter3D(x_test[:,0], x_test[:,1], y_extra, label=y_extra_label)

  if zlim is not None:
    ax.set_zlim([-0.3,1])
  
  ax.set_xlabel('asset 0')
  ax.set_ylabel('asset 1')
  ax.set_zlabel('value at T1')
  
  rmse_val = rmse(target = pred_vals[("standard", size)], prediction = y_test)
  plt.title(f"rmse: {np.round(rmse_val, 5)}")

  plt.legend()
  plt.show()


#seeds = [np.random.randint(0,1000) for _ in range(5)]
seeds = [f"rand{i}" for i in range(15)]

sizes = [2**12]
print(f"Number of paths: {sizes[0]}")

differential = True
eps_PCA = 1e-12
eps_diff_PCA = 2e-4
eps_diff_fixed_dim = None

simul_seed = None
weight_seed = None

show_plots = False
save_data = True

### Market observables ###
full_info_ = [True,
              False,
              False,
              False,
              False,
              False,
              False]
#tau in years
tau_call_obs_ = [[],
                 [],
                 [1/12],
                 [3/12],
                 [12/12],
                 [3/12, 3/12, 3/12],
                 [1/12, 3/12, 12/12]]
#K in fractions
K_factors_ = [[],
              [],
              [1],
              [1],
              [1],
              [0.95, 1, 1.05],
              [1, 1, 1]]

### Portfolio ####
n_assets = 64
half_n_assets = int(np.round(0.49 + n_assets/2, decimals=0))
call_idx = [i for i in range(half_n_assets)]
call_K = [np.random.normal(1,0.05) for _ in range(len(call_idx))]
put_idx = [i for i in range(half_n_assets,n_assets)]
put_K = [np.random.normal(1,0.05) for _ in range(len(put_idx))]

print("\n===== Seeds:", seeds, "=====\n")
for seed in seeds:
  for K_factors, tau_call_obs, full_info in zip(K_factors_, tau_call_obs_, full_info_):
    corr = 0
    cov_matrix = corr * np.ones(shape=(n_assets, n_assets))
    np.fill_diagonal(cov_matrix, 1)

    T_maturity = [1 for _ in range(n_assets)]
    T_eval = [0.5, np.max(T_maturity)]

    spot0 = 1
    vol0 = 0.01
    ##### [Q-param, P-param] #####
    theta = [0.01, 0.01] #long-run variance
    kappa = [2, 2] #mean reversion rate
    xi = [0.1, 0.1] #volatility of volatility
    rho = [-0.2, -0.2] #correlation of BMs - asset and vol
    mu = [0, 0] #drift of asset
    volMult = [[1, 1], [1, 1]]
    n_t = [100, 100] #total nbr of simulation time-points


    generatorQ = multi_Heston_algopy(spot0 = [spot0 for _ in range(n_assets)],
                                    vol0 = [vol0 for _ in range(n_assets)],
                                    T_eval = T_eval, #the times at which to evaluate spot
                                    T_maturity = T_maturity, #maturities
                                    theta = [theta[0] for _ in range(n_assets)], #long-run variance
                                    kappa = [kappa[0] for _ in range(n_assets)], #mean reversion rate
                                    xi = [xi[0] for _ in range(n_assets)], #volatility of volatility
                                    rho = [rho[0] for _ in range(n_assets)], #correlation of BMs - asset and vol
                                    cov_matrix = cov_matrix, #correlation between different assets (10 combinations)
                                    mu = [mu[0] for _ in range(n_assets)], #drift of asset
                                    n_t = n_t[0], #total nbr of simulation time-points
                                    volMult = volMult[0],
                                    call_idx = call_idx, #max(call_idx) <= n_assets
                                    call_K = call_K,
                                    put_idx = put_idx, #max(put_idx) <= n_assets
                                    put_K = put_K)

    generatorP = multi_Heston_algopy(spot0 = [spot0 for _ in range(n_assets)],
                                    vol0 = [vol0 for _ in range(n_assets)],
                                    T_eval = T_eval, #the times at which to evaluate spot
                                    T_maturity = T_maturity, #maturities
                                    theta = [theta[1] for _ in range(n_assets)], #long-run variance
                                    kappa = [kappa[1] for _ in range(n_assets)], #mean reversion rate
                                    xi = [xi[1] for _ in range(n_assets)], #volatility of volatility
                                    rho = [rho[1] for _ in range(n_assets)], #correlation of BMs - asset and vol
                                    cov_matrix = cov_matrix, #correlation between different assets (10 combinations)
                                    mu = [mu[1] for _ in range(n_assets)], #drift of asset
                                    n_t = n_t[1], # total nbr of simulation time-points
                                    volMult = volMult[1],
                                    call_idx = call_idx, #max(call_idx) <= n_assets
                                    call_K = call_K,
                                    put_idx = put_idx, #max(put_idx) <= n_assets
                                    put_K = put_K)

    eps_dim_fix_tests = [None]
    learning_rate_shifts = [10000.0] #, 1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]
    rmse_values = np.zeros(shape=(2,len(eps_dim_fix_tests)))

    #for k in range(len(eps_dim_fix_tests)):
    eps_dim_fix = eps_dim_fix_tests[0]
    learning_rate_shift = learning_rate_shifts[0]

    if not isinstance(seed, str): #if seed is not "randX"
      print(f"========== SETTING SEED {seed} ==========")
      np.random.seed(seed) #same seed for each run

    x_test, y_test, pred_values, y_testP, x_train, y_train, training_time, prePCA_dim, postPCA_dim = \
      test(differential, generatorQ, generatorP, sizes, simul_seed, None,
          weight_seed, eps_PCA = eps_PCA, eps_diff_PCA = eps_diff_PCA,
          eps_diff_fixed_dim = eps_dim_fix, learning_rate_shift = learning_rate_shift,
          full_info = full_info, tau_call_obs = tau_call_obs, K_factors = K_factors)

    rmse_val_Q = rmse(target = np.maximum(pred_values[("standard", sizes[0])], pred_values[("standard", sizes[0])]*0), prediction = y_test)
    print(f"===== RMSE Q target = {np.round(rmse_val_Q, 5)} =====")

    rmse_val_P = rmse(target = np.maximum(pred_values[("standard", sizes[0])], pred_values[("standard", sizes[0])]*0), prediction = y_testP)
    print(f"===== RMSE P target = {np.round(rmse_val_P, 5)} =====")


    #Format:
    # assets_size_seed_marketobsspecs
    # size is specified as the exponent of 2
    # K is given times 100 to int
    # tau is specified in months
    if full_info:
      file_name = f'{n_assets}assets_size{int(np.log2(sizes[0]))}_seed{seed}_full_info'
    else:
      if len(K_factors) == 0:
        file_name = f'{n_assets}assets_size{int(np.log2(sizes[0]))}_seed{seed}_no_market_obs'

      if len(K_factors) == 1:
        K_factors = [int(K_factors[k]*100) for k in range(len(K_factors))]
        tau_call_obs = [int(tau_call_obs[k]*12) for k in range(len(tau_call_obs))]
        file_name = f'{n_assets}assets_size{int(np.log2(sizes[0]))}_seed{seed}_K{K_factors[0]}_tau{tau_call_obs[0]}'

      if len(K_factors) == 2:
        K_factors = [int(K_factors[k]*100) for k in range(len(K_factors))]
        tau_call_obs = [int(tau_call_obs[k]*12) for k in range(len(tau_call_obs))]
        file_name = f'{n_assets}assets_size{int(np.log2(sizes[0]))}_seed{seed}_K{K_factors[0]}-{K_factors[1]}_tau{tau_call_obs[0]}-{tau_call_obs[1]}'

      if len(K_factors) == 3:
        K_factors = [int(K_factors[k]*100) for k in range(len(K_factors))]
        tau_call_obs = [int(tau_call_obs[k]*12) for k in range(len(tau_call_obs))]
        file_name = f'{n_assets}assets_size{int(np.log2(sizes[0]))}_seed{seed}_K{K_factors[0]}-{K_factors[1]}-{K_factors[2]}_tau{tau_call_obs[0]}-{tau_call_obs[1]}-{tau_call_obs[2]}'
    
    file_path = './data/' + file_name
    if save_data:
      np.savetxt(f"{file_path}_x_test.csv", x_test, delimiter=",")
      np.savetxt(f"{file_path}_y_test.csv", y_test, delimiter=",")
      np.savetxt(f"{file_path}_pred_values.csv", pred_values[("standard", sizes[0])], delimiter=",")
      np.savetxt(f"{file_path}_y_testP.csv", y_testP, delimiter=",")
      np.savetxt(f"{file_path}_rmse_vals.csv", np.array([rmse_val_Q, rmse_val_P]), delimiter=",")
      np.savetxt(f"{file_path}_portfolio_call_strikes_idx.csv", np.array([call_K, call_idx]), delimiter=",")
      np.savetxt(f"{file_path}_portfolio_put_strikes_idx.csv", np.array([put_K, put_idx]), delimiter=",")
      np.savetxt(f"{file_path}_x_train.csv", x_train, delimiter=",")
      np.savetxt(f"{file_path}_y_train.csv", y_train, delimiter=",")
      np.savetxt(f"{file_path}_training_time.csv", np.array([training_time]), delimiter=",")
      np.savetxt(f"{file_path}_prePCA_dim.csv", np.array([prePCA_dim]), delimiter=",")
      np.savetxt(f"{file_path}_postPCA_dim.csv", np.array([postPCA_dim]), delimiter=",")
      

      print(f"\n=== {file_name} was saved ===\n")
    else:
      print(f"\n=== {file_name} WAS NOT SAVED ===\n")


  if show_plots:
    plotData(differential=True,
              predictions=pred_values,
              x_vals=x_test[:,0],
              x_axisname='X1 at time T1',
              y_axisname='MtM at time T1',
              targets=y_testP,
              sizes=sizes,
              compute_rmse=False,
              y_extra=y_train,
              y_extra_label='Training data',
              x_extra = x_train[:,0])


    plotDensity(differential = differential,
                predictions = pred_values[('standard', sizes[0])],
                x_vals = x_test[:,0].reshape((-1,1)),
                x_axisname = "X1 at time T1",
                y_axisname = "MtM at time T1",
                targets = y_test,
                sizes = sizes,
                compute_rmse = False,
                title = f'seed {seed}')


"""
plotdata_3d(x_test = x_test, #test data, shape = (m, 2)
            y_test = y_test,  #test data
            pred_vals = pred_values, #predicted values
            x_train = None, #training data
            y_train = None, #training data
            size = sizes[0], #m
            zlim = None,
            show_training_data = False,
            y_extra = None,
            y_extra_label = None)
"""
