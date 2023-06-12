import numpy as np

eps = 1.0e-08

def normalize_data(x_raw, y_raw, dydx_raw=None, crop=None):
  m = crop if crop is not None else x_raw.shape[0]
  x_crop = x_raw[:m]
  y_crop = y_raw[:m]
  dycrop_dxcrop = dydx_raw[:m] if dydx_raw is not None else None

  x_mean = x_crop.mean(axis=0)
  x_std = x_crop.std(axis=0) + eps
  x = (x_crop - x_mean) / x_std

  y_mean = y_crop.mean(axis=0)
  y_std = y_crop.std(axis=0) + eps
  y = (y_crop - y_mean) / y_std

  if dycrop_dxcrop is not None:
    dy_dx = dycrop_dxcrop / y_std * x_std
    #weights of deriv in cost func:
    lambda_j = 1.0 / np.sqrt((dy_dx ** 2).mean(axis=0)).reshape(1,-1)
  else:
    dy_dx = None
    lambda_j = None

  return x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j

def diff_PCA(x_raw, y_raw, z_raw, eps_PCA, eps_diff_PCA, eps_diff_fixed_dim, crop=None):
  # Crop
  m = crop if crop is not None else x_raw.shape[0]
  x_0 = x_raw[:m]
  y_0 = y_raw[:m]
  z_0 = z_raw[:m]

  # needed information
  y_std = y_0.std(axis=0)
  y_mean = y_0.mean(axis=0)
  x_mean = x_0.mean(axis=0)

  ### 1. Basic processing ###
  x_1 = (x_0 - x_mean)
  y_1 = (y_0 - y_mean) / y_std
  z_1 = z_0 / y_std

  ### 2. PCA ###
  cov_x = x_1.T @ x_1 / m
  d_2, P_2 = np.linalg.eigh(cov_x)
  d_2 = d_2[::-1] #descending order instead
  P_2 = P_2[:,::-1] #corresp to eigvals in d_2

  d_2_tilde = d_2[d_2>eps_PCA]
  D_2_tilde_sqrt = np.diag(np.sqrt(d_2_tilde))
  D_2_tilde_sqrt_inv = np.linalg.inv(D_2_tilde_sqrt)
  P_2_tilde = P_2[:, :d_2_tilde.size]
  x_2 = x_1 @ P_2_tilde @ D_2_tilde_sqrt_inv 
  z_2 = z_1 @ P_2_tilde @ D_2_tilde_sqrt

  ### 3. Differential PCA ###
  d_3, P_3 = np.linalg.eigh(z_2.T @ z_2 / m)
  d_3 = d_3[::-1] #descending order instead
  P_3 = P_3[:,::-1] #corresp to eigvals in d_2
  
  if eps_diff_fixed_dim is None:
    d_3_tilde = d_3[d_3>eps_diff_PCA]
  else: #number of dim in output fixed
    d_3_tilde = d_3[:eps_diff_fixed_dim]
  
  P_3_tilde = P_3[:, :d_3_tilde.size]
  x_3 = x_2 @ P_3_tilde
  z_3 = z_2 @ P_3_tilde
  
  lambda_j = 1.0 / np.sqrt((z_3 ** 2).mean(axis=0)).reshape(1,-1)

  return x_3, y_1, z_3, x_mean, y_mean, y_std, P_2_tilde, D_2_tilde_sqrt_inv, P_3_tilde, lambda_j
