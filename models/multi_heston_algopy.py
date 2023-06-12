import numpy as np
import scipy.integrate

import sys
sys.path.append('../kod') #needed to import from parent directory
from instruments import *

from algopy import UTPM

class multi_Heston_algopy:
    def __init__(self,
                 spot0 = [1, 1, 1, 1, 1],
                 vol0 = [0.1, 0.1, 0.1, 0.1, 0.1],
                 T_eval = [0.5,1], #the times at which to evaluate spot
                 T_maturity = [1, 1, 1, 1, 1], #maturities
                 theta = [0.1, 0.1, 0.1, 0.1, 0.1], #long-run variance
                 kappa = [2.0, 2.0, 2.0, 2.0, 2.0], #mean reversion rate
                 xi = [0.1, 0.1, 0.1, 0.1, 0.1], #volatility of volatility
                 rho = [-0.2, -0.2, -0.2, -0.2, -0.2], # correlation of BMs - asset and vol
                 cov_matrix = np.eye(5), #correlation between different assets (10 combinations)
                 mu = [0, 0, 0, 0, 0], # drift of asset
                 n_t = 100, # total nbr of simulation time-points
                 volMult = [1,1],
                 call_idx = [0,1], #max(call_idx) <= n_assets
                 call_K = [1,1],
                 put_idx = [2,3], #max(put_idx) <= n_assets
                 put_K = [1,1]
                 ):
        """
        Default: 5 assets
        
        NOTE:
            - Include all time points of interest in T_eval, including final maturity
            - max(call_idx), max(put_idx) <= n_assets
            - Indeces in call_idx and put_idx have to agree
        """
        
        self.spot0 = np.array(spot0)
        self.vol0 = np.array(vol0)
        self.T_eval = np.array(T_eval)
        self.T_maturity = np.array(T_maturity)
        self.theta = np.array(theta)
        self.kappa = np.array(kappa)
        self.xi = np.array(xi)
        self.rho = np.array(rho)
        self.cov_matrix = np.array(cov_matrix)
        self.mu = np.array(mu)
        self.n_t = np.array(n_t)
        self.n_assets = len(spot0)
        self.volMult = np.array(volMult)
        self.call_idx = call_idx
        self.call_K = np.array(call_K)
        self.put_idx = put_idx
        self.put_K = np.array(put_K)
        
        for i in range(self.n_assets):
            if 2*kappa[i]*theta[i] < xi[i]*xi[i]:
                print(f"\nNOTE: FELLER CONDITION NOT FULFILLED FOR ASSET {i}: {2*kappa[i]*theta[i]} < {xi[i]*xi[i]}\n")
        
        if len(self.call_idx) != len(self.call_K):
            raise Exception('STRIKE PRICE NEEDS TO BE SPECIFIED FOR ALL CALLS')
        
        if len(self.put_idx) != len(self.put_K):
            raise Exception('STRIKE PRICE NEEDS TO BE SPECIFIED FOR ALL PUTS')
        
        if len(self.call_idx) > 0 and max(self.call_idx) > self.n_assets:
            raise Exception('AT LEAST ONE CALL INDEX DOES NOT EXIST')
        
        if len(self.put_idx) > 0 and max(self.put_idx) > self.n_assets:
            raise Exception('AT LEAST ONE PUT INDEX DOES NOT EXIST')

    def simulatePaths(self, m, spot, vol, T, dt, xi, seed = None, for_MC=False):
        
        #Wiener processes for spot
        N_rand_assets = np.random.multivariate_normal(mean = np.zeros(shape=(self.n_assets)),
                                                      cov = self.cov_matrix,
                                                      size = (m,int(T/dt)))
        
        #Standard multivariate Gaussian
        N_rand_rand = np.random.multivariate_normal(mean = np.zeros(shape=(self.n_assets)),
                                                    cov = np.eye(self.n_assets),
                                                    size = (m,int(T/dt)))
        
        #Wiener processes for vol
        N_rand_vol = self.rho * N_rand_assets + np.sqrt((1 - np.power(self.rho,2))) * N_rand_rand
        
        N_rand = np.concatenate((N_rand_assets, N_rand_vol), axis=2)
        
        #Set initial values
        spot_ret = spot
        vol_ret = vol
        
        for t in range(int(T/dt)):
            vol_plus = UTPM.maximum(vol_ret, vol_ret*0 + 1e-12)
            spot_ret = spot_ret * np.exp(1)**((self.mu - 0.5*vol_plus) * dt + (vol_plus * dt)**0.5 * N_rand[:,t,0:self.n_assets])
            vol_ret = vol_ret + self.kappa * (self.theta - vol_plus) * dt + xi * (vol_plus * dt)**0.5 * N_rand[:,t,self.n_assets:2*self.n_assets]
        
        return spot_ret, UTPM.maximum(vol_ret, vol_ret*0)

    def simulateSet(self, m, differential = True, seed = None, get_X1_only = False, full_info=False, tau_call_obs=[], K_factors=[]):
        
        #Split data into batches to facilitate derivatives computations
        nbr_per_batch = 2**4
        nbr_splits = int(np.round(m/nbr_per_batch))
        splits_algopy = [int(np.round(m*i/nbr_splits, decimals=0)) for i in range(nbr_splits+1)]
        
        ### For market obs
        h = 1e-8
        n_market_obs = len(K_factors) #NOTE: not counting spot price
        
        if full_info:
            X = np.zeros(shape=(m, self.n_assets*2))
            Z = np.zeros(shape=(m, self.n_assets*2))
        else:
            X = np.zeros(shape=(m, self.n_assets*(n_market_obs+1)))
            Z = np.zeros(shape=(m, self.n_assets*(n_market_obs+1)))
        
        X_spot, X_vol = self.simulatePaths(m,
                                           spot = self.spot0,
                                           vol = self.vol0,
                                           T = self.T_eval[0],
                                           dt = self.T_eval[-1]/self.n_t,
                                           xi = self.xi*self.volMult[0],
                                           seed = seed)
        
        if get_X1_only:
            
            if full_info:
                for asset in range(self.n_assets):
                    X[:, 2*asset] = X_spot[:, asset]
                    X[:, 2*asset+1] = X_vol[:, asset]
                
                return X.reshape((-1, self.n_assets*2)), X_spot, X_vol, None
            
            for asset in range(self.n_assets):
                asset_spot_idx = asset * (n_market_obs + 1)
                X[:, asset_spot_idx] = X_spot[:, asset]
                
                for k in range(n_market_obs):
                    X[:, asset_spot_idx + k + 1] = self.Heston_call(spot = X_spot[:, asset],
                                                                    vol = X_vol[:, asset],
                                                                    K = K_factors[k]*X_spot[:, asset],
                                                                    tau = tau_call_obs[k],
                                                                    idx = asset)
            
            return X.reshape((-1, self.n_assets*(1+n_market_obs))), X_spot, X_vol, None
        
        Y = np.zeros(shape=m)
        n_params = self.n_assets * 2
        Z_spot = np.zeros((m, self.n_assets))
        Z_vol = np.zeros((m, self.n_assets))
        
        for i in range(nbr_splits):
            if i%10 == 0:
                print(f"split {i} out of {nbr_splits}")
            x_alg = UTPM.init_jacobian(np.concatenate((X_spot[splits_algopy[i]:splits_algopy[i+1],:],
                                                       X_vol[splits_algopy[i]:splits_algopy[i+1],:]),
                                                      axis=1))
            
            for t in range(1, len(self.T_eval)):
                x_alg[:, :self.n_assets], x_alg[:, self.n_assets:] = self.simulatePaths(m = splits_algopy[i+1] - splits_algopy[i],
                                                                            spot = x_alg[:, :self.n_assets],
                                                                            vol = x_alg[:, self.n_assets:],
                                                                            T = self.T_eval[t] - self.T_eval[t-1],
                                                                            dt = self.T_eval[-1]/self.n_t,
                                                                            xi = self.xi*self.volMult[t],
                                                                            seed = seed)
            
            Y_temp = vanilla_call_put(X = x_alg[:, 0:self.n_assets],
                                      call_K = self.call_K,
                                      put_K = self.put_K,
                                      call_idx = self.call_idx,
                                      put_idx = self.put_idx)
            
            Y[splits_algopy[i]:splits_algopy[i+1]] = Y_temp.data[0,0]
            
            if differential:
                Z_temp = UTPM.extract_jacobian(Y_temp)
                
                Z_out = np.zeros((splits_algopy[i+1] - splits_algopy[i], n_params))
                
                #select right samples and reshape Z array
                for sample in range(splits_algopy[i+1]-splits_algopy[i]):
                    Z_out[sample, :] = Z_temp[sample, sample*n_params : (sample+1)*n_params]
                
                Z_spot[splits_algopy[i]:splits_algopy[i+1],:] = Z_out[:, :self.n_assets]
                Z_vol[splits_algopy[i]:splits_algopy[i+1],:] = Z_out[:, self.n_assets:]

        
        ### WITH FULL INFO ###
        if full_info:
            for asset in range(self.n_assets):
                X[:, 2*asset] = X_spot[:, asset]
                X[:, 2*asset+1] = X_vol[:, asset]
                Z[:, 2*asset] = Z_spot[:, asset]
                Z[:, 2*asset+1] = Z_vol[:, asset]
            
            return X.reshape((-1, self.n_assets*2)), Y.reshape((-1,1)), Z.reshape((-1,self.n_assets*2))
        
        ### WITH MARKET OBS - At the money call price ###
        """
            When calculating pathwise derivatives wrt. call price C, we use that dY/dC = dY/dv * 1/(dC/dv)
        """
        
        for asset in range(self.n_assets):
            calls = np.zeros(shape=(m,n_market_obs))

            if differential:
                calls_h_spot_upper = np.zeros(shape=(m,n_market_obs))
                calls_h_spot_lower = np.zeros(shape=(m,n_market_obs))
                calls_h_vol_upper = np.zeros(shape=(m,n_market_obs))
                calls_h_vol_lower = np.zeros(shape=(m,n_market_obs))
        
            for call_idx in range(n_market_obs):
                calls[:, call_idx] = self.Heston_call(spot = X_spot[:, asset],
                                                        vol = X_vol[:, asset],
                                                        K = K_factors[call_idx]*X_spot[:, asset],
                                                        tau = tau_call_obs[call_idx],
                                                        idx = asset)
                
                if differential:
                    calls_h_spot_upper[:, call_idx] = self.Heston_call(spot = X_spot[:, asset] + h,
                                                        vol = X_vol[:, asset],
                                                        K = K_factors[call_idx] * (X_spot[:, asset] + h),
                                                        tau = tau_call_obs[call_idx],
                                                        idx = asset)
                    calls_h_spot_lower[:, call_idx] = self.Heston_call(spot = X_spot[:, asset] - h,
                                                        vol = X_vol[:, asset],
                                                        K = K_factors[call_idx] * (X_spot[:, asset] - h),
                                                        tau = tau_call_obs[call_idx],
                                                        idx = asset)
                    
                    calls_h_vol_upper[:, call_idx] = self.Heston_call(spot = X_spot[:, asset],
                                                        vol = X_vol[:, asset] + h,
                                                        K = K_factors[call_idx] * X_spot[:, asset],
                                                        tau = tau_call_obs[call_idx],
                                                        idx = asset)
                    calls_h_vol_lower[:, call_idx] = self.Heston_call(spot = X_spot[:, asset],
                                                        vol = X_vol[:, asset] - h,
                                                        K = K_factors[call_idx] * X_spot[:, asset],
                                                        tau = tau_call_obs[call_idx],
                                                        idx = asset)
            
            #re-order X to get same as Z
            X[:, asset*(n_market_obs+1) : (asset+1)*(n_market_obs+1)] = np.column_stack((X_spot[:,asset], calls))
            
            if differential:
                jacobian_asset = np.zeros((m, 2, n_market_obs+1))
                jacobian_asset[:, 0, 0] = np.ones(shape=m)
                
                for i in range(n_market_obs):
                    jacobian_asset[:,0,i+1] = (calls_h_spot_upper[:, i] - calls_h_spot_lower[:, i]) * 0.5 / h
                    jacobian_asset[:,1,i+1] = (calls_h_vol_upper[:, i] - calls_h_vol_lower[:, i]) * 0.5 / h
                
                for path in range(m):
                    Z[path,  asset*(n_market_obs+1) : (asset+1)*(n_market_obs+1)], _, _, _ = \
                        np.linalg.lstsq(jacobian_asset[path, :, :], \
                                        np.array((Z_spot[path, asset],Z_vol[path, asset])))
        
        
        return  X.reshape((-1, self.n_assets*(n_market_obs+1))), Y.reshape((-1,1)), Z.reshape((-1, self.n_assets*(n_market_obs+1)))
    
    def phi(self, u, tau, vol0, idx):
        alpha_hat = -0.5 * u * (u + 1j)
        beta = self.kappa[idx] - 1j * u * self.xi[idx] * self.rho[idx]
        gamma = 0.5 * self.xi[idx] * self.xi[idx]
        d = np.sqrt(beta*beta - 4 * alpha_hat * gamma)
        g = (beta - d) / (beta + d)
        h = np.exp(-d*tau)
        A_ = (beta-d)*tau - 2*np.log((g*h-1) / (g-1))
        A = self.kappa[idx] * self.theta[idx] / (self.xi[idx] * self.xi[idx]) * A_
        B = (beta - d) / (self.xi[idx] * self.xi[idx]) * (1 - h) / (1 - g*h)
        return np.exp(A + B * vol0)

    def integral(self, k, tau, vol0, idx):
        integrand = (lambda u: 
            np.real(np.exp((1j*u + 0.5)*k)*self.phi(u - 0.5j, tau, vol0, idx))/(u*u + 0.25))

        i, err = scipy.integrate.quad_vec(integrand, 0, np.inf)
        return i

    def Heston_call(self, spot, vol, K, tau, idx):
        """vol is automatically vol = max(vol,0)"""
        
        vol[np.where(vol<=0)] = 0 #Equation does not work with negative vol

        a = np.log(spot/K) + self.mu[idx]*tau
        i = self.integral(a, tau, vol, idx=idx)
        
        return spot * np.exp(self.mu[idx]*tau) - K /np.pi*i
    
    def valueSet(self, X_spot, X_vol, differential=False):
        """Value vanilla call \n
        Args:
            X: spot and vol at time point at which we want value
            X.shape: (m, 2*n_assets)
            differential: NOT used atm \n
        Returns:
            call price
        """
        value = np.zeros(shape=(len(X_spot[:,0])))
        
        for k in range(len(self.call_idx)):
            idx = self.call_idx[k]
            spot = X_spot[:,idx]
            vol = X_vol[:,idx]
            vol[np.where(vol<=0)] = 0
            K = self.call_K[k]

            # NOTE: ASSUMES EXPOSURE MODELLIING FROM T_eval[0] TO MATURITIES
            tau = self.T_maturity[idx] - self.T_eval[0]
            
            value += self.Heston_call(spot=spot, vol=vol, K=K, tau=tau, idx=idx)
        
        for k in range(len(self.put_idx)):
            idx = self.put_idx[k]
            spot = X_spot[:,idx]
            vol = X_vol[:, idx]
            vol[np.where(vol<=0)] = 0
            K = self.put_K[k]
            
            # NOTE: ASSUMES EXPOSURE MODELLIING AT T_eval[0]
            tau = self.T_maturity[idx] - self.T_eval[0]
            
            #put-call parity: put = call - spot + K * B(t,T)
            #Assumes constant r = 0, i.e. B(t,T) = 1
            value += self.Heston_call(spot=spot, vol=vol, K=K, tau=tau, idx=idx) - spot + K
        
        return value

    def uniformSet(self, lower=0.65, upper=2, num=100):
        """Value vanilla call \n
        Args:
            spot: spot prices \n
        Returns:
            call price
        """
        spot = np.linspace(start=lower, stop=upper, num=num)
        tau = self.T2 - self.T1
        a = np.log(spot/self.K) + self.mu*tau
        i = self.integral(a, tau)
        return spot * np.exp(self.mu*tau) - self.K /np.pi*i

