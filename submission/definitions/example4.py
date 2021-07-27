# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:34:09 2020

@author: CanBo
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy import signal
from scipy.interpolate import interp1d
from pyFRF.pyFRF import FRF
# pyrdo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["PATH"] += pyrdo_dir + ";"
# curdir = os.path.abspath(".")
# os.chdir(pyrdo_dir)


#Friction-induced Vibration in Lead Screw Systems
import threading
import _thread
from contextlib import contextmanager

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()





# def make_ode_dof1(I_ls, r_m, lamba, mass, k_nom, C, R, F_0, T_0):
class LeadScrew(): 

    def __init__(self, x_ref):
        # x[0] : u = z - z_0 where z = theta - theta_i
        # x[1] : du_dt = dz_dt = dtheta_dt - omega_i
        d_m = x_ref[0]
        self.r_m = d_m / 2
        self.omega_i = x_ref[1]
        self.omega_0 = x_ref[2]
        self.lamba = x_ref[3]
        self.mass = x_ref[4]
        self.I_ls = 10**x_ref[5] # kg m # dep para 1
        self.k_ls = x_ref[6]
        self.c_ls = x_ref[7] # Para 7
        self.R = x_ref[8] # Para 6
        self.k_c = 10**x_ref[9]
        self.c_c = x_ref[10]
        self.F_0 = 0 # x_ref[8]
        self.T_0 = 0 # x_ref[9]
        # self.mu_1 = x_ref[8] # para 8
        # self.mu_2 = x_ref[9] # para 9
        # self.mu_3 = x_ref[10] # para 10
        self.mu_1 = 2.18e-1
        self.mu_2 = 2.03e-2
        self.mu_3 = -4.47e-4
        

        v0 = 0.013704179917841673 # Stribeck constant
        self.r_0 = self.r_m / np.cos(self.lamba) / v0 # depends on Stribeck constant
        self.r_1 = 2
        self.set_xi0gamma0()
        
        fact = (self.r_m * np.sin(self.lamba))**2
        self.k_c_hat = fact * self.k_c
        self.c_c_hat = fact * self.c_c
        fact = (self.r_m * np.tan(self.lamba))
        self.mass_hat = self.mass * fact**2
        self.R_hat = self.R * fact / (1 + self.mu0 * np.tan(self.lamba))
        # self.x0 = [0, self.theta_i, 0,
        #            self.r_m * np.tan(self.lamba) * self.theta_i]
        
        # self.y0 = [0, -self.omega_i,
        #            0, -self.r_m * np.tan(self.lamba) * self.omega_i]
        # self.y0 = [self.omega_i, 0, self.r_m * np.tan(self.lamba) * self.omega_i, 0]
        
        # sign_theta = np.sign(self.omega_i)
        # dtheta2_d2t = (self.R - self.F_0 * sign_theta) * self.xi0 - self.T_0 * sign_theta
        self.u01 = (-1 * self.c_ls * self.omega_i - self.xi0 * self.R) / self.k_ls
        self.u02 = self.R / (self.k_c * np.cos(self.lamba)**2 * (1 + self.mu0 * np.tan(self.lamba)))
        self.u02 += self.r_m * np.tan(self.lamba) * self.u01
        self.y0 = [self.omega_0, 0, 0, 0] # sets initial slide velocity to omega_0
        #i.e. omega = omega_i + omega_0
        # self.r_m * np.tan(self.lamba) * dtheta_init
        # self.x0 = [0, 0, 0, 0]
        # Stability check
        # self.check_stability()
        self.last_time = 0
        self.inform_nonconvergence = True
        
        self.K_mat, self.C_mat = np.zeros((2, 2)), np.zeros((2, 2))
        mu0cot = self.mu0 / np.tan(self.lamba)
        mu0tan = self.mu0 * np.tan(self.lamba) 
        self.K_mat[0, 0] = self.k_ls + self.k_c_hat * (1 - mu0cot)
        self.K_mat[0, 1] = self.k_c_hat * (mu0cot - 1)
        self.K_mat[1, 0] = self.k_c_hat * (-mu0tan - 1)
        self.K_mat[1, 1] = self.k_c_hat * (mu0tan + 1)
        self.C_mat[0, 0] = self.c_ls + self.c_c_hat * (1 - mu0cot)
        self.C_mat[0, 1] = self.c_c_hat * (mu0cot - 1)
        self.C_mat[1, 0] = self.c_c_hat * (-mu0tan - 1)
        self.C_mat[1, 1] = self.c_c_hat * (mu0tan + 1)


        
    def is_stable(self):
        """ Implements eqns 6.44-48 from orang Valid phd"""
        D1 = self.mass_hat * (1 - self.mu0 * np.sign(self.R * self.omega_i) * np.tan(self.lamba)**-1)
        mutanlam = self.mu0 * np.sign(self.R * self.omega_i) * np.tan(self.lamba)
        D1 += self.I_ls * ( 1 + mutanlam)
        D2 = self.c_c_hat * D1 + self.c_ls*self.mass_hat
        if D2 <= 0:
            return False
        
        
        
    
    def set_xi0gamma0(self):
        self.mu0 = self.get_mu([None, 0]) * np.sign(self.R * self.omega_i)
        tan_lamb = np.tan(self.lamba)
        self.xi0 = self.r_m * (self.mu0 - tan_lamb) / (1 + self.mu0 * tan_lamb)
        self.gamma0 = self.I_ls -  self.r_m * tan_lamb * self.xi0 * self.mass
    
    def get_dmu(self, dtheta=0):
        abs_theta = np.abs(dtheta + self.omega_i)
        res = self.r_0 * self.mu_2 * np.exp(-abs_theta * self.r_0)
        res += self.mu_3 
        return res #* (1 - np.exp(-abs_theta * self.r_1))
    
    def get_mu(self, x):
        abs_theta = np.abs(x[1] + self.omega_i)
        # omega_max = 100
        res = self.mu_1
        res += self.mu_2 * np.exp(-abs_theta * self.r_0)
        res += self.mu_3 * abs_theta #omega_max * np.tanh(abs_theta/omega_max)
        return res #* (1 - np.exp(-abs_theta * self.r_1))
    
    def get_xi_gamma(self, mu_s, tan_lamb):
        xi = self.r_m * (mu_s - tan_lamb) / (1 + mu_s * tan_lamb)
        gamma = self.I_ls -  self.r_m * tan_lamb * xi * self.mass
        return xi, gamma
    
    def y_to_dtheta(self, y):
        return y[0] + self.omega_i
        
    def y_to_dx(self, y):
        return (y[2] + self.omega_i) * self.r_m * np.tan(self.lamba)


    
    def get_mus_N(self, x, t):
        # since we need signum N to get mu_s, this has to be solved
        # by trying combinations and return the matching one
        cos_lamb, sin_lamb = np.cos(self.lamba), np.sin(self.lamba)
        tan_lamb = np.tan(self.lamba)
        mu = self.get_mu(x)
        y = [0] * 4
        y[0] = x[0] + self.omega_i * t + self.u01
        y[1] = x[1] + self.omega_i
        y[2] = (x[2] + self.omega_i * t) * self.r_m * tan_lamb + self.u02
        y[3] = (x[2] + self.omega_i) * self.r_m * tan_lamb

        delta = y[2] * cos_lamb - self.r_m * y[0] * sin_lamb
        ddelta_dt = y[3] * cos_lamb - self.r_m*y[1] * sin_lamb
        N = self.k_c * delta + self.c_c * ddelta_dt
        theta_sign = np.sign(y[1])
        mu_s = mu * np.sign(N) * theta_sign
        xi, gamma = self.get_xi_gamma(mu_s, tan_lamb)
        return mu_s, xi, gamma, N
    
    # def _get_N(self, x, mu_s, gamma, theta_sign, tan_lamb, cos_lamb, sin_lamb):
    #     intforce = self.k_ls * x[0] + self.c_ls * x[1] + self.T_0 * theta_sign
    #     intforce *= self.mass * self.r_m * tan_lamb
    #     numer = self.gamma0 * (self.R - self.F_0 * theta_sign) + intforce
    #     numer = theta_sign
    #     denom = gamma * (cos_lamb + mu_s * sin_lamb)
    #     return numer / denom
    
    def _ode_step_dt2(self, t, x):
        # x => u = z - z_0 where z = theta - theta_i
        if t - self.last_time >= self.print_dura:
            self.last_time = t
            print(f"{t} seconds")
        mu_s, xi, _, N = self.get_mus_N(x, t)
        tan_lamb = np.tan(self.lamba)
        cot_lamb = 1 / tan_lamb 
        # if locked:
        #     return [-self.omega_i, 0]
        # get forces first
        shared = self.k_c_hat * (x[2] - x[0]) + self.c_c_hat * (x[3] - x[1]) + self.R_hat
        shared *= (self.mu0 - mu_s)
        f_theta, f_x = shared * cot_lamb, shared * tan_lamb
        d2theta_dt2 = f_theta - self.K_mat[0, 0] * x[0] - self.K_mat[0, 1] * x[2]
        d2theta_dt2 -= (self.C_mat[0, 0] * x[1] + self.C_mat[0, 1] * x[3])
        d2theta_dt2 /= self.I_ls
        
        d2x_dt2 = f_x - self.K_mat[1, 0] * x[0] - self.K_mat[1, 1] * x[2]
        d2x_dt2 -= (self.C_mat[1, 0] * x[1] + self.C_mat[1, 1] * x[3])
        d2x_dt2 /= self.mass_hat
        return [x[1], d2theta_dt2 , x[3], d2x_dt2]
    
    
    
    def ode_step_dt2(self, t, x):
        if abs(x[1]) > 1e9 or abs(x[3]) > 1e9:
            raise RuntimeError("Divergence? d2x_dt=", x[1], x[3])
        res = self._ode_step_dt2(t, x)
        return np.c_[res[0], res[1], res[2], res[3]].T
    
    def solve(self, method="RK45", duration=1.):
        self.print_dura = duration / 10
        t_eval = np.linspace(0, duration, num=2048)
        return solve_ivp(self.ode_step_dt2, (0, duration), self.y0, method=method,
                         vectorized=True, t_eval=t_eval)

def resample_amps(amps, init_time, target_time):
    """
    

    Parameters
    ----------
    amps : np.ndarray
        Amplitudes with shape (n_outputs, timesteps).
    time : array_like
        current time axis
    target_time : array_like
        target time axis
        

    Returns
    -------
    res : np.ndarray
        Amplitudes at target_time axis with shape(n_outputs, timesteps)
    """
    # amps = np.atleast_2d(amps)
    if amps.ndim != 2:
        amps = amps.reshape((-1, init_time.size))
    res = np.zeros((amps.shape[0], target_time.size))
    for i_amp in range(amps.shape[0]):
        # for d in [amps[i_amp, :], init_time.ravel(), target_time.ravel()]:
        #     print(d.shape, d.min(), d.max())
        mod = interp1d(init_time.ravel(), amps[i_amp, :], fill_value="extrapolate",
                       assume_sorted=True)
        res[i_amp, :] = mod(target_time.ravel())
    return res

def get_time_ind(time, seconds):
    return np.argmin((time - seconds)**2)

def get_FRF(time, dx_dt, ampl):
    fs = int(np.round((1 / np.diff(time).mean())/2))

    new_time = np.linspace(0, 1, fs)
    dx_dt = resample_amps(dx_dt, time, new_time).ravel()
    # ind8 = get_time_ind(new_time, 0.7)
    # end_mean = np.mean(dx_dt[ind8:])
    # ind9 = get_time_ind(new_time, 0.9)
    # print(ind8, ind9, dx_dt.shape, fs)
    # end_ind = 1 + ind9 + np.argmin(np.abs(dx_dt[ind9:] - end_mean))
    # print(end_ind )
    # dx_dt = dx_dt[:end_ind]
    # new_time = new_time[:end_ind]
    
    
    n = int(2 ** np.ceil(np.log(dx_dt.size)/np.log(2))) # get next power of two
    excitation = np.zeros(dx_dt.size)
    excitation[0] += ampl
    frf = FRF(sampling_freq=fs, exc=excitation, resp=dx_dt, 
              exc_window='None', exc_type="v", resp_type='v', 
                resp_window="Hann", 
               fft_len=n, nperseg=n//4, #noverlap=n//8
              )
    f = frf.get_f_axis()
    ind_max = np.where(f > fs / 2)[0]
    if ind_max.size < 1:
        ind_max = -1
    else:
        ind_max = ind_max[0]
    f = f[1:ind_max]
    pwelch_spec = np.abs(frf.get_FRF()).ravel()[1:ind_max]
    pwelch_spec[np.logical_not(np.isfinite(pwelch_spec))] = pwelch_spec.min() / 2
    pwelch_spec = 10 * np.log10(pwelch_spec / np.abs(ampl))
    return f, pwelch_spec

def get_first_mode(inp, result, freq_min=1, freq_max=None):
    dx_dt = result.y[2].copy()
    time = result.t
    f, pwelch_spec = get_FRF(time, dx_dt, inp[2])
    minind = get_time_ind(f, freq_min)
    maxind = -1
    if freq_max:
        maxind = get_time_ind(f, freq_max) + 1
    maxind = find_first_peak(pwelch_spec.ravel()[minind:maxind], f)
    if maxind == -1:
        return 2048
    return min(f[minind + maxind], 2048)

def find_first_peak(sig, f):
    # try:
    #     sigc = signal.detrend(sig.ravel())
    # except ValueError:
    sigc = sig.copy()
    wlen = max(3, get_time_ind(f, 10))
    peaks = signal.find_peaks(sigc, wlen=wlen)[0] # No peaks so assume only HF
    if peaks.size == 0:
        return -1
    mid = (sigc.min() + sigc.max()) / 2
    peaks = peaks[sigc[peaks] > mid]
    if peaks.size == 0:
        return -1
    proms = signal.peak_prominences(sigc, peaks,# wlen=wlen,
                                    )[0]#, )
    ind = np.argmax(proms)
    return peaks[ind]

def get_con(res):
    max_dx = 0.1 # not open-closed
    return max_dx - min(3, np.abs(np.cumsum(res.y[2][:-1] * np.diff(res.t))).max())

if __name__ == "__main__":
    import sys
    import pickle
    help_text = """
        d_m = args[0] \n
        omega_i = args[1] \n
        omega_0 = args[2] \n
        lamba = args[3] \n
        mass = args[4] \n
        I_ls = 10**args[5] \n 
        k_ls = args[6] \n
        c_ls = args[7] \n
        R = args[8] \n
        k_c = 10**args[9] \n
        c_c = args[10] \n
        save_path = args[11]
        """
    if len(sys.argv) < 3:
        raise ValueError("Model requires 3 arguments to run:\n" + help_text)
    if "--help" in sys.argv or "-h" in sys.argv:
        print(help_text)
    else:
        print(sys.argv)
        print("Results will be written to", sys.argv[2])
        # x_ref = [float(a) for a in sys.argv[1:12]]
        # with open(sys.argv[1], "rb") as f:
        #     x = pickle.load(f)
        x = np.load(sys.argv[1])
        results = []
        for x_ref in x:
            ls = LeadScrew(x_ref)
            try:
                with time_limit(1800):
                    res = ls.solve(duration=1)
            except (RuntimeError, TimeoutException) as e:
                print("Timed out with RK!", e) # Probably instability so bad result
                # try:
                #     with time_limit(900):
                #         res = ls.solve(method="Radau", duration=1)
                # except (RuntimeError, TimeoutException) as e:
                #         print("Timed out with Radau!", e) # Probably instability so bad result
                results.append([0, -3])
                continue
            final = [-1 * get_first_mode(x_ref, res), get_con(res)]
            results.append(final)
        
        p = sys.argv[2]
        if not p.endswith(".pkl"):
            p += ".pkl"
        with open(sys.argv[2], "wb") as f:
            pickle.dump(results, f)
        