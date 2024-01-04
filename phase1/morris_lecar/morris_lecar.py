"""Implementation of Reservoir machine learning with FORCE training using the Morris Lecar model.
We will be abiding by Dale's law stating that each neuron can either have excitatory of inhibitory outgoing synapses.
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class MorrisLecar:
    def __init__(self, supervisor:np.ndarray, dt:float, T:float, 
                 N:int=1000, BIAS:float=100, C:float=20, g_L:float=2, g_K:float=8, 
                 g_Ca:float=4, E_L:float=-60, E_K:float=-84, E_Ca:float=120, v1:float=-1.2,
                 v2:float=18, v3:float=12, v4:float=17.4, phi:float=0.067, a_r:float=1.1, 
                 a_d:float=0.19, v_t:float=2, k_p:float=5, t_max:float=1.0,
                 E_AMPA:float=-70, E_GABA:float=-70, Q:float=5e3, l:float=2.0) -> None:
        # Morris Lecar model parameters
        self._N = N
        self._BIAS = BIAS
        self._C = C
        self._g_L = g_L
        self._g_K = g_K
        self._g_Ca = g_Ca
        self._E_L = E_L
        self._E_K = E_K
        self._E_Ca = E_Ca
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
        self._v4 = v4
        self._phi = phi
        self._v_t = v_t
        self._a_r = a_r
        self._a_d = a_d
        self._k_p = k_p
        self._t_max = t_max
        self._E_AMPA = E_AMPA
        self._E_GABA = E_GABA
        
        # Network connections
        self.v = np.zeros(shape=(N, 1), dtype=float)
        self.w = np.random.normal(loc=0, scale=1/np.sqrt(N))
        self.s = np.zeros(shape=(N, 1), dtype=float)
        self.n = np.zeros(shape=(N, 1), dtype=float)
        self.E = np.random.choice(a=[self._E_AMPA, self._E_GABA], size=(N, ))
        # Let's follow Dale's law
        self.E = np.tile(self.E, reps=(N, 1))
        
        # Time series
        self._dt = dt
        self._duration = T
        self.time = np.arange(0, T, dt)
        self._nt = self.time.size
        
        # Encoding and decoding
        self.l = l
        self.sup = supervisor
        dim = min(self.sup.shape)
        self.dec = np.zeros(shape=(N, dim), dtype=float)
        self.eta = Q * (2 * np.random.rand(N, dim) - 1)
        self.Pinv = np.eye(self._N) * self.l
        self.x_hat = self.dec.T @ self.s
        self.x_hat_rec = np.zeros((self.time.size, self.x_hat.size), dtype=float)
        self.ipsc = np.zeros(shape=(N, 1), dtype=float)
        
    def m_ss(self) -> np.ndarray:
        return 0.5 * (1 + np.tanh((self.v - self._v1) / self._v2))    
    
    def n_ss(self) -> np.ndarray:
        return 0.5 * (1 + np.tanh((self.v - self._v3) / self._v4))
    
    def tau_n(self) -> np.ndarray:
        return 1 / (self._phi * np.cosh((self.v - self._v3) / (2 * self._v4)))
    
    def T(self):
        return self._t_max / (1 + np.exp(-(self.v - self._v_t) / self._k_p))
        
    def s_dot(self):
        return self._a_r * self.T() * (1 - self.s) - self._a_d * self.s
    
    def n_dot(self):
        return (self.n_ss() - self.n) / self.tau_n()
    
    def calc_ipsc(self) -> None:
        self.ipsc = (self.w * (self.v - self.E)) @ self.s
        
    def v_dot(self):
        self.calc_ipsc()    # Calculate the new post-synaptic potential
        
        return (self._BIAS - self._g_L * (self.v - self._E_L) \
            - self._g_K * self.n * (self.v - self._E_K) \
            - self._g_Ca * self.m_ss() * (self.v - self._E_Ca) \
            + self.ipsc) / self._C
    
    def render(self, n_neurons:int=10):
        random_neuron = np.random.choice(a=self._N, size=n_neurons, replace=False)
        voltage_trace = np.zeros(shape=(self.time.size, n_neurons), dtype=float)
        for i in tqdm(range(self._nt)):
            dv = self._dt * self.v_dot()
            self.n += self._dt * self.n_dot()
            self.s += self._dt * self.s_dot()
            self.v += dv

            voltage_trace[i] = self.v[random_neuron, 0]
            
            self.x_hat = self.dec.T @ self.s
            self.x_hat_rec[i] = self.x_hat.flatten()
        
        return random_neuron, voltage_trace