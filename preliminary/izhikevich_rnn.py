"""In this module I simulate a SRNN with arbitrary input signals. The models used 
are izhikevich and theta function.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Izhikevich:
    def __init__(self, C:float=250, v_r:float=-60, v_t:float=-20, b:float=0,
                 v_peak:float=35, v_reset:float=-65, a:float=0.01, d:float=200,
                 I_BIAS:float=1000, k:float=2.5, tau_r:float=2, tau_d:float=20,
                 dt:float=.04, T:float=5e3, g:float=5e3, N:int=1000, p:float=1.0) -> None:
        """Initialize the system with corresponding parameters.

        Parameters
        ----------
        C : float, optional
            Capacitance of neurons in [mu F], by default 250
        v_r : float, optional
            Resting membrane potential in [mV], by default -60
        v_t : float, optional
            Threshold voltage in [mV], by default -20
        b : float, optional
            Resonance parameter in [nS], by default 0
        v_peak : float, optional
            Peak voltage where the neuron spikes in [mV], by default 35
        v_reset : float, optional
            Voltage for the spiking neuron to reset to in [mV], by default -65
        a : float, optional
            Adaptation Reciprocal time constant in [ms^-1], by default 0.01
        d : float, optional
            Adaptation jump current in [pA], by default 200
        I_BIAS : float, optional
            Bias current input to all neurons for them to enter the
            near spike regime in [pA], by default 1000
        k : float, optional
            Gain on voltage `v` in [ns/mV], by default 2.5
        tau_r : float, optional
            Synaptic rise time constant in [ms], by default 2
        tau_d : float, optional
            Synaptic decay time constant in [ms], by default 20
        dt : float, optional
            Time step of the simulation im [ms], by default 1e-4
        T : float, optional
            Duration of the simulation in [ms], by default 5e3
        g : float, optional
            Gain on the static weight matrix, by default 3
        N : int, optional
            Number of neurons in SRNN, by default 1000
        p : float, optional
            Sparsity coefficient, between 0 and 1 where 1 means fully 
            connected, by default 1.0
        """
        self.C = C
        self.v_r = v_r
        self.v_t = v_t
        self.b = b
        self.v_peak = v_peak
        self.v_reset = v_reset
        self.a = a
        self.d = d
        self.I_BIAS = I_BIAS
        self.k = k
        self.tau_r = tau_r
        self.tau_d = tau_d
        self._dt = dt
        self._T = T
        self.time = np.arange(0, T, dt)
        self._N = N
        self._G = g
        self._p = p
        self._w = self._G * np.random.randn(N, N) \
            * (np.random.rand(N, N)<p) / (p*np.sqrt(N))    # weight matrix. Constant for now.
        self.v = v_r + (v_peak - v_r) * np.random.rand(N, 1)    # Voltage of all neurons
        self.u = np.zeros(shape=(N, 1), dtype=float)    # Adaptation parameter
        self.s = np.zeros(shape=(N, 1), dtype=float)    # Synaptic filter
        self.r = np.zeros(shape=(N, 1), dtype=float)    # spike rate
        self.h = np.zeros(shape=(N, 1), dtype=float)    # Synaptic filter
        # Record the spike times of the system
        self.tspike = []
        for i in range(self._N):
            self.tspike.append([])
        
    def _v_dot(self):
        return (self.k * (self.v - self.v_r) * (self.v - self.v_t) \
            - self.u + self.I_BIAS + self.s) / self.C
        
    def _u_dot(self):
        return self.a * (self.b * (self.v - self.v_r) - self.u)
    
    def _r_dot(self):
        return -self.r / self.tau_d + self.h

    def _h_dot(self):
        return -self.h / self.tau_r
    
    def render(self):
        """Simulate the system.
        Randomly choose a neuron and return its voltage trace.
        And return its spike times.
        """
        n_neurons = 1
        random_neuron = np.random.randint(0, self._N, size=(n_neurons, ))
        voltage_trace = np.zeros(shape=(self.time.size, n_neurons), dtype=float)
        
        for i in tqdm(range(len(self.time))):
            v_new = self.v + self._dt * self._v_dot()
            self.u += self._dt * self._u_dot()
            self.v = v_new
            self.r += self._dt * self._r_dot()
            self.h += self._dt * self._h_dot()
            
            # Set the spiking rules
            spike_mask = (self.v > self.v_peak)
            self.v[spike_mask] = self.v_reset
            self.u[spike_mask] += self.d
            self.h[spike_mask] += 1 / (self.tau_d * self.tau_r)
            # self.s = self._w @ self.r
            self.s = np.random.rand(self._N, 1) * self._G * self._p

            if True in spike_mask:
                spiking_neurons = np.where(spike_mask)[0]
                for neuron in spiking_neurons:
                    self.tspike[neuron].append(self.time[i])

            # record
            voltage_trace[i] = self.v[random_neuron, 0]
        
        return voltage_trace

def main():
    """Main body to test the code
    """
    np.random.seed(2023)
    network = Izhikevich(N=2000, T=2000, I_BIAS=1000)
    voltage_trace = network.render()
    
    plt.plot(network.time, voltage_trace)
    plt.show()

if __name__ == "__main__":
    main()
                