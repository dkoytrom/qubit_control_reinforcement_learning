from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.policies import random_py_policy
from tf_agents.policies import random_tf_policy
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver

import matplotlib.pyplot as plt
import qutip
from qutip import Bloch, QobjEvo, basis, sesolve, sigmay, sigmaz, sigmax, expect, fidelity, sigmap, sigmam, ket2dm
import random

"""
NAME: QubitContEnvContFunc
DESC: 
"""
class QubitContEnvContFunc(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, max_Δ, restriction_fg = True, fidelity_threshold = 0.95, \
                 verbose_fg = False, initial_state = 0, final_state = 1, seed = 1989, reward_gradually = False,
                 nb_harmonics = 3):
        
        if seed is not None:
            tf.random.set_seed(seed)

        # 2 * nb_harmonics + 1 => params for Ω / Δ
        # so total params number is the double
        self._nb_harmonics = nb_harmonics
        self._nb_func_params = 2 * (2 * nb_harmonics + 1)
        self._params = np.zeros(self._nb_func_params)
        self._omega_params = np.zeros(int(self._nb_func_params / 2))
        self._detuning_params = np.zeros(int(self._nb_func_params / 2))

        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (self._nb_func_params, ), dtype = np.float32, 
                                                        minimum = -100 * np.ones(self._nb_func_params), maximum = 100 * np.ones(self._nb_func_params), 
                                                        name = 'action')

        # define the state space
        self._observation_spec = array_spec.BoundedArraySpec(shape = (3, ), dtype = np.float32, 
            minimum = [0., 0., 0.], 
            maximum = [1., 1., 1.], 
            name = 'state'
        )

        self._quantum_state = basis(2, 0).unit()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

        # initial state (time step, magnetic field)
        rho = ket2dm(self._quantum_state)

        self._density_matrix = rho

        if (self._verbose_fg):
            print("INITIAL FIDELITY", )

        self._time_step = 0
        self._Δ = 0
        self._Ω = 0
        self._omegas = []
        self._T = T
        self._N = N
        self._max_steps = N
        self._max_Ω = max_Ω
        self._max_Δ = max_Δ
        self._ξ = 1

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False

        self._initial_state = initial_state
        self._final_state = final_state
        self._reward_gradually = reward_gradually

        self.update_state()
        self.optimal_omega_params = None
        self.optimal_detuning_params = None
        self.max_fidelity = 0.0
        self._finished = False

    """
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
    def update_state(self):
        rho = self._density_matrix

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        self._state = [np.real(rho11), np.real(rho12), np.imag(rho12)]

    """
    NAME: observation_spec
    DESC: 
    """
    def observation_spec(self):
        return self._observation_spec
    
    def get_time_steps(self):
        return self._N
    
    """
    NAME: reset
    DESC: 
    """
    def _reset(self):
        self._quantum_state = basis(2, 0).unit()
        self._time_step = 0
        self._Ω = 0
        self._episode_ended = False
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

        rho = ket2dm(self._quantum_state)
        self._density_matrix = rho

        self.update_state()

        if self._verbose_fg:
            print("INITIAL STATE : time step: ",  self._time_step, " s = ", (self._time_step, self._Ω))
        
        return ts.restart(np.array(self._state, dtype = np.float32))
    
    def _step(self, action):
        # if episode ended, reset the environment
        if self._episode_ended:
            return self.reset()
        
        # we also need some restrictions:
        # in this model we will use only bang-bang pulses
        # control Ω can be +1 / -1
        # actions can also be 0, +2 / -2
        # So when we are at Ω = +1 we can only apply action -1

        # Here it is time to define the model at which we can choose the actions
        # State space: S = T x Ω = {(t, Ω(t)) | t in R and Ωt in Z}, where state s in S: s = (t, Ω(t))
        # Actions space: A = {-1, 1} (bang-bang pulses) inspired by optimal control

        # increment time step
        self._time_step += 1

        # extract parameters from action array
        nb_harmonics = self._nb_harmonics

        action_diff = action
        self._params += action_diff
        self._params /= np.linalg.norm(self._params)

        # construct trigonometric series for Ω
        omega_params = self._params[0:2 * nb_harmonics + 1]
        detuning_params = self._params[2 * nb_harmonics + 1: len(self._params)]

        self._omega_params = omega_params
        self._detuning_params = detuning_params

        def omega(t, args):
            Ω = omega_params[0]

            for harmonic in range(1, nb_harmonics + 1):
                Ω += omega_params[2 * harmonic - 1] * np.cos(harmonic * t) + omega_params[2 * harmonic] * np.sin(harmonic * t)

            return Ω

        # construct trigonometric series for Δ
        def detuning(t, args):
            Δ = detuning_params[0]

            for harmonic in range(1, nb_harmonics + 1):
                Δ += detuning_params[2 * harmonic - 1] * np.cos(harmonic * t) + detuning_params[2 * harmonic] * np.sin(harmonic * t)

            return Δ

        # set Hamiltonian for the two level system
        H = QobjEvo([[0.5 * sigmaz(), detuning], [0.5 * sigmax(), omega]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T, num = N)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[-1]

        rho = ket2dm(self._quantum_state)

        self._density_matrix = rho

        self.update_state()

        # compute fidelity at this time step        
        transfer_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = transfer_fidelity
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = transfer_fidelity ** 2

        self._episode_ended = True

        # we have to calculate the inner product between final state and expected final state
        reward = transfer_fidelity * 10

        if (self._verbose_fg):
            print(" fidelity ", " = ", round(squared_fidelity, 5), self._omega_params, self._detuning_params)

        """ if squared_fidelity > self.max_fidelity:
            self.max_fidelity = squared_fidelity
            self.optimal_omega_params = self._omega_params
            self.optimal_detuning_params = self._detuning_params
            self.states = result.states """
        
        if squared_fidelity > self.max_fidelity:
            self.max_fidelity = squared_fidelity
        
        if squared_fidelity > self._fidelity_threshold and self._finished is False:
            self.max_fidelity = squared_fidelity
            self.optimal_params = self._params
            self.optimal_omega_params = self._omega_params
            self.optimal_detuning_params = self._detuning_params
            self.states = result.states
            self._finished = True
            print(" fidelity ", " = ", round(squared_fidelity, 5), self.optimal_omega_params, self.optimal_detuning_params)

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)