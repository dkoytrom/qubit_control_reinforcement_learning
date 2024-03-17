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
from tf_agents.utils.common import OUProcess

import matplotlib.pyplot as plt
import qutip
from qutip import Bloch, QobjEvo, basis, sesolve, sigmay, sigmaz, sigmax, expect, fidelity, sigmap, sigmam, ket2dm, rand_ket
import random

"""
NAME: QubitEnv
DESC: 
"""
class QubitEnv(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, actions, restriction_fg = True, fidelity_threshold = 0.95, verbose_fg = False, nb_actions = 3, seed = 1989):
        if seed is not None:
            tf.random.set_seed(seed)
        
        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (), dtype = np.int32, minimum = 0, maximum = nb_actions - 1, name = 'action')
        self._nb_actions = nb_actions

        # define the state space
        self._observation_spec = array_spec.BoundedArraySpec(shape = (2, ), dtype = np.float32, minimum = [0., -max_Ω + 0.], maximum = [T + 0., max_Ω + 0.], name = 'state')

        # initial state (time step, magnetic field)
        self._state = [0, 0]
        self._quantum_state = basis(2, 0).unit()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg

        if (self._verbose_fg):
            print("INITIAL FIDELITY", fidelity(self._quantum_state, self._final_quantum_state)**2)

        self._time_step = 0
        self._Δ = 0
        self._Ω = 0
        self._omegas = []
        self._T = T
        self._N = N
        self._max_steps = N
        self._max_Ω = max_Ω

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False
        self._actions = actions

    """>
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
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
        # self._state = (0, 0)
        self._state = [0, 0]
        self._quantum_state = basis(2, 0).unit()
        self._time_step = 0
        self._Ω = 0
        self._episode_ended = False
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

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

        # evolve quantum state with the new Hamiltonian created by the action        
        action_value = self._actions[action]

        # restrict Ω between (-max, +max)
        # get new Ω
        new_Ω = self._Ω + action_value

        # if it exceeds max value, get max value
        if new_Ω < - self._max_Ω:
            self._Ω = - self._max_Ω
        elif new_Ω > self._max_Ω:
            self._Ω = self._max_Ω
        else:
            self._Ω = new_Ω

        # print("action index = ", action, " δΩ = ",  action_value, "new Ω = ", new_Ω, " Final Ω = ", self._Ω)

        omega = self._Ω
        self._omegas.append(omega)

        Ω = lambda t, args: omega
        Δ = lambda t, args: 0
        
        H = QobjEvo([[0.5 * sigmaz(), Δ], [0.5 * sigmax(), Ω]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T / N, num = 2)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[1]

        # compute fidelity at this time step        
        entanglement_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = entanglement_fidelity

        # update evironment state (t, Ω)
        self._state = (self._time_step, omega)
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = entanglement_fidelity ** 2

        if self._time_step == N or squared_fidelity >= self._fidelity_threshold:
            self._episode_ended = True

            # we have to calculate the inner product between final state and expected final state
            reward = entanglement_fidelity

            # maybe reward also sequences that achive the fidelity with a constant reward
            if squared_fidelity >= self._fidelity_threshold:
                reward += 10

            if (self._verbose_fg):
                print("Episode ended: fidelity = ", entanglement_fidelity, "time steps = ", self._time_step)
        else:
            # we can penalty long sequences
            reward = entanglement_fidelity - 1

        if (self._verbose_fg):
            print("time step",  self._time_step, ": a = ", action, " s = ", (self._time_step, omega), " r = ", reward, " fidelity ", " = ", round(entanglement_fidelity, 4))

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)
        
"""
NAME: QubitStochasticEnv
DESC: 
"""
class QubitStochasticEnv(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, actions, restriction_fg = True, fidelity_threshold = 0.95, verbose_fg = False, nb_actions = 3, seed = 1989):
        if seed is not None:
            tf.random.set_seed(seed)
        
        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (), dtype = np.int32, minimum = 0, maximum = nb_actions - 1, name = 'action')
        self._nb_actions = nb_actions

        # define the state space
        self._observation_spec = array_spec.BoundedArraySpec(shape = (2, ), dtype = np.float32, minimum = [0., -max_Ω + 0.], maximum = [T + 0., max_Ω + 0.], name = 'state')

        # initial state (time step, magnetic field)
        self._state = [0, 0]
        self._quantum_state = basis(2, 0).unit()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg

        if (self._verbose_fg):
            print("INITIAL FIDELITY", fidelity(self._quantum_state, self._final_quantum_state)**2)

        self._time_step = 0
        self._Δ = 0
        self._Ω = 0
        self._omegas = []
        self._T = T
        self._N = N
        self._max_steps = N
        self._max_Ω = max_Ω

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False
        self._actions = actions

        self.noise_model = OUProcess([0.0], seed = 1989)
        self.error = self.noise_model().numpy()[0]

    """>
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
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
        # self._state = (0, 0)
        self._state = [0, 0]
        self._quantum_state = basis(2, 0).unit()
        self._time_step = 0
        self._Ω = 0
        self._episode_ended = False
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

        self.noise_model = OUProcess([0.0], seed = 1989)
        self.error = self.noise_model().numpy()[0]

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

        # evolve quantum state with the new Hamiltonian created by the action        
        action_value = self._actions[action]

        # restrict Ω between (-max, +max)
        # get new Ω
        new_Ω = self._Ω + action_value

        # if it exceeds max value, get max value
        if new_Ω < - self._max_Ω:
            self._Ω = - self._max_Ω
        elif new_Ω > self._max_Ω:
            self._Ω = self._max_Ω
        else:
            self._Ω = new_Ω

        # print("action index = ", action, " δΩ = ",  action_value, "new Ω = ", new_Ω, " Final Ω = ", self._Ω)

        omega = self._Ω

        # get the stochastic error for the rabi frequency
        # error model will be e(t) * σz
        self.error = self.noise_model().numpy()[0]

        self._omegas.append(omega)

        Ω = lambda t, args: omega
        Δ = lambda t, args: 0 + self.error
        
        H = QobjEvo([[0.5 * sigmaz(), Δ], [0.5 * sigmax(), Ω]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T / N, num = 2)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[1]

        # compute fidelity at this time step        
        entanglement_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = entanglement_fidelity

        # update evironment state (t, Ω)
        self._state = (self._time_step, omega)
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = entanglement_fidelity ** 2

        if self._time_step == N or squared_fidelity >= self._fidelity_threshold:
            self._episode_ended = True

            # we have to calculate the inner product between final state and expected final state
            reward = entanglement_fidelity

            # maybe reward also sequences that achive the fidelity with a constant reward
            if squared_fidelity >= self._fidelity_threshold:
                reward += 10

            if (self._verbose_fg):
                print("Episode ended: fidelity = ", entanglement_fidelity, "time steps = ", self._time_step)
        else:
            # we can penalty long sequences
            reward = entanglement_fidelity - 1

        if (self._verbose_fg):
            print("time step",  self._time_step, ": a = ", action, " s = ", (self._time_step, omega), " r = ", reward, " fidelity ", " = ", round(entanglement_fidelity, 4))

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)
        
"""
NAME: QubitStateContEnv
DESC: 
"""
class QubitStateContEnv(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, actions, restriction_fg = True, fidelity_threshold = 0.95, verbose_fg = False, nb_actions = 3, seed = 1989):
        if seed is not None:
            tf.random.set_seed(seed)
        
        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (), dtype = np.int32, minimum = 0, maximum = nb_actions - 1, name = 'action')
        self._nb_actions = nb_actions

        # define the state space
        # self._observation_spec = array_spec.BoundedArraySpec(shape = (2, ), dtype = np.float32, minimum = [0., -max_Ω + 0.], maximum = [T + 0., max_Ω + 0.], name = 'state')
        self._observation_spec = array_spec.BoundedArraySpec(shape = (4, ), dtype = np.float32, minimum = [-max_Ω, 0., 0., 0.], maximum = [max_Ω, 1., 1., 1.], name = 'state')

        # initial state (time step, magnetic field)
        self._state = [0, 0, 0, 0]
        self._quantum_state = basis(2, 0).unit()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg

        if (self._verbose_fg):
            print("INITIAL FIDELITY", fidelity(self._quantum_state, self._final_quantum_state)**2)

        self._time_step = 0
        self._Δ = 0
        self._Ω = 0
        self._omegas = []
        self._T = T
        self._N = N
        self._max_steps = N
        self._max_Ω = max_Ω

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False
        self._actions = actions

    """>
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
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
        # self._state = (0, 0)
        self._state = [0, 0, 0, 0]
        self._quantum_state = basis(2, 0).unit()
        self._time_step = 0
        self._Ω = 0
        self._episode_ended = False
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

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
        # State space: S = (Ω, ρ11, Re{ρ12}, Im{ρ12})
        # Actions space: A = {-1, 1} (bang-bang pulses) inspired by optimal control

        # increment time step
        self._time_step += 1

        # evolve quantum state with the new Hamiltonian created by the action        
        action_value = self._actions[action]

        # restrict Ω between (-max, +max)
        # get new Ω
        new_Ω = self._Ω + action_value

        # if it exceeds max value, get max value
        if new_Ω < - self._max_Ω:
            self._Ω = - self._max_Ω
        elif new_Ω > self._max_Ω:
            self._Ω = self._max_Ω
        else:
            self._Ω = new_Ω

        omega = self._Ω
        self._omegas.append(omega)

        Ω = lambda t, args: omega
        Δ = lambda t, args: 0
        
        H = QobjEvo([[0.5 * sigmaz(), Δ], [0.5 * sigmax(), Ω]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T / N, num = 2)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[1]

        # compute fidelity at this time step        
        rho = ket2dm(self._quantum_state)

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        # compute fidelity at this time step        
        # entanglement_fidelity = fidelity(self._final_quantum_state, self._quantum_state)**2
        current_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = current_fidelity

        # update evironment state (t, Ω)
        self._state = (omega, np.real(rho11), np.real(rho12), np.imag(rho12))
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = current_fidelity ** 2

        if self._time_step == N or squared_fidelity >= self._fidelity_threshold:
            self._episode_ended = True

            # we have to calculate the inner product between final state and expected final state
            reward = current_fidelity

            # maybe reward also sequences that achive the fidelity with a constant reward
            if squared_fidelity >= self._fidelity_threshold:
                reward += 10

            if (self._verbose_fg):
                print("Episode ended: fidelity = ", current_fidelity, "time steps = ", self._time_step)
        else:
            # we can penalty long sequences
            reward = current_fidelity - 1

        if (self._verbose_fg):
            print("time step",  self._time_step, ": a = ", action, " s = ", (self._time_step, omega), " r = ", reward, " fidelity ", " = ", round(current_fidelity, 4))

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)

"""
NAME: QubitResonantContEnv
DESC: 
"""
class QubitResonantContEnv(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, restriction_fg = True, fidelity_threshold = 0.95, verbose_fg = False, seed = 1989):
        if seed is not None:
            tf.random.set_seed(seed)

        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (), dtype = np.float32, minimum = -2 * max_Ω, maximum = 2 * max_Ω, name = 'action')

        # define the state space
        self._observation_spec = array_spec.BoundedArraySpec(shape = (3, ), dtype = np.float32, minimum = [0., -max_Ω, 0.], maximum = [T, max_Ω, 1.], name = 'state')

        self._quantum_state = basis(2, 0).unit()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

        # initial state (time step, magnetic field)
        self._state = [0, 0, self._fidelity]

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

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False

    """>
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
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

        self._state = [0, 0, self._fidelity]

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

        # evolve quantum state with the new Hamiltonian created by the action        
        # restrict Ω between (-max, +max)
        # get new Ω
        new_Ω = self._Ω + action

        # if it exceeds max value, get max value
        if new_Ω < - self._max_Ω:
            self._Ω = - self._max_Ω
        elif new_Ω > self._max_Ω:
            self._Ω = self._max_Ω
        else:
            self._Ω = new_Ω

        # print("action index = ", action, " δΩ = ",  action_value, "new Ω = ", new_Ω, " Final Ω = ", self._Ω)

        omega = self._Ω
        self._omegas.append(omega)

        Ω = lambda t, args: omega
        Δ = lambda t, args: 0
        
        H = QobjEvo([[0.5 * sigmaz(), Δ], [0.5 * sigmax(), Ω]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T / N, num = 2)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[1]

        # compute fidelity at this time step        
        entanglement_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = entanglement_fidelity

        # update evironment state (t, Ω)
        self._state = (self._time_step, omega, self._fidelity)
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = entanglement_fidelity ** 2

        if self._time_step == N or squared_fidelity >= self._fidelity_threshold:
            self._episode_ended = True

            # we have to calculate the inner product between final state and expected final state
            reward = entanglement_fidelity

            # maybe reward also sequences that achive the fidelity with a constant reward
            if squared_fidelity >= self._fidelity_threshold:
                reward += 10
        else:
            # we can penalty long sequences
            reward = entanglement_fidelity - 1

        if (self._verbose_fg):
            print("time step",  self._time_step, ": a = ", action, " s = ", (self._time_step, omega), " r = ", reward, " fidelity ", " = ", round(entanglement_fidelity, 4))

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)
        
"""
NAME: QubitPIResonantContEnv
DESC: Physics informed
"""
class QubitPIResonantContEnv(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, restriction_fg = True, fidelity_threshold = 0.95, verbose_fg = False, seed = 1989):
        if seed is not None:
            tf.random.set_seed(seed)

        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (), dtype = np.float32, minimum = -2 * max_Ω, maximum = 2 * max_Ω, name = 'action')

        # define the state space
        self._observation_spec = array_spec.BoundedArraySpec(shape = (3, ), dtype = np.float32, minimum = [0., 0., 0.], maximum = [1., 1., 1.], name = 'state')

        self._quantum_state = basis(2, 0).unit()

        # update mdp state
        self._update_state()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

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

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False

    """>
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
    def _update_state(self):
        # initial state (time step, magnetic field)
        rho = ket2dm(self._quantum_state)

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

        # update mdp state
        self._update_state()

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

        # evolve quantum state with the new Hamiltonian created by the action        
        # restrict Ω between (-max, +max)
        # get new Ω
        new_Ω = self._Ω + action

        # if it exceeds max value, get max value
        if new_Ω < - self._max_Ω:
            self._Ω = - self._max_Ω
        elif new_Ω > self._max_Ω:
            self._Ω = self._max_Ω
        else:
            self._Ω = new_Ω

        # print("action index = ", action, " δΩ = ",  action_value, "new Ω = ", new_Ω, " Final Ω = ", self._Ω)

        omega = self._Ω
        self._omegas.append(omega)

        Ω = lambda t, args: omega
        Δ = lambda t, args: 0
        
        H = QobjEvo([[0.5 * sigmaz(), Δ], [0.5 * sigmax(), Ω]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T / N, num = 2)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[1]

        # compute fidelity at this time step        
        entanglement_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = entanglement_fidelity

        # update mdp state
        self._update_state()
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = entanglement_fidelity ** 2

        if self._time_step == N or squared_fidelity >= self._fidelity_threshold:
            self._episode_ended = True

            # we have to calculate the inner product between final state and expected final state
            reward = entanglement_fidelity

            # maybe reward also sequences that achive the fidelity with a constant reward
            if squared_fidelity >= self._fidelity_threshold:
                reward += 10
        else:
            # we can penalty long sequences
            reward = entanglement_fidelity - 1

        if (self._verbose_fg):
            print("time step",  self._time_step, ": a = ", action, " s = ", (self._time_step, omega), " r = ", reward, " fidelity ", " = ", round(entanglement_fidelity, 4))

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)

"""
NAME: QubitContinuousEnv
DESC: 
"""
class QubitContinuousEnv(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, max_Δ, restriction_fg = True, fidelity_threshold = 0.95, verbose_fg = False, seed = 1989):
        if seed is not None:
            tf.random.set_seed(seed)
        
        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (2, ), dtype = np.float32, minimum = [-2 * max_Ω, - 2 * max_Δ], maximum = [2 * max_Ω, 2 * max_Δ], name = 'action')

        # define the state space
        self._observation_spec = array_spec.BoundedArraySpec(shape = (5, ), dtype = np.float32, minimum = [-max_Ω, -max_Δ, 0., 0., 0.], maximum = [max_Ω, max_Δ, 1., 1., 1.], name = 'state')

        self._quantum_state = basis(2, 0).unit()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

        # initial state (time step, magnetic field)
        rho = ket2dm(self._quantum_state)

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        self._state = [0, 0, np.real(rho11), np.real(rho12), np.imag(rho12)]

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

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False

    """
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
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

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        self._state = [0, 0, np.real(rho11), np.real(rho12), np.imag(rho12)]

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

        # evolve quantum state with the new Hamiltonian created by the action        
        # restrict Ω between (-max, +max)
        # get new Ω
        [δΩ, δΔ] = action
        new_Ω = self._Ω + δΩ
        new_Δ = self._Δ + δΔ

        # if it exceeds max value, get max value
        if new_Ω < - self._max_Ω:
            self._Ω = - self._max_Ω
        elif new_Ω > self._max_Ω:
            self._Ω = self._max_Ω
        else:
            self._Ω = new_Ω

        if new_Δ < - self._max_Δ:
            self._Δ = - self._max_Δ
        elif new_Δ > self._max_Δ:
            self._Δ = self._max_Δ
        else:
            self._Δ = new_Δ

        omega = self._Ω
        detuning = self._Δ
        self._omegas.append(omega)

        Ω = lambda t, args: omega
        Δ = lambda t, args: detuning
        
        H = QobjEvo([[0.5 * sigmaz(), Δ], [0.5 * sigmax(), Ω]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T / N, num = 2)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[1]

        rho = ket2dm(self._quantum_state)

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        # compute fidelity at this time step        
        entanglement_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = entanglement_fidelity

        # update evironment state (t, Ω)
        self._state = (omega, detuning, np.real(rho11), np.real(rho12), np.imag(rho12))
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = entanglement_fidelity ** 2

        if self._time_step == N or squared_fidelity >= self._fidelity_threshold:
            self._episode_ended = True

            # we have to calculate the inner product between final state and expected final state
            reward = entanglement_fidelity

            # maybe reward also sequences that achive the fidelity with a constant reward
            if squared_fidelity >= self._fidelity_threshold:
                reward += 10
        else:
            # we can penalty long sequences
            reward = entanglement_fidelity - 1

        if (self._verbose_fg):
            print("time step",  self._time_step, ": a = ", action, " s = ", (self._time_step, omega), " r = ", reward, " fidelity ", " = ", round(entanglement_fidelity, 4))

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)
        
"""
NAME: QubitContinuousDirectEnv
DESC: 
"""
class QubitContinuousDirectEnv(py_environment.PyEnvironment):
    """
    NAME: __init__
    DESC: 
    """
    def __init__(self, T, N, max_Ω, max_Δ, restriction_fg = True, fidelity_threshold = 0.95, verbose_fg = False, seed = 1989):
        if seed is not None:
            tf.random.set_seed(seed)
        
        # define the action space
        self._action_spec = array_spec.BoundedArraySpec(shape = (2, ), dtype = np.float32, minimum = [max_Ω, max_Δ], maximum = [max_Ω, max_Δ], name = 'action')

        # define the state space
        self._observation_spec = array_spec.BoundedArraySpec(shape = (5, ), dtype = np.float32, minimum = [-max_Ω, -max_Δ, 0., 0., 0.], maximum = [max_Ω, max_Δ, 1., 1., 1.], name = 'state')

        self._quantum_state = basis(2, 0).unit()

        # final state
        self._final_quantum_state = basis(2, 1).unit()
        self._verbose_fg = verbose_fg
        self._fidelity = fidelity(self._quantum_state, self._final_quantum_state)

        # initial state (time step, magnetic field)
        rho = ket2dm(self._quantum_state)

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        self._state = [0, 0, np.real(rho11), np.real(rho12), np.imag(rho12)]

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

        self._restriction = restriction_fg
        self._fidelity_threshold = fidelity_threshold
        self._episode_ended = False

    """
    NAME: action_spec
    DESC: 
    """
    def action_spec(self):
        return self._action_spec
    
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

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        self._state = [0, 0, np.real(rho11), np.real(rho12), np.imag(rho12)]

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

        # evolve quantum state with the new Hamiltonian created by the action        
        # restrict Ω between (-max, +max)

        [new_Ω, new_Δ] = action

        self._Ω = new_Ω
        self._Δ = new_Δ

        omega = self._Ω
        detuning = self._Δ
        self._omegas.append(omega)

        Ω = lambda t, args: omega
        Δ = lambda t, args: detuning
        
        H = QobjEvo([[0.5 * sigmaz(), Δ], [0.5 * sigmax(), Ω]])

        # define time span
        T = self._T
        N = self._N
        time = np.linspace(start = 0, stop = T / N, num = 2)

        result = sesolve(H, self._quantum_state, time, [])

        self._quantum_state = result.states[1]

        rho = ket2dm(self._quantum_state)

        rho11 = rho.matrix_element(basis(2, 0), basis(2, 0))
        rho12 = rho.matrix_element(basis(2, 0), basis(2, 1))

        # compute fidelity at this time step        
        entanglement_fidelity = fidelity(self._final_quantum_state, self._quantum_state)
        self._fidelity = entanglement_fidelity

        # update evironment state (t, Ω)
        self._state = (omega, detuning, np.real(rho11), np.real(rho12), np.imag(rho12))
        
        # if we have reached already final time, then reset the environment
        squared_fidelity = entanglement_fidelity ** 2

        if self._time_step == N or squared_fidelity >= self._fidelity_threshold:
            self._episode_ended = True

            # we have to calculate the inner product between final state and expected final state
            reward = entanglement_fidelity

            # maybe reward also sequences that achive the fidelity with a constant reward
            if squared_fidelity >= self._fidelity_threshold:
                reward += 10
        else:
            # we can penalty long sequences
            reward = entanglement_fidelity - 1

        if (self._verbose_fg):
            print("time step",  self._time_step, ": a = ", action, " s = ", (self._time_step, omega), " r = ", reward, " fidelity ", " = ", round(entanglement_fidelity, 4))

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype = np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype = np.float32), reward = reward, discount = 1.0)

def extract_policy(Qtable, environment, actions, omegas, max_steps):
    greedy_actions = []
    optimal_omegas = []
    states = []
    fidelities = []
    # t, Ω = (0, 0)

    time_step = environment.reset()
    t, Ω = time_step.observation

    states.append(environment._quantum_state)
    fidelities.append(environment._fidelity)
    
    while t < max_steps and environment._episode_ended is False:
        # get initial state (t, Ω)
        omega_index = omegas.index(Ω)
        
        # select maximum value action
        # TODO: here instead of Qtable we can use a neural network to get the estimate for the best action 
        # action_index = np.argmax(Qtable[t][omega_index])
        # Following command is like argmax, but will break ties randomly
        Q_subtable = Qtable[int(t)][omega_index]
        action_index = random.choice(np.flatnonzero(Q_subtable == Q_subtable.max()))

        # add action into array of actions
        greedy_actions.append(action_index)

        # get actual action from inde
        action = actions[action_index]

        # apply action into the environment
        time_step = environment.step(action_index)

        states.append(environment._quantum_state)

        # get the new state/observation after tha action
        t, Ω = time_step.observation

        # apply field in tha array
        optimal_omegas.append(Ω)

        fidelities.append(environment._fidelity ** 2)
        
    return (greedy_actions, optimal_omegas, states, fidelities)

def start_from_random_state(environment, agent):
    greedy_actions = []
    optimal_omegas = []
    states = []
    fidelities = []
    optimal_policy = agent.policy
    t, Ω = (0, 0)

    environment.reset()
    tf_env = tf_py_environment.TFPyEnvironment(environment)
    time_step = tf_env.reset()

    environment._quantum_state = rand_ket(2)
    print(environment._quantum_state)
    
    environment._fidelity = fidelity(environment._quantum_state, environment._final_quantum_state)

    environment._update_state()

    states.append(environment._quantum_state)
    fidelities.append(environment._fidelity)
    
    while t < environment._max_steps: #and tf_env._episode_ended is False:        
        # get action(s) from policy network
        policy_step = optimal_policy.action(time_step)

        # add action into array of actions
        greedy_actions.append(policy_step)

        # apply action into the environment
        # time_step = tf_env.step(action_index)
        time_step = tf_env.step(policy_step)

        states.append(environment._quantum_state)

        # get the new state/observation after tha action
        # t, Ω = time_step.observation
        Ω = environment._Ω

        # apply field in tha array
        optimal_omegas.append(Ω)

        fidelities.append(environment._fidelity)

        if environment._episode_ended:
            break

        t += 1

    # append another omega so that the last step can be shown in the figure
    optimal_omegas.append(optimal_omegas[-1])
        
    return (greedy_actions, optimal_omegas, states, fidelities)