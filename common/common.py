import random
import numpy as np
from tf_agents.environments import tf_py_environment

def get_average_return(environment, policy, nb_episodes = 10):
    total_return = 0.0

    for _ in range(nb_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            action = action_step.action

            time_step = environment.step(action)
            episode_return += time_step.reward

        total_return += episode_return

    average_return = total_return / nb_episodes

    return average_return.numpy()[0]

def EpsilonGreedyPolicy(Qtable, state, omegas, ε = 0.05, nb_actions = 3):
    random_int = random.uniform(0, 1)

    # with ε probability choose a random action, else use greedy action
    if random_int > ε:
        t, Ω = state # tuple

        # break the ties randomly # {0, 1, ...}
        # action = np.argmax(Qtable[t][omegas.index(Ω)]) 
        Q_subtable = Qtable[t][omegas.index(Ω)]
        action = random.choice(np.flatnonzero(Q_subtable == Q_subtable.max()))
    else:
        action = random.randint(0, nb_actions - 1) # {0, 1, ...}

    return action

def extract_single_action_policy(environment, agent):
    greedy_actions = []
    optimal_omegas = []
    states = []
    fidelities = []
    optimal_policy = agent.policy
    t, Ω = (0, 0)

    environment.reset()
    tf_env = tf_py_environment.TFPyEnvironment(environment)
    time_step = tf_env.reset()

    states.append(environment._quantum_state)
    fidelities.append(0.0)
    
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

        fidelities.append(environment._fidelity ** 2)

        if environment._episode_ended:
            break

    # append another omega so that the last step can be shown in the figure
    optimal_omegas.append(optimal_omegas[-1])
        
    return (greedy_actions, optimal_omegas, states, fidelities)

def extract_double_action_policy(environment, agent):
    greedy_actions = []
    optimal_omegas = []
    optimal_detunning = []
    states = []
    fidelities = []
    optimal_policy = agent.policy
    t, Ω = (0, 0)

    environment.reset()
    tf_env = tf_py_environment.TFPyEnvironment(environment)
    time_step = tf_env.reset()

    states.append(environment._quantum_state)
    fidelities.append(0.0)
    
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
        Δ = environment._Δ

        # apply field in tha array
        optimal_omegas.append(Ω)
        optimal_detunning.append(Δ)

        fidelities.append(environment._fidelity ** 2)

        if environment._episode_ended:
            break
        
    # append another omega so that the last step can be shown in the figure
    optimal_omegas.append(optimal_omegas[-1])
    optimal_detunning.append(optimal_detunning[-1])

    optimal_controls = [optimal_omegas, optimal_detunning]

    return (greedy_actions, optimal_controls, states, fidelities)

def extract_triple_action_policy(environment, agent):
    greedy_actions = []
    optimal_omegas_a = []
    optimal_omegas_b = []
    optimal_detunning = []
    states = []
    fidelities = []
    optimal_policy = agent.policy
    t, Ω = (0, 0)

    environment.reset()
    tf_env = tf_py_environment.TFPyEnvironment(environment)
    time_step = tf_env.reset()

    states.append(environment._quantum_state)
    fidelities.append(0.0)
    
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
        Ωa = environment._Ωa
        Ωb = environment._Ωb
        Δ = environment._Δ

        # apply field in tha array
        optimal_omegas_a.append(Ωa)
        optimal_omegas_b.append(Ωb)
        optimal_detunning.append(Δ)

        fidelities.append(environment._fidelity ** 2)

        if environment._episode_ended:
            break
        
    # append another omega so that the last step can be shown in the figure
    optimal_omegas_a.append(optimal_omegas_a[-1])
    optimal_omegas_b.append(optimal_omegas_b[-1])
    optimal_detunning.append(optimal_detunning[-1])

    optimal_controls = [optimal_omegas_a, optimal_omegas_b, optimal_detunning]

    return (greedy_actions, optimal_controls, states, fidelities)

def extract_trigonometric_controls(environment, agent):
    greedy_actions = []
    states = []
    optimal_policy = agent.policy

    environment.reset()
    tf_env = tf_py_environment.TFPyEnvironment(environment)
    time_step = tf_env.reset()

    states.append(environment._quantum_state)
    
    # get action(s) from policy network
    policy_step = optimal_policy.action(time_step)

    # add action into array of actions
    greedy_actions.append(policy_step)

    # apply action into the environment
    time_step = tf_env.step(policy_step)

    states.append(environment._quantum_state)

    squared_fidelity = environment._fidelity ** 2
    
    return (policy_step, states, squared_fidelity)