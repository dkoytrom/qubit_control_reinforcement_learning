import tensorflow as tf
import numpy as np
import random

from common.common import get_average_return
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common as cmn

# Define TD learning algorithms Q-Learning, SARSA and Double QLearning
def double_qlearing_algorithm(environment, omegas, collect_policy, nb_episodes = 2000, learning_rate = 0.9, discount = 0.95, max_steps = 30, nb_actions = 3):
    Qtable1 = np.zeros((max_steps, len(omegas), nb_actions))
    Qtable2 = np.zeros((max_steps, len(omegas), nb_actions))
    rewards = np.zeros(nb_episodes)

    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005

    for episode in range(nb_episodes):
        # print("EPISODE:", episode)
        # initial state
        t = 0
        Ω = 0
        total_reward = 0

        ε = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        initial_state = environment.reset()

        learning_rate = 1 / (episode + 1) ** 0.5

        learning_rate = max(learning_rate, 0.05)

        for time in range(max_steps):
            # select an action
            Q1_plus_Q2 = np.add(Qtable1, Qtable2) / 2
            action_index: int = collect_policy(Q1_plus_Q2, (time, Ω), omegas, nb_actions = nb_actions, ε = ε)

            time_step = environment.step(action_index)

            reward = time_step.reward

            new_state = time_step.observation
            new_Ω = environment._Ω

            omega = omegas.index(Ω)
            new_omega = omegas.index(new_Ω)
            Ω = new_Ω

            # update rule, update Q1 or Q2 with prob 50%
            choise = random.choices([1, 2], [5, 5])[0]

            if choise == 1:
                if time < max_steps - 1 or not environment._episode_ended:
                    optimal_action = np.argmax(Qtable1[time + 1][new_omega])
                    new_state_Q = Qtable2[time + 1][new_omega][optimal_action]
                else: # : Q value of terminal state is 0
                    new_state_Q = 0

                
                Qtable1[time][omega][action_index] = Qtable1[time][omega][action_index] + learning_rate * (reward + discount * new_state_Q - Qtable1[time][omega][action_index])
            else:
                if time < max_steps - 1 or not environment._episode_ended:
                    optimal_action = np.argmax(Qtable2[time + 1][new_omega])
                    new_state_Q = Qtable1[time + 1][new_omega][optimal_action]
                else: # : Q value of terminal state is 0
                    new_state_Q = 0

                Qtable2[time][omega][action_index] = Qtable2[time][omega][action_index] + learning_rate * (reward + discount * new_state_Q - Qtable2[time][omega][action_index])

            total_reward += reward

            # check new state is terminal, if it is break loop
            if environment._episode_ended:
                break

        rewards[episode] = total_reward

    return (np.add(Qtable1, Qtable2) / 2, rewards)

def qlearing_algorithm(environment, omegas, collect_policy, nb_episodes = 2000, learning_rate = 0.9, discount = 0.95, max_steps = 100, nb_actions = 3):
    Qtable = np.zeros((max_steps, len(omegas), nb_actions))
    rewards = np.zeros(nb_episodes)

    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005

    for episode in range(nb_episodes):
        # initial state
        t = 0
        Ω = 0

        ε = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        initial_state = environment.reset()
        total_reward = 0

        learning_rate = 1 / (episode + 1) ** 0.5

        learning_rate = max(learning_rate, 0.05)

        for time in range(max_steps):
            # select an action
            action_index: int = collect_policy(Qtable, (time, Ω), omegas, nb_actions = nb_actions, ε = ε)
        
            time_step = environment.step(action_index)

            reward = time_step.reward

            new_state = time_step.observation
            new_Ω = environment._Ω

            omega = omegas.index(Ω)
            new_omega = omegas.index(new_Ω)
            Ω = new_Ω

            if time < max_steps - 1:
                new_state_Q = Qtable[time + 1][new_omega]
            else: # : Q value of terminal state is 0
                new_state_Q = 0

            # update rule
            Qtable[time][omega][action_index] = Qtable[time][omega][action_index] + learning_rate * (reward + discount * np.max(new_state_Q) - Qtable[time][omega][action_index])

            total_reward += reward

            # check new state is terminal, if it is break loop
            if environment._episode_ended:
                break

        rewards[episode] = total_reward
        
    return (Qtable, rewards)

def sarsa_algorithm(environment, omegas, collect_policy, nb_episodes = 2000, learning_rate = 0.1, discount = 0.95, max_steps = 100, nb_actions = 3):
    Qtable = np.zeros((max_steps, len(omegas), nb_actions))
    # initialize with a negative because we panish each step and initial rewards may be negative
    # Qtable.fill(0)
    actions = environment._actions

    rewards = np.zeros(nb_episodes)
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005

    for episode in range(nb_episodes):
        # initial state
        t = 0
        Ω = 0

        ε = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

        learning_rate = 1 / (episode + 1) ** 0.5

        # initialize state
        initial_state = environment.reset()
        state1 = initial_state.observation
        t1, Ω1 = state1
        omega1 = omegas.index(Ω1)

        # choose action from ε-greedy policy
        action_index1: int = collect_policy(Qtable, (t, Ω), omegas, nb_actions = nb_actions, ε = ε)
        
        total_reward = 0

        for time in range(max_steps):
            # take action and observe the environment
            # time_step = environment.step(action1)   
            time_step = environment.step(action_index1)         

            # new reward
            reward = time_step.reward
            # print("REWARD =", reward)

            # new state
            state2 = time_step.observation
            t2, Ω2 = state2

            omega2 = omegas.index(Ω2)

            # take a new action based on new state
            action_index2: int = collect_policy(Qtable, (t2, Ω2), omegas, nb_actions = nb_actions)

            # prediction
            prediction = Qtable[t1][omega1][action_index1]

            # value of next state action pair
            new_reward = reward + discount * Qtable[t2][omega2][action_index2]

            # update rule
            Qtable[t1][omega1][action_index1] = Qtable[t1][omega1][action_index1] + learning_rate * (new_reward - prediction)

            total_reward += reward

            # second state becomes 1st
            t1, Ω1 = t2, Ω2

            # action 2 becomes 1
            action_index1 = action_index2

            if t2 == max_steps - 1 or environment._episode_ended is True:
                break

        rewards[episode] = total_reward

    return (Qtable, rewards)

def expected_sarsa(environment, omegas, collect_policy, nb_episodes = 2000, learning_rate = 0.9, discount = 0.95, max_steps = 100, nb_actions = 3):
    Qtable = np.zeros((max_steps, len(omegas), nb_actions))
    Qtable.fill(-30)
    rewards = np.zeros(nb_episodes)

    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005

    for episode in range(nb_episodes):
        # initial state
        t = 0
        Ω = 0

        ε = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        initial_state = environment.reset()
        total_reward = 0

        learning_rate = 1 / (episode + 1) ** 0.5

        learning_rate = max(learning_rate, 0.05)

        for time in range(max_steps):
            # select an action
            action_index: int = collect_policy(Qtable, (time, Ω), omegas, nb_actions = nb_actions, ε = ε)
        
            time_step = environment.step(action_index)

            reward = time_step.reward

            new_state = time_step.observation
            new_Ω = environment._Ω

            omega = omegas.index(Ω)
            new_omega = omegas.index(new_Ω)
            Ω = new_Ω

            if time < max_steps - 1:
                new_state_Q = Qtable[time + 1][new_omega]
            else: # : Q value of terminal state is 0
                new_state_Q = 0

            # update rule
            # create an array with no negatives and no zeros
            new_state_Q_for_probs = new_state_Q + np.min(new_state_Q) + 10
            probabilities = new_state_Q_for_probs / np.sum(new_state_Q_for_probs)
            expected_value = np.dot(probabilities, new_state_Q)

            Qtable[time][omega][action_index] = Qtable[time][omega][action_index] + learning_rate * (reward + discount * expected_value - Qtable[time][omega][action_index])

            total_reward += reward

            # check new state is terminal, if it is break loop
            if environment._episode_ended:
                break

        rewards[episode] = total_reward
        
    return (Qtable, rewards)

def Dqn(environment, eval_environement, nb_iterations = 2000, 
        learning_rate = 1e-3, gamma = 0.99, max_steps = 100, 
        epsilon_greedy = 0.1, fc_layer_params = ((100, 75)), checkpoint_dir = None,
        replay_buffer_capacity = 75000
    ):
    batch_size = 64
    log_interval = 50
    eval_interval = 10
    gamma = 0.99

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    avg_returns = []

    # wrap environment into a tf env
    train_env  = tf_py_environment.TFPyEnvironment(environment)
    eval_env = tf_py_environment.TFPyEnvironment(eval_environement)

    train_step_counter = tf.Variable(0)

    # create a neural network
    q_net = QNetwork(
        input_tensor_spec = train_env.time_step_spec().observation,
        action_spec = train_env.action_spec(),
        fc_layer_params = fc_layer_params
    )

    # create a q network agent
    agent = DqnAgent(
        time_step_spec = environment.time_step_spec(),
        action_spec = environment.action_spec(),
        q_network = q_net,
        optimizer = optimizer,
        td_errors_loss_fn = common.element_wise_squared_loss,
        train_step_counter = train_step_counter,
        epsilon_greedy = epsilon_greedy,
        gamma = gamma,
    )

    agent.initialize()

    # collect data
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = agent.collect_data_spec,
        batch_size = train_env.batch_size,
        max_length = replay_buffer_capacity
    )

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Reset the environment.
    time_step = environment.reset()

    # create a driver which will run the simulations and collect tha dataset for our training
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers = [replay_buffer.add_batch],
        num_steps = max_steps
    )

    # create checkpointer to load training and continue
    if checkpoint_dir is not None:
        checkpointer = cmn.Checkpointer(
            ckpt_dir = checkpoint_dir,
            max_to_keep = 1,
            agent = agent,
            policy = agent.policy,
            replay_buffer = replay_buffer,
        )

        checkpointer.initialize_or_restore()

    # loop for all iterations
    for _ in range(nb_iterations):
        collect_driver.run()

        # get the dataset
        dataset = replay_buffer.as_dataset(
            num_parallel_calls = 3,
            sample_batch_size = batch_size,
            num_steps = 2
        ).prefetch(3)

        iterator = iter(dataset)

        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = get_average_return(eval_env, agent.policy, 10)
            avg_returns.append(avg_return)

    if checkpoint_dir is not None:
        checkpointer.save(train_step_counter)

    return agent, q_net, replay_buffer, avg_returns

def Ddqn(environment, eval_environement, 
        nb_iterations = 2000, learning_rate = 1e-3, gamma = 0.99, max_steps = 100, 
        epsilon_greedy = 0.1, fc_layer_params = ((100, 75)), checkpoint_dir = None,
        replay_buffer_capacity = 75000
    ):
    
    batch_size = 64
    log_interval = 50
    eval_interval = 10
    gamma = 0.99

    # optimize with Adap
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    # global_step = tf.compat.v1.train.get_or_create_global_step()
    avg_returns = []

    # wrap environment into a tf env
    train_env  = tf_py_environment.TFPyEnvironment(environment)
    eval_env = tf_py_environment.TFPyEnvironment(eval_environement)

    train_step_counter = tf.Variable(0)

    # create a neural network
    q_net = QNetwork(
        input_tensor_spec = train_env.time_step_spec().observation,
        action_spec = train_env.action_spec(),
        fc_layer_params = fc_layer_params
    )

    # create a q network agent
    agent = DdqnAgent(
        time_step_spec = environment.time_step_spec(),
        action_spec = environment.action_spec(),
        q_network = q_net,
        optimizer = optimizer,
        td_errors_loss_fn = common.element_wise_squared_loss,
        train_step_counter = train_step_counter,
        epsilon_greedy = epsilon_greedy,
        gamma = gamma,
        n_step_update = 5
    )

    agent.initialize()

    # collect data
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = agent.collect_data_spec,
        batch_size = train_env.batch_size,
        max_length = replay_buffer_capacity
    )

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Reset the environment.
    time_step = environment.reset()

    # create a driver which will run the simulations and collect tha dataset for our training
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers = [replay_buffer.add_batch],
        num_steps = max_steps
    )

    # create checkpointer to load training and continue
    if checkpoint_dir is not None:
        checkpointer = cmn.Checkpointer(
            ckpt_dir = checkpoint_dir,
            max_to_keep = 1,
            agent = agent,
            policy = agent.policy,
            replay_buffer = replay_buffer,
        )

        checkpointer.initialize_or_restore()

    # loop for all iterations
    for _ in range(nb_iterations):
        collect_driver.run()

        # get the dataset
        dataset = replay_buffer.as_dataset(
            num_parallel_calls = 3,
            sample_batch_size = batch_size,
            num_steps = 2 + 4
        ).prefetch(3)

        iterator = iter(dataset)

        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = get_average_return(eval_env, agent.policy, 10)
            avg_returns.append(avg_return)

    if checkpoint_dir is not None:
        checkpointer.save(train_step_counter)

    return agent, q_net, replay_buffer, avg_returns