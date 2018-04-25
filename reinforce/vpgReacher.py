import numpy as np
import gym
import scipy
import scipy.special
import pickle


env = gym.make('Reacher-v2')

# preparation
# ----------------------------------------


n_steps = 50 # actions of each rollout
n_iter = 6000 # iterations of training
batch_size = 50 # samples of each iteration

dim_sp = env.observation_space.shape[0]
dim_act = env.action_space.shape[0]
dim_theta = (dim_sp + 1) * dim_act

theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)


# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = True





# helper functions
# ----------------------------------------

def rolloutReward(theta):

    total = 0
    cur_state = env.reset()

    for t in range(n_steps):

        # print("dim_theta_before_genAct", theta.size)
        a = generateAct(theta, cur_state)
        next, reward, done, info = env.step(a)

        total += reward * 1
        #  total += reward * discount ** t

        if done: break

    return total



def generateAct(theta, state):

    # print("dim_theta_before_W", theta.size)
    W = theta[0 : dim_sp * dim_act].reshape(dim_sp, dim_act)
    b = theta[dim_sp * dim_act : None]

    a = np.clip(state.dot(W) + b, env.action_space.low, env.action_space.high)

    return a



def testRollout(theta):

    total = 0
    cur_state = env.reset()

    for t in range(n_steps):

        a = generateAct(theta, cur_state)
        next, reward, done, info = env.step(a)

        total += reward * discount ** t

        if t % 3 == 0:
            env.render()

        if done: break

    return total



def weighted_sample(logits, rng=np.random):
    weights = softmax(logits)
    return min(
        int(np.sum(rng.uniform() > np.cumsum(weights))),
        len(weights) - 1
    )


def compute_logits(theta, ob):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :return: A vector of size |A|
    """
    ob_1 = include_bias(ob)
    logits = ob_1.dot(theta.T)
    return logits


def log_softmax(logits):
    return logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)


def softmax(logits):
    x = logits
    x = x - np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def include_bias(x):
    # Add a constant term (1.0) to each entry in x
    return np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)



def get_action(theta, ob, rng=np.random):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :return: An integer
    """
    return weighted_sample(compute_logits(theta, ob), rng=rng)


def get_grad_logp_action(theta, ob, action):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :param action: An integer
    :return: A matrix of size |A| * (|S|+1)
    """
    # grad = np.zeros_like(theta)
    a = np.zeros(theta.shape[0])
    a[action] = 1
    p = softmax(compute_logits(theta, ob))
    ob_1 = include_bias(ob)
    return np.outer(a - p, ob_1)


# main
# ----------------------------------------

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# get_action = generateAct
# get_grad_logp_action = get_grad_logp_action

env.seed(42)
timestep_limit = env.spec.timestep_limit

# Initialize parameters
rng = np.random.RandomState(42)
theta = rng.normal(scale=0.1, size=(action_dim, obs_dim + 1))

# Store baselines for each time step.
baselines = np.zeros(timestep_limit)






# iterations
# ----------------------------------------

n_itrs = 1000

for itr in range(n_itrs):


        # Collect trajectory loop
        n_samples = 0
        grad = np.zeros_like(theta)
        episode_rewards = []

        # Store cumulative returns for each time step
        all_returns = [[] for _ in range(timestep_limit)]

        all_observations = []
        all_actions = []


        while n_samples < batch_size:
            observations = []
            actions = []
            rewards = []
            ob = env.reset()
            done = False

            # Only render the first trajectory
            render_episode = n_samples == 0

            # Collect a new trajectory
            while not done:

                action = get_action(theta, ob, rng=rng)
                next_ob, rew, done, _ = env.step(action)

                observations.append(ob)
                actions.append(action)
                rewards.append(rew)

                ob = next_ob
                n_samples += 1
                if render and render_episode:
                    env.render()


            # Go back in time to compute returns and accumulate gradient
            # Compute the gradient along this trajectory
            R = 0.
            for t in reversed(range(len(observations))):
                print (a)


