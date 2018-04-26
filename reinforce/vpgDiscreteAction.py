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
discount = 0.99  # discount factor for reward
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
        total += reward
        #total += reward * discount ** t

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


def compute_entropy(logits):
    """
    :param logits: A matrix of size N * |A|
    :return: A vector of size N
    """
    logp = log_softmax(logits)
    return -np.sum(logp * np.exp(logp), axis=-1)







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



        # each iteration has batch_size samples
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


            # this episode / sample finished !!!
            # ------------------------------------------------------------
            # Go back in time to *compute returns* and *accumulate gradient*
            # Compute the gradient along this trajectory

            R = 0.
            for t in reversed(range(len(observations))):

                def compute_update(discount, R_tplus1, theta, s_t, a_t, r_t, b_t, get_grad_logp_action):
                    """
                    :param discount: A scalar
                    :param R_tplus1: A scalar
                    :param theta: A matrix of size |A| * (|S|+1)
                    :param s_t: A vector of size |S|
                    :param a_t: Either a vector of size |A| or an integer, depending on the environment
                    :param r_t: A scalar
                    :param b_t: A scalar
                    :param get_grad_logp_action: A function, mapping from (theta, ob, action) to the gradient (a
                    matrix of size |A| * (|S|+1) )
                    :return: A tuple, consisting of a scalar and a matrix of size |A| * (|S|+1)
                    """
                    R_t = discount * R_tplus1 + r_t
                    pg_theta = get_grad_logp_action(theta, s_t, a_t) * (R_t - b_t)  # modulate the gradient with advantage (PG magic happens right here.)
                    return R_t, pg_theta


                R, grad_t = compute_update(
                        discount = discount,
                        R_tplus1=R,
                        theta=theta,
                        s_t=observations[t],
                        a_t=actions[t],
                        r_t=rewards[t],
                        b_t=baselines[t],
                        get_grad_logp_action=get_grad_logp_action
                )

                all_returns[t].append(R)
                grad += grad_t


            # append this episode / sample 's rew obs actions to this iterations 's
            episode_rewards.append(np.sum(rewards))
            all_observations.extend(observations)
            all_actions.extend(actions)



        # this iterations (with batch_size samples) finished!!!

        # computing baseline
        # --------------------------------------------
        def compute_baselines(all_returns):
            """
            :param all_returns: A list of size T, where the t-th entry is a list of numbers, denoting the returns
            collected at time step t across different episodes
            :return: A vector of size T
            """
            baselines = np.zeros(len(all_returns))
            for t in range(len(all_returns)):
                if len(all_returns[t]) > 0:
                    baselines[t] = np.mean(all_returns[t])

            return baselines

        baselines = compute_baselines(all_returns)
        # baselines = np.zeros(timestep_limit)



        # update parameter theta
        # --------------------------------------------

        grad = grad / (np.linalg.norm(grad) + 1e-8) # Roughly normalize the gradient

        theta += learning_rate * grad


        # print statements
        # --------------------------------------------

        logits = compute_logits(theta, np.array(all_observations))
        ent = np.mean(compute_entropy(logits))
        perp = np.exp(ent)

        print("Iteration: %d AverageReturn: %.2f Entropy: %.2f Perplexity: %.2f |theta|_2: %.2f" % (
            itr, np.mean(episode_rewards), ent, perp, np.linalg.norm(theta)))
        # print("Iteration: %d AverageReturn: %.2f |theta|_2: %.2f" % (itr, np.mean(episode_rewards), np.linalg.norm(theta)))
