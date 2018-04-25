import numpy as np
import gym

env = gym.make('InvertedPendulum-v2')


# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break


# print(env.action_space)
# #> Box(1,)
# print(env.observation_space)
# #> Box(10,)
#
# print(env.observation_space.high)
# #> array([inf inf inf inf inf inf inf inf inf inf])
# print(env.observation_space.low)
# #> array([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf])
# print(env.action_space.high)
# # [1.]
# print(env.action_space.low)
# # [-1.]






# preparation
# ----------------------------------------


n_steps = 50 # actions of each rollout
n_iter = 6000 # iterations of training
batch_size = 50 # samples of each iteration

top = 0.3
n_top = int(batch_size * top)
discount=0.9

dim_sp = env.observation_space.shape[0]
dim_act = env.action_space.shape[0]
dim_theta = (dim_sp + 1) * dim_act

# print("dim_act", dim_act)
# print("dim_theta", dim_theta)

# dim_mu = dim_theta
# dim_sig = dim_theta
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)






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








# main
# ----------------------------------------


# iterations
for itr in range(n_iter):

    thetaS = np.random.multivariate_normal(mean=theta_mean, cov = np.diag(np.array(theta_std**2)), size=batch_size)
    rewards = np.apply_along_axis(rolloutReward, 1, thetaS)
    # rewards = np.array(map(rolloutReward, thetaS))
    # rewards = np.array(rolloutReward(thetaS))

    top_ind = rewards.argsort()[-n_top:]
    top_thetas = thetaS[top_ind]

    theta_mean = top_thetas.mean(axis=0)
    theta_std = top_thetas.std(axis=0)

    if itr % 25 == 0:
        print ("iteration %i. mean reward: %8.3g. max reward: %8.3g. " % (itr, np.mean(rewards), np.max(rewards)))
        # print ("mean theta:", theta_mean)

    if itr % 1 == 0:
        testRollout(theta_mean)

    # testRollout(theta_mean)





