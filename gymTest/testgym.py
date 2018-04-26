import gym
env = gym.make('CartPole-v0')

dim_sp = env.observation_space.shape[0]
dim_act = env.action_space.n