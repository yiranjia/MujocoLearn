import gym


env = gym.make('InvertedPendulum-v2')

print(env.action_space)
# Box(1,)

print(env.observation_space)
# Box(4,)


