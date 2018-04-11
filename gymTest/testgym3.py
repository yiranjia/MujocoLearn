#  https://gym.openai.com/docs/
import gym

# env = gym.make('CartPole-v0')
#
# print(env.action_space)
# #> Discrete(2)
# print(env.observation_space)
# #> Box(4,)

# print(env.observation_space.high)
# #> array([ 2.4       ,         inf,  0.20943951,         inf])
# print(env.observation_space.low)
# #> array([-2.4       ,        -inf, -0.20943951,        -inf])

# from gym import envs
# print(envs.registry.all())


env = gym.make('CartPole-v0')

print(env.action_space)
# Discrete(2)
#> Box(2,)

print(env.observation_space)
# Box(4,)
#> Box(11,)



print(env.observation_space.high)
#> array([inf inf inf inf inf inf inf inf inf inf inf])

print(env.observation_space.low)
#> array([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf])

# [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
# [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]



# print(env.action_space.high)
# print(env.action_space.low)