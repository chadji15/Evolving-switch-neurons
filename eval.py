import gym
import gym_association_task
import t_maze

def clamp(x,low,high):
    if x < low:
        return low
    if x > high:
        return high
    return x


###
#All the following evaluation functions take as arguement an agent variable. It is assumed that the agent has
#an activate function which takes as input a vector (list) and returns an output which corresponds to the action
#of the agent. The actions are described in each environment
##############

#For a network to be considered to be able to solve the one-to-one 3x3 association task in this case it needs to
#to achieve a score of at least 1976 (2000 - 4*(3*2) = steps - association_changes*(n*m)).
#Note that scores above this threshold do not mean better performance since the score of 1976 is already considered optimal.
#The network here needs to accept 4 inputs (3 for observation and 1 for reward) and return a vector with 3 binary values.
def eval_one_to_one_3x3(agent):
    env = gym.make('OneToOne3x3-v0')
    num_episodes = 2000
    s = num_episodes
    observation = env.reset(rand_iter=500)
    input = tuple(list(observation) + [0])
    for i_episode in range(num_episodes):
        action = agent.activate(input)
        observation, reward, done, info = env.step(action)
        input = list(input)
        input[-1] = reward
        agent.activate(input)
        input = tuple(list(observation) + [0])
        s += reward
    env.close()
    return s

#For a network to be considered to be able to solve the one-to-many 3x2 association task in this case it needs to
#to achieve a score of at least 1964 (2000 - 4*(3*(4-1)) = steps - association_changes*(n*(2^m - 1))).
#Note that scores above this threshold do not mean better performance since the score of 1964 is already considered optimal.
#The network accepts 4 inputs (3 for observation and 1 for reward)  and return a vector with two binary values.
def eval_one_to_many_3x2(agent):
    env = gym.make('OneToMany3x2-v0')
    num_episodes = 2000
    sum = num_episodes
    observation = env.reset(rand_iter=500)
    input = list(observation)
    input.append(0)
    for i_episode in range(num_episodes):
        action = agent.activate(input)
        observation, reward, done, info = env.step(action)
        input = list(input)
        input[-1] = reward
        agent.activate(input)
        input = list(observation)
        input.append(0)
        sum += reward
    env.close()
    return sum

def int_to_action(x):
    if x == 0:
        return "Left"
    if x ==1:
        return "Right"
    return "Forward"

#For a network to be considered to be able to solve the single t-maze non-homing task in this case it needs to
#to achieve a score of at least 95. This is because every time we change the place of the high reward, the optimal
#network needs only one step to figure it out and we change the the place of the high reward 5 times through the
#trial. Performance of 99 or 98 may indicate that network evolved some interesting properties, e.g. maybe it
#figured out when we change the place of the high reward so it doesn't even need that step to learn.
#The network should accept 4 inputs (is agent at home, is agent at turning point, is agent at maze end, reward) and
#return 1 scalar output
def eval_tmaze(agent):
    env = gym.make('MiniGrid-TMaze-v0')
    num_episodes = 100
    s = 0
    pos = 0
    for i_episode in range(num_episodes):
        reward = 0
        if i_episode % 20 == 0:
            pos = (pos + 1) % 2
        observation = env.reset(reward_pos= pos)
        #append 0 for the reward
        input = list(observation)
        input.append(0)
        done = False
        #DEBUG INFO
        #print("Episode: {}".format(i_episode))
        #print("High pos: {}".format(pos))
        while not done:
            action = agent.activate(input)
            observation, reward, done, info = env.step(action)
            input = list(observation)
            input.append(reward)
            #DEBUG INFO
            #print("     {}".format(int_to_action(action)))
        #print(input)
        s += reward
        agent.activate(input)
        #DEBUG INFO
        #print("Reward: {}".format(reward))
        #print("--------------")
    env.close()
    return s

#The simplest case for test the network's implementation. Use when in doubt about the rest.
#Optimal network should have a score of 4 or very close to 4.
#The network takes 2 input and returns one output

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

TRIALS = 100

def eval_net_xor(net):
    sum = 0
    for i in range(TRIALS):
        fitness = 4
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            if output[0] < 0:
                output[0] = 0
            elif output[0] > 1:
                output[0] =1
            fitness -= abs(output[0] - xo[0])
        sum += fitness
    return sum/TRIALS