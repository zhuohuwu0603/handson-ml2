#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 12: Deep Learning and Security**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 12 Video Material
# 
# * Part 12.1: Introduction to the OpenAI Gym [[Video]](https://www.youtube.com/watch?v=_KbUxgyisjM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_01_ai_gym.ipynb)
# * **Part 12.2: Introduction to Q-Learning** [[Video]](https://www.youtube.com/watch?v=uwcXWe_Fra0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_02_qlearningreinforcement.ipynb)
# * Part 12.3: Keras Q-Learning in the OpenAI Gym [[Video]](https://www.youtube.com/watch?v=Ya1gYt63o3M&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_03_keras_reinforce.ipynb)
# * Part 12.4: Atari Games with Keras Neural Networks [[Video]](https://www.youtube.com/watch?v=t2yIu6cRa38&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_04_atari.ipynb)
# * Part 12.5: How Alpha Zero used Reinforcement Learning to Master Chess [[Video]](https://www.youtube.com/watch?v=ikDgyD7nVI8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_05_alpha_zero.ipynb)
# 

# # Part 12.2: Introduction to Q-Learning
# 
# ### Single Action Cart
# 
# Mountain car actions:
# 
# * 0 - Apply left force
# * 1 - Apply no force
# * 2 - Apply right force
# 
# State values:
# 
# * state[0] - Position 
# * state[1] - Velocity
# 
# The following shows a cart that simply applies full-force to climb the hill.  The cart is simply not strong enough.  It will need to use momentum from the hill behind it.

# In[1]:


import gym

env = gym.make("MountainCar-v0")
env.reset()
done = False

i = 0
while not done:
    i += 1
    state, reward, done, _ = env.step(2)
    env.render()
    print(f"Step {i}: State={state}, Reward={reward}")
    
env.close()


# ### Programmed Car
# 
# This is a car that I hand-programmed.  It uses a simple rule, but solves the problem. The programmed car constantly applies force to one direction or another.  It does not reset.  Whatever direction the car is currently rolling, it applies force in that direction.  Therefore, the car begins to climb a hill, is overpowered, and rolls backward.  However, once it begins to roll backwards force is immediately applied in this new direction.

# In[1]:


import gym

env = gym.make("MountainCar-v0")
state = env.reset()
done = False

i = 0
while not done:
    i += 1
    
    if state[1]>0:
        action = 2
    else:
        action = 0
    
    state, reward, done, _ = env.step(action)
    env.render()
    print(f"Step {i}: State={state}, Reward={reward}")
    
env.close()


# ### Reinforcement Learning
# 
# ![Reinforcement Learning](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/reinforcement.png "Reinforcement Learning")
# 
# 
# ### Q-Learning Car
# 
# We will now use Q-Learning to produce a car that learns to drive itself.  Look out Tesla! 
# 
# Q-Learning works by building a table that provides a lookup table to determine which of several actions should be taken. As we move through a number of training episodes this table is refined.
# 
# $ Q^{new}(s_{t},a_{t}) \leftarrow (1-\alpha) \cdot \underbrace{Q(s_{t},a_{t})}_{\text{old value}} + \underbrace{\alpha}_{\text{learning rate}} \cdot  \overbrace{\bigg( \underbrace{r_{t}}_{\text{reward}} + \underbrace{\gamma}_{\text{discount factor}} \cdot \underbrace{\max_{a}Q(s_{t+1}, a)}_{\text{estimate of optimal future value}} \bigg) }^{\text{learned value}} $

# In[2]:


import gym
import numpy as np

def calc_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/buckets
    return tuple(discrete_state.astype(np.int))  

def run_game(q_table, render, should_update):
    done = False
    discrete_state = calc_discrete_state(env.reset())
    success = False
    
    while not done:
        # Exploit or explore
        if np.random.random() > epsilon:
            # Exploit - use q-table to take current best action (and probably refine)
            action = np.argmax(q_table[discrete_state])
        else:
            # Explore - t
            action = np.random.randint(0, env.action_space.n)
            
        # Run simulation step
        new_state, reward, done, _ = env.step(action)
        
        # 
        new_state_disc = calc_discrete_state(new_state)

        # 
        if new_state[0] >= env.goal_position:
            success = True
          
        # Update q-table
        if should_update:
            max_future_q = np.max(q_table[new_state_disc])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_state_disc
        
        if render:
            env.render()
            
    return success


# In[3]:


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 1000

DISCRETE_GRID_SIZE = [10, 10]
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2


# In[4]:


env = gym.make("MountainCar-v0")

epsilon = 1  
epsilon_change = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
buckets = (env.observation_space.high - env.observation_space.low)/DISCRETE_GRID_SIZE
q_table = np.random.uniform(low=-3, high=0, size=(DISCRETE_GRID_SIZE + [env.action_space.n]))
success = False


# In[5]:


episode = 0
success_count = 0

while episode<EPISODES:
    episode+=1
    done = False

    if episode % SHOW_EVERY == 0:
        print(f"Current episode: {episode}, success: {success_count} ({float(success_count)/SHOW_EVERY})")
        success = run_game(q_table, True, False)
        success_count = 0
    else:
        success = run_game(q_table, False, True)
        
    if success:
        success_count += 1

    # Move epsilon towards its ending value, if it still needs to move
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_change

print(success)


# In[6]:


run_game(q_table, True, False)


# In[7]:


import pandas as pd

df = pd.DataFrame(q_table.argmax(axis=2))


# In[8]:


df.columns = [f'v-{x}' for x in range(DISCRETE_GRID_SIZE[0])]
df.index = [f'p-{x}' for x in range(DISCRETE_GRID_SIZE[1])]
df


# In[11]:


np.argmax(q_table[(2,0)])


# In[ ]:




