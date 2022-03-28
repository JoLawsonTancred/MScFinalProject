#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from gym import spaces
import random
import pandas as pd
import numpy as np


# In[2]:


class HappyPlace2(gym.Env):
  """
  Custom OpenAI Gym environment simulating a user's meme selections in Happy Place game.
  Created to compare different RL algorithms on training data.
  """

  metadata = {'render.modes': ['console']}

  def __init__(self, df):
    super(HappyPlace2, self).__init__()
    self.df = df
    self.reward_range = (0,1)
    # Define action and observation space
    self.action_space = spaces.MultiDiscrete([6, 3, 35, 35, 5, 2, 6, 3, 35, 35, 5, 2, 6, 3, 35, 35, 5, 2,
                                              6, 3, 35, 35, 5, 2, 6, 3, 35, 35, 5, 2, 6, 3, 35, 35, 5, 2, 
                                              6, 3, 35, 35, 5, 2, 6, 3, 35, 35, 5, 2, 6, 3, 35, 35, 5, 2])
    self.observation_space = spaces.Box(low=np.array([0,0,-1,-1,-30,0,0,0,-1,-1,-30,0,0,0,-1,-1,-30,0,0,0,-1,-1,-30,0,
                                                      0,0,-1,-1,-30,0,0,0,-1,-1,-30,0,0,0,-1,-1,-30,0,0,0,-1,-1,-30,0,
                                                      0,0,-1,-1,-30,0]), 
                                        high=np.array([5,2,1,1,30,1,5,2,1,1,30,1,5,2,1,1,30,1,5,2,1,1,30,1,
                                                       5,2,1,1,30,1,5,2,1,1,30,1,5,2,1,1,30,1,5,2,1,1,30,1,
                                                       5,2,1,1,30,1]))
    

  def _next_obs(self):
    # Return the next observation outputted by environment
    for x in range(len(self.current_step)):
        self.current_step[x] += 54
    row = df.loc[self.current_episode]
    obs = row[self.current_step]
    obs = obs.to_numpy()
    return obs

  def _action(self, action):
    # Returns array resulting from taking that action
    action1 = list(action)
    ind = 2
    ind1 = 3
    ind2 = 4
    for x in range(9):
        if action1[ind] == 34: 
            action1[ind] = 0.333
        elif action1[ind] <= 16: 
            action1[ind] *= -0.05
            action1[ind] = round(action1[ind],2)
        elif action1[ind] > 16: 
            action1[ind] -= 17
            action1[ind] *= 0.05
            action1[ind] = round(action1[ind],2)
        ind += 6         
    for x in range(9):
        if action1[ind1] == 34: 
            action1[ind1] = 0.333
        elif action1[ind1] <= 16:
            action1[ind1] *= -0.05
            action1[ind1] = round(action1[ind1],2)
        elif action1[ind1] > 16: 
            action1[ind1] -= 17
            action1[ind1] *= 0.05 
            action1[ind1] = round(action1[ind1],2)
        ind1 += 6 
    for x in range(9):
        if action1[ind2] <= 2:
            action1[ind2] *= -15
            action1[ind2] = float(action1[ind2])
        else:
            action1[ind2] -= 2
            action1[ind2] *= 15
            action1[ind2] = float(action1[ind2])
        ind2 += 6 
    return action1

  def _calculate_reward(self, action, obs):
    # Calculates reward by comparing the action to the next obs
    reward = 0
    for x in range(54):
        if action[x] != 0:
            if action[x] == obs[x]:
                reward += 1
        elif action[x] == 0:
            if action[x] == obs[x]:
                reward += 0.2
    return reward
    
  def step(self, action):
    # Execute one time step within the environment
    obs = self._next_obs()
    action = self._action(action)
    # Calculate reward according to proximity to selected meme
    reward = self._calculate_reward(action, obs)
    # Episode complete after Gen 20
    done = self.current_step[53] == 1134
    # No additional info required
    info = {}
    return obs, reward, done, info

  def reset(self):
    # Reset the state of the environment to an initial state
    # Pick random row from df to start on random episode
    r = len(self.df)
    self.current_episode = random.randint(0,r-1)
    row = df.loc[self.current_episode]
    # Always start from first available time step
    self.current_step = list(range(1, 54+1))
    obs = row[self.current_step]
    obs = obs.to_numpy()
    return obs

  def render(self, mode = 'console', close=False):
    pass


# In[3]:


# Import Stable Baselines library and relevant packages
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import stable_baselines
from stable_baselines import A2C
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, MlpLstmPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.results_plotter import plot_results

df = pd.read_csv('synthetic_data.csv')
env = HappyPlace2(df)

# Check custom environment comptabile with Stable Baselines
check_env(env, warn=True)

obs_space = env.observation_space.shape[0]
action_space = env.action_space


# In[12]:


#Implement a random policy to check reward function
from statistics import mean

totals = []
for episode in range(1000):
    episode_rewards = 0
    obs = env.reset()
    done = False
    while done == False:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
    totals.append(episode_rewards)
  
#print(totals)
print("Mean reward: ", mean(totals))


# In[42]:


# A2C with default MLP policy (2 hidden layers of 64) 
# (This is an example of separate actor and critic architectures)

# Create log directory and wrap environment with Monitor
log_dir = "tmp/gym/A2C/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

#Create the callback that saves best model at every 1000 timesteps
eval_env = HappyPlace2(df)
callback = EvalCallback(eval_env, best_model_save_path='./logs/A2C',
                       eval_freq=1000, render=False)

# Train model using basic default MLP policy
model = A2C(MlpPolicy, env, gamma=0, verbose=0, seed=93)
time_steps = 25000
model.learn(total_timesteps = time_steps, callback=callback)

#Plot results
plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "Happy Place 2 with default A2C and MLP policy (default gamma of 0.99)")
plt.show()


# In[16]:


# Load best saved model and evaluate it

l_model = A2C.load('./logs/A2C/best_model.zip')
mean_reward, std_reward = evaluate_policy(l_model, env, n_eval_episodes=25)
print('Mean reward: ', mean_reward)
print('Standard deviation: ', std_reward)


# In[4]:


# A2C with custom policy - actor and critic have separate network architectures
        
# Create log directory and wrap environment with Monitor
log_dir = "tmp/gym/A2Cseparate/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

#Create the callback that saves best model at every 1000 timesteps
eval_env = HappyPlace2(df)
callback = EvalCallback(eval_env, best_model_save_path='./logs/A2Cseparate',
                       eval_freq=1000, render=False)

# Build custom network architectures
class A2Cseparate(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(A2Cseparate, self).__init__(*args, **kwargs, 
        net_arch=[dict(pi=[128, 128, 64],
        vf=[128, 128, 64])],
        feature_extraction="mlp")

# Build and train model 
model = A2C(A2Cseparate, env, gamma=0, verbose=0, seed=93)
time_steps = 25000
model.learn(total_timesteps=time_steps, callback=callback)

#Plot results
plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "Happy Place 2 with A2C and custom separate networks")
plt.show()


# In[5]:


# Load best saved model and evaluate it

l_model = A2C.load('./logs/A2Cseparate/best_model.zip')
mean_reward, std_reward = evaluate_policy(l_model, env, n_eval_episodes=25)
print('Mean reward: ', mean_reward)
print('Standard deviation: ', std_reward)


# In[35]:


# A2C with custom policy - actor and critic share network architecture
        
#Create log directory and wrap environment with Monitor
log_dir = "tmp/gym/A2Cshared/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

#Create the callback that saves best model at every 1000 timesteps
eval_env = HappyPlace2(df)
callback = EvalCallback(eval_env, best_model_save_path='./logs/A2Cshared',
                       eval_freq=1000, render=False)

# Build custom shared network architecture 
class A2Cshared(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(A2Cshared, self).__init__(*args, **kwargs,
        net_arch=[128, 128, 128],
        feature_extraction="mlp")

# Build and train model 
model = A2C(A2Cshared, env, gamma=0, verbose=0, seed=93)
time_steps = 25000
model.learn(total_timesteps=time_steps, callback=callback)

#Plot results
plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "Happy Place 2 with default A2C and shared MLP network")
plt.show()


# In[36]:


# Load best saved model and evaluate it

l_model = A2C.load('./logs/A2Cshared/best_model.zip')
mean_reward, std_reward = evaluate_policy(l_model, env, n_eval_episodes=25)
print('Mean reward: ', mean_reward)
print('Standard deviation: ', std_reward)


# In[4]:


# A2C with custom policy - actor and critics' network architecture diverges
        
# Create log directory and wrap environment with Monitor
log_dir = "tmp/gym/A2Cdiverge50000/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Create the callback that saves best model at every 2000 timesteps
eval_env = HappyPlace2(df)
callback = EvalCallback(eval_env, best_model_save_path='./logs/A2Cdiverge50000',
                       eval_freq=2000, render=False)

# Build custom diverging network architecture
class A2Cdiverge(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(A2Cdiverge, self).__init__(*args, **kwargs,
        net_arch= [128, dict(vf=[128], pi=[128])],
        feature_extraction="mlp")

# Build and train model 
model = A2C(A2Cdiverge, env, gamma=0, verbose=0, seed=93)
time_steps = 50000
model.learn(total_timesteps=time_steps, callback=callback)

#Plot results
plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "")
plt.show()


# In[5]:


# Load best saved model and evaluate it

l_model = A2C.load('./logs/A2Cdiverge50000/best_model.zip')
mean_reward, std_reward = evaluate_policy(l_model, env, n_eval_episodes=25)
print('Mean reward: ', mean_reward)
print('Standard deviation: ', std_reward)


# In[31]:


# A2C with recurrent policy

# Parallel environments
env = DummyVecEnv([lambda: env])

# MONITOR WRAPPER DOES NOT WORK WITH VECTORISED ENVIRONMENTS
# Create log directory and wrap environment with Monitor
#log_dir = "tmp/gym/A2Crecurrent/"
#os.makedirs(log_dir, exist_ok=True)
#env = Monitor(env, log_dir)

# Create the callback that saves best model at every 1000 timesteps
eval_env = HappyPlace2(df)
callback = EvalCallback(eval_env, best_model_save_path='./logs/A2Crecurrent',
                       eval_freq=1000, render=False)

# Set custom policy hyperparameters 
#policy_kwargs = dict()

# Build and train model 
model = A2C(MlpLstmPolicy, env, gamma=0, verbose=0, seed=93)
time_steps = 25000
model.learn(total_timesteps=time_steps, callback=callback)

# MONITOR WRAPPER DOES NOT WORK WITH VECTORISED ENVIRONMENTS
#Plot results
#plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "Happy Place 2 with default A2C and recurrent LSTM policy")
#plt.show()


# In[30]:


# Load best saved model and evaluate it

l_model = A2C.load('./logs/A2Crecurrent/best_model.zip')
mean_reward, std_reward = evaluate_policy(l_model, env, n_eval_episodes=25)
print('Mean reward: ', mean_reward)
print('Standard deviation: ', std_reward)

