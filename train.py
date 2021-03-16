from unityagents import UnityEnvironment
import numpy as np
from agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                       # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start                   # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment and set to training mode
        state = env_info.vector_observations[0]           # get the current state
        score = 0                                         # reset score
        for _ in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(np.int32(action))[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # save model parameters if score reaches threshold
        if np.mean(scores_window)>=13.0:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])

agent = Agent(state_size, action_size, seed=0)

scores = dqn()
env.close()

# plot the scores
fig = plt.figure()
fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()