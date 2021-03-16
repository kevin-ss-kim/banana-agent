from unityagents import UnityEnvironment
import torch
from agent import Agent
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

def test(n_episodes=5, max_t=1000, file="saved_model_weights.pth"):
    """Performs tests on an already trained Deep Q-Learning agent.
    
    Params
    ======
        n_episodes (int): number of testing episodes
        max_t (int): maximum number of timesteps per episode
        file: saved model state to load data from
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    agent = Agent(state_size, action_size, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load(file))

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            action = agent.act(state)
            env_info = env.step(np.int32(action))[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    return scores


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
scores = test()
env.close()

# plot the scores
fig = plt.figure()
fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()