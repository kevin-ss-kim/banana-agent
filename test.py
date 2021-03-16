from unityagents import UnityEnvironment
import torch
from agent import Agent
import matplotlib.pyplot as plt

def test(n_episodes=100, max_t=1000):
    """Performs a test on an existing trained model.
    
    Params
    ======
        n_episodes (int): number of testing episodes
        max_t (int): maximum number of timesteps per episode
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    agent = Agent(state_size, action_size, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for _ in range(n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        for _ in range(max_t):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            if done:
                break


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
test()
env.close()