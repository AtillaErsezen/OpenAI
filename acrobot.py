import gymnasium as gym
import torch.nn as nn
import torch
import numpy as np

temp_env = gym.make("Acrobot-v1", render_mode='rgb_array')

env = gym.wrappers.RecordVideo(env=temp_env, video_folder='content/video/acrobot', name_prefix="acrobot")

EPOCHS = 1000

env.start_video_recorder()

#need to call reset before calling step
observation, info = env.reset()

def forward(observation, info) -> int:
    '''
    Makes a forward pass in the neural network
    param: observation: observation received from env.reset() or env.step()
    param: info: information received from env.reset() or env.step()
    returns: action(int): action to be taken
    '''
    
    classnet = nn.Sequential(
        nn.Linear(6, 12),
        nn.Tanh(),
        nn.Linear(12, 6),
        nn.Tanh(),
        nn.Linear(6, 1)
    )
    action = classnet(torch.tensor(observation))
    return int(action)

rew = 0
for _ in range(EPOCHS):
    action = forward(observation, info)
    observation, reward, terminated, truncated, info = env.step(action)

    #increase reward in each step for learning
    rew += reward

    env.render()

    if terminated:
        observation, info = env.reset()

env.close_video_recorder()
env.close()