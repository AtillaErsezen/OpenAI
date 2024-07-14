import gymnasium as gym

env = gym.make("Acrobot-v1", render_mode='human')

EPOCHS = 1000

for _ in range(EPOCHS):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()