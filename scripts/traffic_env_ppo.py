from traffic_env import TrafficSignalEnv
from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Check the custom environment for compatibility issues
base_env = TrafficSignalEnv(max_steps=1000, centralized=False)  # Adjust max_steps or centralized based on preference
env = Monitor(base_env)
# check_env(env)

model = PPO(
    policy='MlpPolicy',
    env=env,
    n_steps=2048,  # Number of steps to run for each update
    batch_size=64,  # Size of minibatches for SGD
    gae_lambda=0.95,  # Generalized Advantage Estimation parameter
    gamma=0.99,  # Discount factor
    clip_range=0.2,  # PPO clipping range
    ent_coef=0.01,  # Entropy coefficient to encourage exploration
    vf_coef=0.5,  # Value function coefficient
    max_grad_norm=0.5,  # Max norm for gradient clipping
    verbose=1  # Show training progress
)
model.learn(total_timesteps=100000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")