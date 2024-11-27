from traffic_env import TrafficSignalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Check the custom environment for compatibility issues
env = TrafficSignalEnv(max_steps=1000, centralized=False)  # Adjust max_steps or centralized based on preference

check_env(env)

model = PPO(policy='MlpPolicy',env=env)

model.learn(total_timesteps=10000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")