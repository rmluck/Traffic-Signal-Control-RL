from traffic_env import TrafficSignalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Check the custom environment for compatibility issues
env = TrafficSignalEnv(max_steps=1000, centralized=False)  # Adjust max_steps or centralized based on preference
check_env(env)
