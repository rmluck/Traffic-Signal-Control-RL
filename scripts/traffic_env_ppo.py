from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.air.config import RunConfig
from ray import tune
from ray.tune import grid_search
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env
from traffic_env import TrafficSignalEnv
import warnings
import os
import csv
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["RLLIB_DISABLE_NEW_API_STACK"] = "True"
training_logs = Path("../logs/training_logs").resolve().as_posix()
tuning_logs = Path("../logs/tuning_logs").resolve().as_posix()

# Define environment configuration
env_config = {
    "max_steps": 1000,
}

# Register custom environment
def env_creator(config):
    return TrafficSignalEnv(**config)

register_env("TrafficSignalEnv", env_creator)

# Create the multi-agent policies configuration
env = TrafficSignalEnv(**env_config)

# Define policy for each agent
policies = {
    f"agent_{i}": (None, env.observation_space[f"agent_{i}"], env.action_space[f"agent_{i}"], {}) for i in range(env.num_intersections)
}

# Map each agent to its respective policy
def policy_mapping_fn(agent_id, episode, **kwargs):
    return agent_id

ppo_config = (
    PPOConfig()
    .environment(env="TrafficSignalEnv", env_config=env_config)
    .framework("torch")
    .env_runners(num_env_runners=4, rollout_fragment_length="auto")
    .training(
        gamma=0.99, # Discount factor
        train_batch_size=2048, # Total batch size for training
        minibatch_size=64, # Minibatch size
        num_epochs=10, # Number of epochs to iterate over each batch
        grad_clip=0.5, # Gradient clipping value
        clip_param=0.2, # Clipping parameter for PPO
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
)

def train_PPO():
    # Create and train PPO algorithm
    ppo = PPO(config=ppo_config)

    # Training loop
    # with open("../logs/training_logs/training_log.txt", "w") as log_file:
    with open("../logs/training_logs/training_data.csv", "w", newline="") as data_file:
        writer = csv.writer(data_file)
        writer.writerow(["Training Iteration", "Mean Reward"])
        for i in range(50): # Number of training iterations
            print(f"Training {i}")
            result = ppo.train()

            # log_file.write(f"\nIteration {i}\n")

            # log_file.write(f"\tTraining Info:")
            # log_file.write(f"\t\tTime This Iteration: " + str(result["time_this_iter_s"]) + "s")
            # log_file.write(f"\t\tTotal Time: " + str(result["time_total_s"]) + "s")

            # log_file.write(f"\tAgent Metrics:")
            # for i in range(len(result["info"]["learner"])):
            #     log_file.write(f"\t\tAgent {i}")
            #     info_learner = result["info"]["learner"][f"agent_{i}"]["learner_stats"]
            #     log_file.write(f"\t\t\tTotal Loss: " + str(info_learner["total_loss"]) + " (how well agent is learning overall)")
            #     log_file.write(f"\t\t\tPolicy Loss: " + str(info_learner["policy_loss"]) + " (indicates agent's optimization on policy)")
            #     log_file.write(f"\t\t\tVF Loss: " + str(info_learner["vf_loss"]) + " (optimization of value function)")
            #     log_file.write(f"\t\t\tEntropy: " + str(info_learner["entropy"]) + " (insight into exploration, higher entropy means more exploration)")
            #     log_file.write(f"\t\t\tKL: " + str(info_learner["kl"]) + " (measures divergence between current policy and previous policies, useful for stability monitoring)")
            #     log_file.write(f"\t\t\tCurrent Learning Rate: " + str(info_learner["cur_lr"]) + " (current learning rate, helps identify if learning rate schedules are being applied correctly)")
            #     log_file.write(f"\t\t\tVF Explained by Value Function: " + str(info_learner["vf_explained_var"]) + " (variance explained by value function, assessing model performance)")

            env_runners = result["env_runners"]
            writer.writerow([i, env_runners["episode_reward_mean"]])
            # log_file.write(f"\tEnvironment Metrics:")
            # log_file.write(f"\tMean Reward: " + str(env_runners['episode_reward_mean']) + " (average reward per episode, reflecting overall performance)\n")
            # log_file.write(f"\t\tMean Reward: " + str(env_runners["episode_reward_mean"]) + " (average reward per episode, reflecting overall performance)")
            # log_file.write(f"\t\tMean Episode Length: " + str(env_runners["episode_len_mean"]) + " (average length of episodes, indicating task difficulty or agent efficiency)")

            # log_file.write(f"\tSystem Metrics:")
            # log_file.write(f"\t\tLearn Time: " + str(result["timers"]["learn_time_ms"]) + "ms (time spent in learning phase per iteration)")
            # log_file.write(f"\t\tSteps Sampled: " + str(result["num_env_steps_sampled"]) + " (progress of sampling in environment)")
            # log_file.write(f"\t\tThroughput: " + str(result["timers"]["learn_throughput"]) + " steps/s (measures training throughput, reflecting efficiency)")

    ppo.save("../models/ppo_trained_model")

def tune_PPO():
    def trainable(config):
        # Initialize PPOConfig and apply configurations
        ppo_tune_config = (
            PPOConfig()
            .environment(env="TrafficSignalEnv", env_config=config["env_config"])
            .framework("torch")
            .training(
                gamma=config["gamma"],
                train_batch_size=config["train_batch_size"],
                num_sgd_iter=config["num_sgd_iter"],
                clip_param=config["clip_param"],
                entropy_coeff=config["entropy_coeff"],
                lr=config["lr"],
            )
            .multi_agent(
                policies=config["policies"],
                policy_mapping_fn=config["policy_mapping_fn"],
            )
        )

        algo = ppo_tune_config.build()
        for _ in range(config["num_iterations"]):
            result = algo.train()
            tune.report(mean_reward=result["episode_reward_mean"])
        algo.stop()
    
    search_space = {
        "gamma": tune.grid_search([0.95, 0.99]),
        "clip_param": tune.grid_search([0.1, 0.2]),
        "entropy_coeff": tune.grid_search([0.005, 0.01, 0.02]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "train_batch_size": tune.choice([1024, 2048]),
        "num_sgd_iter": tune.choice([5, 10, 20]),
        "env_config": {"max_steps": 1000}, # Example environment config
        "policies": {
            f"agent_{i}": (None, env.observation_space[f"agent_{i}"], env.action_space[f"agent_{i}"], {}) for i in range(env.num_intersections)
        },
        "policy_mapping_fn": lambda agent_id: "policy_1",
        "num_iterations": 100,
    }

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=10, # Number of configurations to sample
            scheduler=scheduler,
        ),
        run_config=RunConfig(),
    )

    # Run tuning
    results = tuner.fit()

def tune_hyperparameters():
    tuner = tune.Tuner(
        "PPO",
        param_space={
            "env": "TrafficSignalEnv",
            "env_config": env_config,
            "framework": "torch",
            "gamma": tune.grid_search([0.95, 0.99]),
            "train_batch_size": tune.grid_search([1024, 2048]),
            "minibatch_size": tune.choice([128, 256, 512]),
            "num_sgd_iter": tune.grid_search([10, 20]),
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.grid_search([0.01, 0.05, 0.1]),
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "num_workers": 2,
        },
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=ASHAScheduler(
                max_t=25,
                grace_period=5,
                reduction_factor=2,
            ),
            num_samples=5, # Number of configurations to try
        ),
        run_config=RunConfig(
            name="TrafficSignal_PPO_Tuning",
            storage_path=str(tuning_logs),
            verbose=1,
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result()
    print("Best hyperparameters: ", best_result.config)
    print("Best mean reward: ", best_result.metrics["episode_reward_mean"])

def evaluate_policy():
    ppo = PPO(config=ppo_config)
    ppo.restore("../models/ppo_trained_model")

    env_config = {
        "max_steps": 1000,
        "config_file": "../configs/evaluation/evaluation_config.json"
    }

    # Register custom environment
    def env_creator(config):
        return TrafficSignalEnv(**config)

    register_env("TrafficSignalEnv", env_creator)

    # Create the multi-agent policies configuration
    env = TrafficSignalEnv(**env_config)
    obs, _  = env.reset()
    total_reward = 0
    done = False
    step = 0

    with open("../data/ppo_rewards.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Total Reward"])
        while not done:
            print(step)
            actions = {
                agent_id: ppo.compute_single_action(obs[agent_id], policy_id=agent_id) for agent_id in obs.keys()
            }
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += sum(reward.values())
            writer.writerow([step, total_reward])
            done = terminated["__all__"]
            step += 1
    
    return total_reward

if __name__ == "__main__":
    # train_PPO()
    # tune_PPO()
    # tune_hyperparameters()
    evaluate_policy()