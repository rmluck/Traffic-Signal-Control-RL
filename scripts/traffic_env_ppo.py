from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig
from ray.tune.registry import register_env
from traffic_env import TrafficSignalEnv
import warnings
import os

"""
NEXT STEPS:
1. Run training loop (at bottom of this file), print out different parts of 'result' to make sure rewards and other things are reasonable
    - Make sure observations, rewards are in correct format
    - Make sure rewards are not extreme values/NaN
2. Use ppo.evaluate() or create new evaluation loop to simulate a few episodes with trained model
    - Consider metrics like total reward over episodes, average waiting time per vehicle, throughput, congestion, etc.
    - Can compare results against basic rule-based system to see if rewards are improving
3. Tune hyperparameters
    - Adjust training parameters like gamma, train_batch_size, minibatch_size, num_epochs, grad_clip, clip_param
    - Add entropy_coeff parameter to encourage exploration early in training
    - Add num_sgd_iter parameter (number of gradient updates per iteration)
    - Experiment with lower or higher learning rates to observe their effect on stability
    - Experiment w/ shared policies (all agents share single policy) vs. independent policies (unique policy for each agent)
    - Can use grid search or Ray Tune for automated hyperparameter tuning
        from ray.tune import run
        run(PPO, config={"env": "TrafficSignalEnv", "gamma": tune.grid_search([0.95, 0.99]), "lambda": tune.grid_search([0.9, 0.95]),}, stop={"training_iteration": 100},)
4. Monitor training behavior
    - Detect issues like slow convergence or unstable training
    - Plot rewards over time to see if algorithm is learning
    - Save and analyze traffic metrics during training
    - Use RLib's built-in tensorboard logging:
        tensorboard --logdir ~/ray_results
    - Is reward improving over time? Are vehicles clearing intersections effectively as training progresses?
5. Scale up problem
    - Test on larger grids w/ more intersections, vehicles, and complex traffic flows
    - Increase grid size using generate_grid_scenario.py
    - Increase number of agents (intersections) and ensure centralized critic handles larger state space
    - Does training process scale efficiently?
    - Does policy learned on small grid generalize to larger grids?
    - Can add waiting time feature for vehicles to observation space
    - Implement dynamic normalization for features during runtime for larger grid sizes instead of pre-defined constants
6. Add realistic scenarios
    - Test under different traffic scenarios
    - Vary traffic density (simulate peak and off-peak traffic conditions)
    - Test edge cases (high vehicle inflow, blocked roads, traffic jams)
7. Fine-tune reward function
    - Ensure reward function minimizes congestion, maximizes throughput
    - Monitor components of reward function (normalized metrics)
    - Adjust coefficients for better balance
    - Add additional penalties or bonuses to steer policy more effectively
8. Validate system
    - Compare performance with baseline methods (fixed signal durations, rule-based strategies, random signal switching)
9. Save and load models
    - Save trained policy for future use:
        ppo.save(path_to_model)
    - Load model and test in new script
10. Document results
    - Log findings from baseline comparisons, hyperparameter tuning, real-world scenario tests
    - Include visualizations (graphs of rewards, traffic replays, etc.) to showcase improvements
"""

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["RLLIB_DISABLE_NEW_API_STACK"] = "True"

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

# Correct way to initialize PPOConfig and apply configurations
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


if __name__ == "__main__":
    def trainable(config):
        # Create PPO configuration
        ppo_config = (
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
        algo = ppo_config.build()
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
        "env_config": { "max_steps": 1000},  # Example environment config
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
            num_samples=10,  # Number of configurations to sample
            scheduler=scheduler,
        ),
        run_config=RunConfig(),
    )

    # Run tuning
    results = tuner.fit()

    # Create and train PPO algorithm
    # ppo = PPO(config=ppo_config)

    # # Training loop
    # with open("training_log.txt", "w") as log_file:
    #     for i in range(50):  # Number of training iterations
    #         result = ppo.train()

    #         log_file.write(f"\nIteration {i}\n")
    #         # print(f"\nIteration {i}")

    #         # print(f"\tTraining Info:")
    #         # print(f"\t\tTime This Iteration: " + str(result["time_this_iter_s"]) + "s")
    #         # print(f"\t\tTotal Time: " + str(result["time_total_s"]) + "s")
            
    #         # print(f"\tAgent Metrics:")
    #         # for i in range(len(result["info"]["learner"])):
    #         #     print(f"\t\tAgent {i}")
    #         #     info_learner = result["info"]["learner"][f"agent_{i}"]["learner_stats"]
    #             # print(f"\t\t\tTotal Loss: " + str(info_learner["total_loss"]) + ", " + "Policy Loss: " + str(info_learner["policy_loss"]) + ", " + "VF Loss: " + str(info_learner["vf_loss"]) + ", " + "Entropy: " + str(info_learner["entropy"]) + ", " + "KL: " + str(info_learner["kl"]) + ", " + "Current Learning Rate: " + str(info_learner["cur_lr"]) + ", " + "VF Explained by Value Function: " + str(info_learner["vf_explained_var"]))
    #             # print(f"\t\t\tTotal Loss: " + str(info_learner["total_loss"]) + " (how well agent is learning overall)")
    #             # print(f"\t\t\tPolicy Loss: " + str(info_learner["policy_loss"]) + " (indicates agent's optimization on policy)")
    #             # print(f"\t\t\tVF Loss: " + str(info_learner["vf_loss"]) + " (optimization of value function)")
    #             # print(f"\t\t\tEntropy: " + str(info_learner["entropy"]) + " (insight into exploration, higher entropy means more exploration)")
    #             # print(f"\t\t\tKL: " + str(info_learner["kl"]) + " (measures divergence between current policy and previous policies, useful for stability monitoring)")
    #             # print(f"\t\t\tCurrent Learning Rate: " + str(info_learner["cur_lr"]) + " (current learning rate, helps identify if learning rate schedules are being applied correctly)")
    #             # print(f"\t\t\tVF Explained by Value Function: " + str(info_learner["vf_explained_var"]) + " (variance explained by value function, assessing model performance)")

    #         env_runners = result["env_runners"]
    #         # print(f"\tEnvironment Metrics:")
    #         # print(f"\t\tMean Reward: " + str(env_runners["episode_reward_mean"]) + ", " + "Mean Episode Length: " + str(env_runners["episode_len_mean"]))
    #         log_file.write(f"\tMean Reward: {env_runners['episode_reward_mean']} (average reward per episode, reflecting overall performance)\n")
    #         # print(f"\tMean Reward: " + str(env_runners["episode_reward_mean"]) + " (average reward per episode, reflecting overall performance)")
    #         # print(f"\t\tMean Episode Length: " + str(env_runners["episode_len_mean"]) + " (average length of episodes, indicating task difficulty or agent efficiency)")
            
    #         # print(f"\tSystem Metrics:")
    #         # print(f"\t\tLearn Time: " + str(result["timers"]["learn_time_ms"]) + "ms, " + "Steps Sampled: " + str(result["num_env_steps_sampled"]) + ", " + "Throughput: " + str(result["timers"]["learn_throughput"]) + " steps/s")
    #         # print(f"\t\tLearn Time: " + str(result["timers"]["learn_time_ms"]) + "ms (time spent in learning phase per iteration)")
    #         # print(f"\t\tSteps Sampled: " + str(result["num_env_steps_sampled"]) + " (progress of sampling in environment)")
    #         # print(f"\t\tThroughput: " + str(result["timers"]["learn_throughput"]) + " steps/s (measures training throughput, reflecting efficiency)")

    #         # print(f"Iteration {i}: Mean Reward = {result['episode_reward_mean']}")

    # # Save trained model
    # ppo.save("mappo_traffic_model")



    #####################################################################################


    # from traffic_env import TrafficSignalEnv
    # from stable_baselines3 import PPO
    # # from stable_baselines3.common.env_checker import check_env
    # from stable_baselines3.common.monitor import Monitor
    # from stable_baselines3.common.evaluation import evaluate_policy

    # # Check the custom environment for compatibility issues
    # base_env = TrafficSignalEnv(max_steps=1000, centralized=False)  # Adjust max_steps or centralized based on preference
    # env = Monitor(base_env)
    # # check_env(env)

    # model = PPO(
    #     policy='MlpPolicy',
    #     env=env,
    #     n_steps=2048,  # Number of steps to run for each update
    #     batch_size=64,  # Size of minibatches for SGD
    #     gae_lambda=0.95,  # Generalized Advantage Estimation parameter
    #     gamma=0.99,  # Discount factor
    #     clip_range=0.2,  # PPO clipping range
    #     ent_coef=0.01,  # Entropy coefficient to encourage exploration
    #     vf_coef=0.5,  # Value function coefficient
    #     max_grad_norm=0.5,  # Max norm for gradient clipping
    #     verbose=1  # Show training progress
    # )
    # model.learn(total_timesteps=100000)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # print(f"Mean reward: {mean_reward} +/- {std_reward}")