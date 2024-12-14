import cityflow
# import gymnasium
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import json
import os
import csv
from collections import defaultdict
from statistics import mean

class TrafficSignalEnv(MultiAgentEnv):
    def __init__(self, max_steps, config_file="../configs/training/training_config.json", road_network_file="../configs/training/training_road_network.json"):
        super().__init__()
        self.max_steps = max_steps
        self.config_file = os.path.abspath(config_file)
        self.road_network_file = os.path.abspath(road_network_file)
        self.engine = cityflow.Engine(self.config_file, thread_num=1)

        self.intersections, self.roads = self._parse_network(self.road_network_file)
        self.num_intersections = len(self.intersections)
        self.num_roads = len(self.roads)
        self.num_lanes = sum([len(self.roads[road]) for road in self.roads])
        self.max_lanes = max([sum(len(self.roads[road]) for road in intersection["roads"]) * 2 for intersection in self.intersections])

        self.previous_vehicle_count = self.engine.get_vehicle_count()

        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=0, high=np.inf, shape=(self.max_lanes,), dtype=np.float32) for i in range(self.num_intersections)
        })
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(len(self.intersections[i]["trafficLight"]["lightphases"])) for i in range(self.num_intersections)
        })

        self.agents = [f"agent_{i}" for i in range(self.num_intersections)]

    def reset(self, *, seed=None, options=None):
        self.engine.reset()
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        observations = {
            f"agent_{i}": self._get_observation_for_agent(i) for i in range(self.num_intersections)
        }
        return observations, {}
    
    def step(self, actions):
        for agent_id, action in actions.items():
            intersection_id = int(agent_id.split("_")[1])
            self.engine.set_tl_phase(self.intersections[intersection_id]["id"], action)
        self.engine.next_step()
        observations = {
            f"agent_{i}": self._get_observation_for_agent(i) for i in range(self.num_intersections)
        }
        rewards = {
            f"agent_{i}": reward for i, reward in enumerate(self._get_rewards())
        }
        terminateds = {
            f"agent_{i}": self._is_done() for i in range(self.num_intersections)
        }
        terminateds["__all__"] = self._is_done()
        truncateds = {
            f"agent_{i}": False for i in range(self.num_intersections)
        }
        truncateds["__all__"] = self._is_done()
        infos = {}

        return observations, rewards, terminateds, truncateds, infos
    
    def _parse_network(self, road_network_file):
        with open(road_network_file, "r") as f:
            data = json.load(f)
        intersections = data["intersections"]
        lanes_by_road = defaultdict(list)
        for road in data["roads"]:
            base_id = road["id"]
            for suffix in range(3):  # Suffixes _0, _1, _2
                lanes_by_road[road["id"]].append(f"{base_id}_{suffix}")
        return intersections, lanes_by_road
    
    def _get_observation_for_agent(self, agent_idx):
        # Currently retrieves vehicle count and waiting vehicle count, can add waiting time and other potential features later (more complicated)
        intersection = self.intersections[agent_idx]
        incoming_roads = intersection["roads"]
        vehicle_counts = [
            self.engine.get_lane_vehicle_count().get(lane, 0)
            for road in incoming_roads
            for lane in self.roads[road]
        ]
        waiting_counts = [
            self.engine.get_lane_waiting_vehicle_count().get(lane, 0)
            for road in incoming_roads
            for lane in self.roads[road]
        ]
        state = vehicle_counts + waiting_counts
        padded_state = state + [0] * (self.max_lanes - len(state))
        return np.array(padded_state, dtype=np.float32)
    
    def _get_rewards(self):
        rewards = []
        lane_waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        throughput = max(0,(self.previous_vehicle_count - self.engine.get_vehicle_count()) / self.num_intersections)

        # Pre-defined constants (useful for now, eventually need to implement dynamic normalization during runtime when switching to larger grid sizes)
        MAX_VEHICLE_COUNT = 87 * self.num_intersections
        MAX_WAITING_COUNT = 44 * self.num_lanes
        MAX_THROUGHPUT = 0.5 * self.num_intersections

        for intersection in self.intersections:
            incoming_roads = intersection["roads"]
            total_vehicle_count = sum(self.engine.get_lane_vehicle_count().get(lane, 0) for road in incoming_roads for lane in self.roads[road])
            total_waiting_count = sum(lane_waiting_counts.get(lane, 0) for road in incoming_roads for lane in self.roads[road])
            normalized_waiting_count = total_waiting_count / MAX_WAITING_COUNT
            normalized_vehicle_count = total_vehicle_count / MAX_VEHICLE_COUNT
            normalized_throughput = throughput / MAX_THROUGHPUT
            reward = 0.1 - 0.5 * normalized_waiting_count - 0.5 * normalized_vehicle_count + 5.0 * normalized_throughput
            rewards.append(reward)
        
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return rewards
    
    def _is_done(self):
        return self.engine.get_current_time() >= self.max_steps

# Baseline: fixed signal durations
def fixed_signal_control(env):
    env.reset()
    total_reward = 0
    done = False
    step = 0  # Initialize a step counter
    with open("../data/fixed_rewards.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Total Reward"])
        while not done:
            # Cycle through fixed phases based on the step counter
            actions = {f"agent_{i}": step % env.action_space[f"agent_{i}"].n for i in range(env.num_intersections)}
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += sum(reward.values())
            writer.writerow([step, total_reward])
            done = terminated["__all__"]
            step += 1  # Increment the step counter
    return total_reward

if __name__ == "__main__":
    config_file = "../configs/training/training_config.json"
    road_network_file = "../configs/training/training_road_network.json"
    max_steps = 100
    env = TrafficSignalEnv(max_steps, config_file, road_network_file)

    # obs = env.reset()
    # done = {"__all__": False}
    # total_reward = 0
    # while not done["__all__"]:
    #     action = env.action_space.sample()  # Random action
    #     obs, reward, _, done, info = env.step(action)
    #     print(f"Step Reward: {reward}")
    #     print(f"{info}")
    #     # total_reward += reward
    # print(f"Total Reward for Random Actions: {total_reward}")

    fixed_rewards = fixed_signal_control(env)
    print(fixed_rewards)
