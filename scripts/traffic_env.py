import cityflow
import gym
from gym import spaces
import numpy as np
import json
import os
from collections import defaultdict

class TrafficSignalEnv(gym.Env):
    def __init__(self, max_steps, config_file="../configs/basic/basic_config.json", road_network_file="../configs/basic/basic_road_network.json", centralized=False):
        super().__init__()
        self.max_steps = max_steps
        self.centralized = centralized
        self.config_file = os.path.abspath(config_file)
        self.road_network_file = os.path.abspath(road_network_file)
        self.engine = cityflow.Engine(config_file, thread_num=1)

        self.intersections, self.roads, self.action_space = self._parse_network(road_network_file)
        self.num_intersections = len(self.intersections)

        num_features = len(self.roads[next(iter(self.roads))]) * 2
        if self.centralized:
            self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_intersections * num_features,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_intersections, num_features), dtype=np.float32)
    
    def reset(self):
        self.engine.reset()
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return self._get_observations()
    
    def step(self, actions):
        self._apply_actions(actions)
        self.engine.next_step()
        observations = self._get_observations(self.centralized)
        rewards = self._get_rewards()
        done = self._is_done()
        return observations, rewards, done, {}
    
    def _parse_network(self, road_network_file):
        with open(road_network_file, "r") as f:
            data = json.load(f)
        num_phases = max(len(intersection["trafficLight"]["lightphases"]) for intersection in data["intersections"])
        lanes_by_road = defaultdict(list)
        for road in data["roads"]:
            base_id = road["id"]
            for suffix in range(3):  # Suffixes _0, _1, _2
                lanes_by_road[road["id"]].append(f"{base_id}_{suffix}")
        return data["intersections"], lanes_by_road, spaces.Discrete(num_phases)
    
    def _get_observations(self):
        states = []
        # Currently retrieves vehicle count and waiting vehicle count, can add waiting time and other potential features later (more complicated)
        for intersection in self.intersections:
            roads = intersection["roads"]
            vehicle_counts = []
            waiting_vehicle_counts = []

            for road in roads:
                lanes = self.roads[road]
                vehicle_counts.extend([self.engine.get_lane_vehicle_count().get(lane, 0) for lane in lanes])
                waiting_vehicle_counts.extend([self.engine.get_lane_waiting_vehicle_count().get(lane, 0) for lane in lanes])

            state = vehicle_counts + waiting_vehicle_counts
            states.append(state)
        
        # Return global state (combine all intersection observations) if centralized, else return local states (one observation per intersection)
        return np.array(states).flatten() if self.centralized else np.array(states)
    
    def _apply_actions(self, actions):
        intersection_ids = [intersection["id"] for intersection in self.intersections]
        for i, action in enumerate(actions):
            self.engine.set_tl_phase(intersection_ids[i], action)
    
    def _get_rewards(self):
        rewards = []
        lane_waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        throughput = (self.previous_vehicle_count - self.engine.get_vehicle_count()) / self.num_intersections

        for intersection in self.intersections:
            roads = intersection["roads"]
            total_waiting_time = sum(lane_waiting_counts.get(lane, 0) for road in roads for lane in self.roads[road])
            total_vehicle_count = sum(self.engine.get_lane_vehicle_count().get(lane, 0) for road in roads for lane in self.roads[road])
            reward = -0.5 * total_waiting_time - 0.5 * total_vehicle_count + 1.0 * throughput
            rewards.append(reward)
        
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return np.array(rewards)
    
    def _is_done(self):
        return self.engine.get_current_time() >= self.max_steps

if __name__ == "__main__":
    config_file = "../configs/basic/basic_config.json"
    road_network_file = "../configs/basic/basic_road_network.json"
    max_steps = 100
    environment = TrafficSignalEnv(max_steps, config_file, road_network_file, False)

    observations = environment.reset()
    done = False

    while not done:
        actions = [environment.action_space.sample() for _ in range(environment.num_intersections)]
        observations, rewards, done, info = environment.step(actions)
        print(f"Actions: {actions}, Observations: {observations}, Rewards: {rewards}")
