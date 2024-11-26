import cityflow
import gymnasium
from gymnasium import spaces
import numpy as np
import json
import os
from collections import defaultdict

class TrafficSignalEnv(gymnasium.Env):
    def __init__(self, max_steps, config_file="../configs/basic/basic_config.json", road_network_file="../configs/basic/basic_road_network.json", centralized=False):
        super().__init__()
        self.max_steps = max_steps
        self.centralized = centralized
        self.config_file = os.path.abspath(config_file)
        self.road_network_file = os.path.abspath(road_network_file)
        self.engine = cityflow.Engine(config_file, thread_num=1)

        self.intersections, self.roads, self.action_space = self._parse_network(road_network_file)
        self.num_intersections = len(self.intersections)
        self.num_roads = len(self.roads)
        self.num_lanes = sum([len(self.roads[road]) for road in self.roads])
        self.max_lanes = max([sum(len(self.roads[road]) for road in intersection["roads"]) * 2 for intersection in self.intersections])

        self.max_observed_waiting_count = 0
        self.max_observed_vehicle_count = 0
        self.max_observed_throughput = 0

        if self.centralized:
            self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_intersections * self.max_lanes,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_intersections, self.max_lanes), dtype=np.float32)
    
    def reset(self, seed=None):
        self.engine.reset()
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return self._get_observations(), {}
    
    def step(self, actions):
        self._apply_actions(actions)
        self.engine.next_step()
        observations = self._get_observations()
        rewards = np.sum(self._get_rewards())
        truncated = self._is_done()

        return observations, rewards, False, truncated, {}
    
    def _parse_network(self, road_network_file):
        with open(road_network_file, "r") as f:
            data = json.load(f)
        intersections = data["intersections"]
        num_phases = max(len(intersection["trafficLight"]["lightphases"]) for intersection in intersections)
        lanes_by_road = defaultdict(list)
        for road in data["roads"]:
            base_id = road["id"]
            for suffix in range(3):  # Suffixes _0, _1, _2
                lanes_by_road[road["id"]].append(f"{base_id}_{suffix}")
        return intersections, lanes_by_road, spaces.Discrete(num_phases)
    
    def _get_observations(self):
        states = []
        # Currently retrieves vehicle count and waiting vehicle count, can add waiting time and other potential features later (more complicated)
        for intersection in self.intersections:
            incoming_roads = intersection["roads"]
            vehicle_counts = []
            waiting_counts = []

            for road in incoming_roads:
                lanes = self.roads[road]
                vehicle_counts.extend([self.engine.get_lane_vehicle_count().get(lane, 0) for lane in lanes])
                waiting_counts.extend([self.engine.get_lane_waiting_vehicle_count().get(lane, 0) for lane in lanes])

            state = vehicle_counts + waiting_counts
            states.append(state)
        
        # Return global state (combine all intersection observations) if centralized, else return local states (one observation per intersection)
        if self.centralized:
            # Flatten dynamically (padding is unnecessary for centralized training)
            return np.concatenate([np.array(state) for state in states])
        else:
            # Pad states for decentralized observation
            padded_states = [state + [0] * (self.max_lanes - len(state)) for state in states]
            return np.array(padded_states).astype(np.float32)
    
    def _apply_actions(self, actions):
        if isinstance(actions, (int, np.integer)):
            actions = [actions]
        
        intersection_ids = [intersection["id"] for intersection in self.intersections]
        for i, action in enumerate(actions):
            self.engine.set_tl_phase(intersection_ids[i], action)
    
    def _get_rewards(self):
        rewards = []
        lane_waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        throughput = (self.previous_vehicle_count - self.engine.get_vehicle_count()) / self.num_intersections

        # Pre-defined constants (useful for now, eventually need to implement dynamic normalization during runtime when switching to larger grid sizes)
        MAX_VEHICLE_COUNT = 87 * self.num_intersections
        MAX_WAITING_COUNT = 44 * self.num_lanes
        MAX_THROUGHPUT = 0.5 * self.num_intersections

        for intersection in self.intersections:
            incoming_roads = intersection["roads"]
            total_vehicle_count = sum(self.engine.get_lane_vehicle_count().get(lane, 0) for road in incoming_roads for lane in self.roads[road])
            total_waiting_count = sum(lane_waiting_counts.get(lane, 0) for road in incoming_roads for lane in self.roads[road])
            normalized_waiting_count = round(total_waiting_count / MAX_WAITING_COUNT, 3)
            normalized_vehicle_count = round(total_vehicle_count / MAX_VEHICLE_COUNT, 3)
            normalized_throughput = round(throughput / MAX_THROUGHPUT, 3)
            reward = -0.5 * normalized_waiting_count - 0.5 * normalized_vehicle_count + 1.0 * normalized_throughput
            rewards.append(reward)

            self.max_observed_waiting_count = max(self.max_observed_waiting_count, normalized_waiting_count)
            self.max_observed_vehicle_count = max(self.max_observed_vehicle_count, normalized_vehicle_count)
            self.max_observed_throughput = max(self.max_observed_throughput, normalized_throughput)
        
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
        # Currently just choosing random actions since MAPPO not implemented yet
        actions = [environment.action_space.sample() for _ in range(environment.num_intersections)]
        observations, rewards, terminal, done, info = environment.step(actions)

        print(f"Rewards: {rewards}")
