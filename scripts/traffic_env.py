import cityflow
import gym
from gym import spaces
import numpy as np
import json

class TrafficSignalEnv(gym.Env):
    def __init__(self, num_intersections, max_steps, config_file="../configs/basic/basic_config.json", road_network_file="../configs/basic/basic_road_network.json"):
        super().__init__()
        self.max_steps = max_steps
        self.config_file = config_file
        self.road_network_file = road_network_file
        self.engine = cityflow.Engine(config_file, thread_num=1)

        self.intersections, self.action_space, self.roads = self._parse_network(road_network_file)
        self.num_intersections = len(self.intersections)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_intersections, 2), dtype=np.float32)
    
    def reset(self):
        self.engine.reset()
        return self._get_observations()
    
    def step(self, actions):
        self._apply_actions(actions)
        self.engine.next_step()
        observations = self._get_observations()
        rewards = self._get_rewards()
        done = self._is_done()

        return observations, rewards, done, {}
    
    def _parse_network(self, road_network_file):
        with open(road_network_file, "r") as f:
            data = json.load(f)
        num_phases = max(len(intersection["trafficLight"]["lightphases"]) for intersection in data["intersections"])
        lanes_by_road = {}
        for road in data["roads"]:
            lanes_by_road[road["id"]] = road["lanes"]
        return data["intersections"], spaces.Discrete(num_phases), lanes_by_road
    
    def _get_observations(self):
        states = []
        lane_vehicle_counts = self.engine.get_lane_vehicle_count()
        lane_waiting_times = self.engine.get_lane_waiting_time()
        for intersection in self.intersections:
            roads = intersection["roads"]
            queue_lengths = []
            waiting_times = []

            for road in roads:
                lanes = self.roads[road]
                queue_lengths.append(sum(lane_vehicle_counts.get(lane, 0) for lane in lanes))
                waiting_times.append(sum(lane_waiting_times.get(lane, 0) for lane in lanes))

            total_queue = sum(queue_lengths)
            total_waiting = sum(waiting_times)
            states.append([total_queue, total_waiting])
        
        return np.array(states)
    
    def _apply_actions(self, actions):
        intersection_ids = [intersection["id"] for intersection in self.intersections]
        for i, action in enumerate(actions):
            self.engine.set_traffic_light_phase(intersection_ids[i], action)
    
    def _get_rewards(self):
        rewards = []
        for intersection in self.intersections:
            waiting_time = sum(self.engine.get_lane_waiting_time(intersection["id"]).values())
            queue_length = sum(self.engine.get_lane_vehicle_count(intersection["id"]).values())
            throughput = sum(self.engine.get_lane_vehicle_count(intersection["id"]).values())
            
            reward = -0.5 * waiting_time - 0.5 * queue_length + 1.0 * throughput
            rewards.append(reward)
        return np.array(rewards)
    
    def _is_done(self):
        return self.engine.get_current_time() >= self.max_steps

if __name__ == "__main__":
    config_file = "../configs/basic/basic_config.json"
    road_network_file = "../configs/basic/basic_road_network.json"
    max_steps = 100
    environment = TrafficSignalEnv(max_steps, config_file, road_network_file)

    observations = environment.reset()
    done = False

    while not done:
        actions = environment.action_space.sample()
        observations, rewards, done, info = environment.step(actions)
        print(f"Actions: {actions}, Observations: {observations}, Rewards: {rewards}")
