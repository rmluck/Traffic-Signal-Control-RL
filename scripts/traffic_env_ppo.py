import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from traffic_env import TrafficSignalEnv

class PPOAgent(nn.Module):
    def __init__(self, num_intersections, action_space):
        super(PPOAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_intersections * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(32, action_space.n)
        self.value_head = nn.Linear(32, 1)

        # Initialize learnable coefficients for reward components
        self.coef_waiting_time = nn.Parameter(torch.tensor(-0.5))
        self.coef_queue_length = nn.Parameter(torch.tensor(-0.5))
        self.coef_throughput = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.fc(x)
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        return action_logits, value

    def get_action(self, state):
        logits, _ = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        _, value = self(state)
        return value

def train(env, agent, num_episodes, gamma=0.99, lr=3e-4, clip_epsilon=0.2, update_interval=10):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        for t in range(env.max_steps):
            action, log_prob = agent.get_action(torch.tensor(state, dtype=torch.float32))
            value = agent.get_value(torch.tensor(state, dtype=torch.float32))

            # Take a step in the environment
            next_state, reward, done, _ = env.step([action])
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            state = next_state

            if done:
                break

        # Compute advantages and returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + gamma * G * (1 - done)
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantages = returns - values.detach()

        # PPO Policy Update
        for _ in range(update_interval):
            logits, value = agent(torch.tensor(state, dtype=torch.float32))
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(action)

            ratio = torch.exp(new_log_probs - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - value).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_rewards.append(sum(rewards))

    return all_rewards

if __name__ == "__main__":
    num_intersections = 4  # or however many intersections you want to model
    max_steps = 500  # max steps per episode
    num_episodes = 1000  # number of episodes to train
    learning_rate = 3e-4  # learning rate for optimizer

    # Initialize environment and agent
    env = TrafficSignalEnv(max_steps=max_steps)
    agent = PPOAgent(num_intersections=num_intersections, action_space=env.action_space)

    # Train the agent
    all_rewards = train(env, agent, num_episodes=num_episodes, gamma=0.99, lr=learning_rate)

    # Save the trained model
    torch.save(agent.state_dict(), "ppo_traffic_agent.pth")

    print("Training complete! Model saved to ppo_traffic_agent.pth.")
