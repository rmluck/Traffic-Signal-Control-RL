import matplotlib.pyplot as plt

def parse_rewards():
    rewards = {}

    with open("training_log.txt") as file:
        for line in file:
            split_line = line.split()

            if split_line:
                if split_line[0] == "Iteration":
                    iteration = int(split_line[1])
                elif split_line[0] == "Mean":
                    try:
                        reward = round(float(split_line[2]))
                        rewards[iteration] = reward
                    except ValueError as e:
                        continue

    return rewards

def main():
    rewards = parse_rewards()

    plt.plot(rewards.keys(), rewards.values())

    plt.xlabel("Training Iteration")
    plt.ylabel("Mean Reward")
    plt.title("Mean Rewards Over Training Iterations")

    plt.xlim(0, 50)
    plt.ylim(1000, 1350)

    plt.savefig("rewards")

if __name__ == "__main__":
    main()