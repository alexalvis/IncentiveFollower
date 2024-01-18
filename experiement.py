import json
import time
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from OvercookedEnvironment import OvercookedEnvironment
from GradientCal import GC


def run_experiment(environment, args):
    # This is the function used to generate small transition MDP example.
    V, original_policy = environment.get_policy_entropy([], 1)
    # Learning rate influence the result from the convergence aspect. Small learning rate wll make the convergence
    # criteria satisfy too early.
    lr_x = args["lr_x"]  # The learning rate of side-payment
    modifylist = args["modifylist"]
    epsilon = args["epsilon"]  # Convergence threshold
    weight = args["weight"]  # weight of the cost
    trajectory_samples = args["trajectory_samples"]
    trajectory_length = args["trajectory_length"]
    # Whether we use trajectory to approximate policy. 0 represents exact policy, 1 represents approximate policy
    approximate_flag = 0
    GradientCal = GC(environment, lr_x, original_policy, epsilon, modifylist, weight, approximate_flag)

    # Set initial side payment values
    for n in modifylist:
        GradientCal.x[n] = args["initial_side_payments"]
    x_res, x_history, J_history = GradientCal.SGD(N=50, max_iterations=args["max_iterations"])

    # new_policy[state] -> new_policy[state][action]
    new_policy = GradientCal.policy
    for state in new_policy:
        new_policy[state] = action_list_2_dict(new_policy[state], environment.actions)

    # Delivery rate data
    original_policy_deliver_data = []
    new_policy_deliver_data = []

    # Reward data
    original_policy_total_chef_reward_data = []
    new_policy_total_chef_reward_data = []
    original_policy_total_leader_reward_data = []
    new_policy_total_leader_reward_data = []

    new_chef_reward_function, new_leader_reward_function = get_reward_from_list(environment.initial_chef_reward(),
                                                                                environment.initial_leader_reward(),
                                                                                weight, environment.get_states(),
                                                                                environment.get_actions(), x_res)

    for i in range(trajectory_samples):
        original_policy_sample = environment.generate_experiment_sample(original_policy,
                                                                        max_trajectory_length=trajectory_length)
        new_policy_sample = environment.generate_experiment_sample(new_policy, max_trajectory_length=trajectory_length)

        original_chef_reward_history, original_leader_reward_history = \
            calculate_trajectory_rewards(traj=original_policy_sample,
                                         chef_reward_function=environment.initial_chef_reward(),
                                         leader_reward_function=environment.initial_leader_reward())

        new_chef_reward_history, new_leader_reward_history = \
            calculate_trajectory_rewards(traj=new_policy_sample, chef_reward_function=new_chef_reward_function,
                                         leader_reward_function=new_leader_reward_function)

        # Record delivery count data
        warm_delivers, cold_delivers = count_delivers(original_policy_sample)
        original_policy_deliver_data.append((warm_delivers, cold_delivers))

        warm_delivers, cold_delivers = count_delivers(new_policy_sample)
        new_policy_deliver_data.append((warm_delivers, cold_delivers))

        # Record reward data
        original_policy_total_chef_reward_data.append(sum(original_chef_reward_history))
        new_policy_total_chef_reward_data.append(sum(new_chef_reward_history))
        original_policy_total_leader_reward_data.append(sum(original_leader_reward_history))
        new_policy_total_leader_reward_data.append(sum(new_leader_reward_history))

    reward_data = (original_policy_total_chef_reward_data, new_policy_total_chef_reward_data,
                   original_policy_total_leader_reward_data, new_policy_total_leader_reward_data)
    return x_res, x_history, J_history, original_policy_deliver_data, new_policy_deliver_data, reward_data


def save_experiment(x_res, x_history, J_history, original_policy_deliver_data, new_policy_deliver_data, reward_data,
                    experiment_parameters, environment_parameters, path=f"experiment"):
    if not os.path.exists(path):
        os.mkdir(path)
    # Save experiment parameters
    with open(f"{path}\experiment_parameters.txt", 'w') as file:
        json.dump(experiment_parameters, file, indent=2)
    with open(f"{path}\environment_parameters.txt", 'w') as file:
        json.dump(environment_parameters, file, indent=2)
    # Make graph of side payments vs iteration
    save_iteration_graph(x_history, path=f"{path}\side_payment_graph.png", ylabel="Side Payment Values")
    # Make graph of objective function vs iteration
    save_iteration_graph(J_history, path=f"{path}\objective_function_graph.png", ylabel="J")
    # Make graph showing average number of dishes delivered with new/old policy
    save_dish_deliver_count_graph(original_policy_deliver_data, new_policy_deliver_data,
                                  experiment_parameters["trajectory_length"],
                                  experiment_parameters["trajectory_samples"],
                                  path=f"{path}\dish_deliver_graph.png")
    # Make graph showing average reward for leader and chef for new/old policy
    save_reward_data_graph(reward_data, experiment_parameters["trajectory_length"],
                           experiment_parameters["trajectory_samples"], path=f"{path}\\reward_graph.png")
    # Save final side payment array
    with open(f"{path}\side_payments.txt", 'w') as file:
        for x in x_res:
            file.write(f"{str(x)}\n")

    # Pickle and save full side payment history
    with open(f"{path}\side_payment_history.pickle", 'wb') as file:
        pickle.dump(x_history, file)
    # Save objective function data
    with open(f"{path}\objective_function_data.txt", 'w') as file:
        for j in J_history:
            file.write(f"{j}\n")


def save_iteration_graph(data, path, ylabel=""):
    plt.plot(np.arange(0, len(data)), data, color="blue")
    tick_interval = int(round(len(data) / 10, -1)) if int(round(len(data) / 10, -1)) > 0 else 5
    plt.xticks(np.arange(0, len(data), tick_interval))

    plt.xlabel("Num. of Iterations")
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.clf()


def save_dish_deliver_count_graph(original_policy_deliver_data, new_policy_deliver_data,
                                  trajectory_length, trajectory_samples, path):
    original_warm_average = sum([data[0] for data in original_policy_deliver_data]) / len(original_policy_deliver_data)
    original_cold_average = sum([data[1] for data in original_policy_deliver_data]) / len(original_policy_deliver_data)
    new_warm_average = sum([data[0] for data in new_policy_deliver_data]) / len(new_policy_deliver_data)
    new_cold_average = sum([data[1] for data in new_policy_deliver_data]) / len(new_policy_deliver_data)

    policies = ["Original Policy", "New Policy"]
    data = {
        "Average Warm Dishes Delivered": (original_warm_average, new_warm_average),
        "Average Cold Dishes Delivered": (original_cold_average, new_cold_average)
    }

    x = np.arange(len(policies))
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Average Num. of Dishes Delivered')
    ax.set_title(f"Average Dishes Delivered Over {trajectory_samples} Runs of Length {trajectory_length}")
    ax.set_xticks(x + width, policies)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.5 * max(original_warm_average, original_cold_average, new_warm_average, new_cold_average))
    plt.savefig(path)
    plt.clf()


def save_reward_data_graph(reward_data, trajectory_samples, trajectory_length, path):
    original_policy_total_chef_reward_data, new_policy_total_chef_reward_data, \
    original_policy_total_leader_reward_data, new_policy_total_leader_reward_data = reward_data

    original_chef_reward_average = sum(original_policy_total_chef_reward_data) / len(original_policy_total_chef_reward_data)
    new_chef_reward_average = sum(new_policy_total_chef_reward_data) / len(new_policy_total_chef_reward_data)
    original_leader_reward_average = sum(original_policy_total_leader_reward_data) / len(original_policy_total_leader_reward_data)
    new_leader_reward_average = sum(new_policy_total_leader_reward_data) / len(new_policy_total_leader_reward_data)

    policies = ["Original Policy", "New Policy"]
    data = {
        "Chef Average Reward": (original_chef_reward_average, new_chef_reward_average),
        "Leader Average Reward": (original_leader_reward_average, new_leader_reward_average)
    }

    x = np.arange(len(policies))
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Average Total Reward')
    ax.set_title(f"Average Total Reward Over {trajectory_samples} Runs of Length {trajectory_length}")
    ax.set_xticks(x + width, policies)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.5*max(original_chef_reward_average, new_chef_reward_average, original_leader_reward_average, new_leader_reward_average))
    plt.savefig(path)
    plt.clf()


def action_list_2_dict(action_list, actions):
    new_entry = dict()
    for i, action in enumerate(actions):
        new_entry[action] = action_list[i]
    return new_entry


def count_chef_delivers(traj):
    warm_delivers = 0
    cold_delivers = 0

    for i in range(0, len(traj) - 1, 2):
        stove_state, counter_state = traj[i]
        action = traj[i + 1]

        # Deliver action will give priority to cold dishes
        if action == "deliver" and "cold" in counter_state:
            cold_delivers += 1
        elif action == "deliver" and "warm" in counter_state:
            warm_delivers += 1

    return warm_delivers, cold_delivers


def count_delivers(traj):
    warm_delivers = 0
    cold_delivers = 0

    for i in range(0, len(traj), 2):
        if i >= 2:
            old_stove_state, old_counter_state = traj[i - 2]
            new_stove_state, new_counter_state = traj[i]

            warm_delivers += sum([1 for i, _ in enumerate(new_counter_state)
                                  if (old_counter_state[i] == "warm") and new_counter_state[i] == "empty"])
            cold_delivers += sum([1 for i, _ in enumerate(new_counter_state)
                                  if (old_counter_state[i] == "cold") and new_counter_state[i] == "empty"])

    return warm_delivers, cold_delivers


def get_reward_from_list(initial_chef_reward, initial_leader_reward, weight, states, actions, x):
    i = 0
    chef_reward = dict(zip(states, [dict(zip(actions, [0 for _ in actions])) for _ in states]))
    leader_reward = dict(zip(states, [dict(zip(actions, [0 for _ in actions])) for _ in states]))
    for st in states:
        for act in actions:
            chef_reward[st][act] = initial_chef_reward[st][act] + x[i]
            leader_reward[st][act] = initial_leader_reward[st][act] - weight * x[i]
            i += 1
    return chef_reward, leader_reward


def calculate_trajectory_rewards(traj, chef_reward_function, leader_reward_function):
    chef_reward_history = []
    leader_reward_history = []
    for i in range(0, len(traj) - 1, 2):
        state = traj[i]
        action = traj[i + 1]

        chef_reward_history.append(chef_reward_function[state][action])
        leader_reward_history.append(leader_reward_function[state][action])

    return chef_reward_history, leader_reward_history


def main():
    environment_parameters = {
        "stove_size": 2,
        "counter_size": 2,
        "burn_probability": 0.1,
        "cold_probability": 1,
        "deliver_probability": 0.2,
        "cook_food_reward": 10,
        "burned_food_penalty": -10,
        "warm_food_delivered_reward": 50,
        "cold_food_delivered_reward": 5,
    }
    environment = OvercookedEnvironment(
        environment_parameters["stove_size"],
        environment_parameters["counter_size"],
        environment_parameters["burn_probability"],
        environment_parameters["cold_probability"],
        environment_parameters["deliver_probability"],
        environment_parameters["cook_food_reward"],
        environment_parameters["burned_food_penalty"],
        environment_parameters["warm_food_delivered_reward"],
        environment_parameters["cold_food_delivered_reward"]
    )

    # Allow side payments for delivery
    modifylist = [i for i in range(len(environment.get_state_action_list())) if
                  environment.get_state_action_list()[i][-7:] == "deliver"]

    experiment_args = {
        "lr_x": 0.1,
        "epsilon": 1e-5,
        "max_iterations": 50,  # value of -1 will iterate until convergence
        "weight": 1,
        "modifylist": modifylist,
        "initial_side_payments": 1,
        "trajectory_length": 100,  # How long each sample trajectory should be
        "trajectory_samples": 5  # How many trajectories to sample with the new/old policy
    }

    x_res, x_history, J_history, original_policy_deliver_data, new_policy_deliver_data, reward_data = \
        run_experiment(environment, experiment_args)

    save_experiment(x_res, x_history, J_history,
                    experiment_parameters=experiment_args,
                    environment_parameters=environment_parameters,
                    original_policy_deliver_data=original_policy_deliver_data,
                    new_policy_deliver_data=new_policy_deliver_data,
                    reward_data=reward_data,
                    path=f"experiments\\{time.strftime('%Y_%m_%d_%H_%M_%S')}")


if __name__ == "__main__":
    main()
