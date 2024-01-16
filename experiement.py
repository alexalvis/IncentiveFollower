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
    V, policy = environment.get_policy_entropy([], 1)
    # Learning rate influence the result from the convergence aspect. Small learning rate wll make the convergence
    # criteria satisfy too early.
    lr_x = args["lr_x"]  # The learning rate of side-payment
    modifylist = args["modifylist"]
    epsilon = args["epsilon"]  # Convergence threshold
    weight = args["weight"]  # weight of the cost
    # Whether we use trajectory to approximate policy. 0 represents exact policy, 1 represents approximate policy
    approximate_flag = 0
    GradientCal = GC(environment, lr_x, policy, epsilon, modifylist, weight, approximate_flag)
    # Set initial side payment values
    for n in modifylist:
        GradientCal.x[n] = args["initial_side_payments"]
    x_res, x_history, J_history = GradientCal.SGD(N=50, max_iterations=args["max_iterations"])
    print(x_history)
    return x_res, x_history, J_history


def save_experiment(x_res, x_history, J_history, experiment_parameters, environment_parameters, path=f"experiment"):
    if not os.path.exists(path):
        os.mkdir(path)
    # Save experiment parameters
    with open(f"{path}\experiment_parameters.txt", 'w') as file:
        json.dump(experiment_parameters, file)
    with open(f"{path}\environment_parameters.txt", 'w') as file:
        json.dump(environment_parameters, file)
    # Make graph of side payments vs iteration
    save_iteration_graph(x_history, path=f"{path}\side_payment_graph.png", ylabel="Side Payment Values")
    # Make graph of objective function vs iteration
    save_iteration_graph(J_history, path=f"{path}\objective_function_graph.png", ylabel="J")

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


def main():
    environment_parameters = {
        "stove_size": 2,
        "counter_size": 2,
        "burn_probability": 0.1,
        "cold_probability": 1,
        "deliver_probability": 0.0,
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
        "max_iterations": 100, # value of -1 will iterate until convergence
        "weight": 1,
        "modifylist": modifylist,
        "initial_side_payments": 1,
    }

    # x_res, x_history, J_history = run_experiment(environment, experiment_args)
    # save_experiment(x_res, x_history, J_history, experiment_parameters=experiment_args,
    #                 environment_parameters=environment_parameters,
    #                 path=f"experiments\\{time.strftime('%Y_%m_%d_%H_%M_%S')}")
    V, policy = environment.get_policy_entropy([], 1)
    # print(policy[('empty', 'empty'), ('empty', 'empty')].values())
    traj = environment.generate_sample(policy, 20)
    for state in traj:
        print(state)


if __name__ == "__main__":
    main()
