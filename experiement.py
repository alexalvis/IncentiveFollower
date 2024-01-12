import time
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from OvercookedEnvironment import OvercookedEnvironment
from GradientCal import GC


def run_experiment(environment, max_iterations=-1):
    # This is the function used to generate small transition MDP example.
    V, policy = environment.get_policy_entropy([], 1)
    # Learning rate influence the result from the convergence aspect. Small learning rate wll make the convergence criteria satisfy too early.
    lr_x = 0.05  # The learning rate of side-payment
    # modifylist = [144]  # The action reward you can modify
    # modifylist = [1, 8, 18, 19, 31, 38, 167]
    # modifylist = [i for i in range(len(environment.get_state_action_list()))]
    # Allow side payments for delivery
    modifylist = [i for i in range(len(environment.get_state_action_list())) if environment.get_state_action_list()[i][-7:] == "deliver"]
    epsilon = 1e-5  # Convergence threshold
    weight = 1  # weight of the cost
    approximate_flag = 0  # Whether we use trajectory to approximate policy. 0 represents exact policy, 1 represents approximate policy
    GradientCal = GC(environment, lr_x, policy, epsilon, modifylist, weight, approximate_flag)
    # Set initial side payment values
    for n in modifylist:
        GradientCal.x[n] = 1
    # GradientCal.x[18] = 5 # state 18 = (empty, empty), (warm, empty)
    x_res, x_history = GradientCal.SGD(N=50, max_iterations=max_iterations)

    return x_res, x_history


def save_experiment(x_res, x_history, path=f"experiment"):
    if not os.path.exists(path):
        os.mkdir(path)
    # Make graph of side payments vs iteration
    plt.plot(np.arange(0, len(x_history)), x_history, color="blue")
    tick_interval = int(round(len(x_history)/10, -1)) if int(round(len(x_history)/10, -1)) > 0 else 5
    plt.xticks(np.arange(0, len(x_history), tick_interval))

    plt.savefig(f"{path}\side_payment_graph.png")

    # Save final side payment array
    with open(f"{path}\side_payments.txt", 'w') as file:
        for x in x_res:
            file.write(f"{str(x)}\n")

    # Pickle and save full side payment history
    with open(f"{path}\side_payment_history", 'wb') as file:
        pickle.dump(x_history, file)


def main():
    environment = OvercookedEnvironment(stove_size=2, counter_size=2, burn_probability=0.1, cold_probability=0.5,
                                        deliver_probability=0.2, cook_food_reward=10, burned_food_penalty=-10,
                                        warm_food_delivered_reward=50, cold_food_delivered_reward=5)

    x_res, x_history = run_experiment(environment, max_iterations=20)
    path = f"experiments\{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    save_experiment(x_res, x_history, path=path)


if __name__ == "__main__":
    main()