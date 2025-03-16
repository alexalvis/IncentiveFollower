# IncentiveFollower: A Reinforcement Learning Project

This project explores the application of reinforcement learning to model a leader-follower interaction, where a leader incentivizes a follower to achieve a desired outcome.  The project includes implementations using both a simple MDP and a more complex GridWorld environment. Model-free and model-based approaches are explored.


## Key Features

* **Multiple Environments:**  The project simulates two environments: a simple Markov Decision Process (MDP) and a more complex GridWorld.  GridWorldV2 offers an enhanced, customizable version of the GridWorld.
* **Leader-Follower Interaction:** Models a scenario where a leader incentivizes a follower (agent) using side payments.
* **Model-Based and Model-Free Approaches:** Implements both model-based and model-free reinforcement learning techniques to solve for optimal leader strategies.
* **Stochastic Transitions:** Incorporates stochasticity in state transitions, adding complexity to the problem.
* **Policy Iteration:** Employs policy iteration to find optimal policies in both environments.
* **Side Payment Optimization:** The leader optimizes side payments to maximize their own reward.
* **Trajectory Sampling:** Uses trajectory sampling to approximate the follower's policy in the model-free approach.
* **Visualization:** Includes visualization scripts (`draw.py`) to display experiment results.


## Technologies Used

* Python 3
* NumPy
* Matplotlib


## Prerequisites

* Python 3.7 or higher
* Required Python packages: NumPy, Matplotlib
  ```bash
  pip install numpy matplotlib
  ```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   ```
2. **Navigate to the project directory:**
   ```bash
   cd IncentiveFollower
   ```
3. **Install dependencies:** (If not already installed as per the Prerequisites).
   ```bash
   pip install numpy matplotlib
   ```


## Usage Examples

The project contains several Python files implementing different aspects of the leader-follower problem:

* **`GridWorld.py` and `GridWorldV2.py`:** Define the GridWorld environment. `GridWorldV2.py` is an enhanced version. You can modify parameters like `width`, `height`, `stoPar`, `F`, `G`, `IDS`, `Barrier`, `gamma`, and `tau` within the `createGridWorldBarrier_new2()` function to customize the environment.

* **`OvercookedEnvironment.py`:** Defines a more complex Overcooked-style environment. Parameters are configurable in the constructor. A `main` function is provided to showcase its usage.

* **`MDP.py`:** Defines a simple Markov Decision Process environment. The `create_mdp()` function creates a default MDP, but you can modify its parameters within the `MDP` class constructor.

* **`GradientCal.py`:** Contains the core logic for gradient calculation and optimization of side payments.  It includes examples (`MDP_example`, `GridW_example`, `GridW_example_V2`) showing how to use the different environments. You can adjust parameters such as learning rate (`lr_x`), the list of modifiable reward states (`modifylist`), convergence threshold (`epsilon`), weight of cost (`weight`), and whether to approximate policy (`approximate_flag`).

* **`Sample.py`:**  Provides the functionality for generating and analyzing trajectories.

* **`draw.py`:** Contains functions for visualizing the results of experiments.


**Running Examples:**

To run the examples:

```bash
python GradientCal.py 
```
This will execute one of the example scenarios within `GradientCal.py`.  Uncomment the desired example function (`MDP_example`, `GridW_example`, `GridW_example_V2`) to choose the specific environment.

To generate and view the plots after running `GradientCal.py`, execute:

```bash
python draw.py 
```
Remember to uncomment the desired plot function in draw.py.  The script will save plots to EPS and PDF files.


## Project Structure

```
IncentiveFollower/
├── GradientCal.py       # Gradient calculation and optimization
├── GridWorld.py         # GridWorld environment
├── GridWorldV2.py       # Enhanced GridWorld environment
├── MDP.py               # Simple MDP environment
├── OvercookedEnvironment.py # Overcooked-style environment
├── README.md            # This file
├── Sample.py            # Trajectory generation and analysis
├── draw.py              # Plotting functions
```

## Contributing Guidelines

(No explicit contributing guidelines were found in the provided codebase.)


## License Information

(No explicit license information was found in the provided codebase.)

## Error Messages

* **"Transition is incorrect"**: This error indicates that the probabilities in the transition matrix for a given state-action pair do not sum to 1 (within a tolerance).  Check the `gettrans()` or `get_transition()` functions in the respective environment files.
* **Convergence issues:** The optimization algorithms (`SGD` in `GradientCal.py`) may not converge if the learning rate (`lr_x`) is too high or too low.  Experiment with different learning rates.

This README provides a starting point for understanding and using this project.  Further documentation within the individual Python files may provide more specific details and explanations.
