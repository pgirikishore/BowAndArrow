
# BowAndArrow: A Reinforcement Learning Shooter Simulation

## Abstract
BowAndArrow is a reinforcement learning (RL) project exploring agent interactions within a simulated environment where the objective is to hit moving targets—specifically balloons. The agent receives positive rewards for hitting red balloons and penalties for hitting yellow ones. Utilizing various RL algorithms and custom convolutional neural networks (CNNs), this project aims to enhance the precision of targeting moving objects. While the agent demonstrates effective shooting capabilities, aiming accuracy remains an area for further improvement.

## Methods and Model

### Game Environment
- **Platform:** Pygame
- **Game Dynamics:** Dynamic environment with balloons moving from the bottom of the screen. 
- **Objective:** Maximize score by hitting red balloons while avoiding yellow balloons.
- **Gameplay:** Each game session allows 20 arrows with 15 balloons appearing randomly.

### Reinforcement Learning Environment
- **Integration:** Custom RL environment integrated with OpenAI Gym.
- **Models Explored:**
  - **Model 1:** MlpPolicy, CnnPolicy, PPO and DQN
  - **Model 2:** Monte Carlo Reinforce with a custom CNN
  - **Model 3:** Monte Carlo Reinforce with a custom CNN
  - **Model 4:** Linear QNet(Deep Q-Learning)

### Technical Stack
- **Libraries:** Pygame, OpenAI Gym, Stable Baselines3, PyTorch
- **Key Algorithms:** Proximal Policy Optimization (PPO), Deep Q-Network (DQN), Monte Carlo Reinforce
- **Policies:** Multi-layer Perceptron Policy (MlpPolicy), Convolutional Neural Network Policy (CnnPolicy)

## Experimental Setup and Hyperparameters

- **Observation Space:** Initially 1000 x 800 x 3, reduced to 600 x 400 x 3 in subsequent models.
- **Action Space:** Discrete actions including 'No action', 'Shoot', 'Move Up', and 'Move Down'.
- **Rewards:** +1 for hitting a red balloon, -1 for a yellow balloon. Modifications in later models included immediate rewards and penalty restructuring to improve decision-making strategies.

## Project Files Description

### `BowAndArrowEnv.py`
Defines the custom environment for the game based on OpenAI's Gym interface.

### `BowAndArrowRL.py`
Implements the reinforcement learning models used to train the agent.

### `PreprocessObservation.py`
Contains functions to preprocess game frames for neural network input.

### `RLPlayGame.py`
Executes the game using trained models to demonstrate the agent's performance.

### `__init__.py`
Initializes the directory as a Python package to simplify imports.

### `BownArrow.py`
Contains the main game logic, including user inputs and game object management.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/pgirikishore/BowAndArrow.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the trained models from the google drive link (https://drive.google.com/drive/folders/1J4p6OlAoJvnTkvi_kAuhiFXMjwM4gJOK?usp=sharing), unzip it and add it to the root folder.
4. Run the simulation:
   ```bash
   python3 /{model-name}/env/RLPlayGame.py 
   ```

## Conclusion and Future Work
The BowAndArrow project provides foundational insights into using reinforcement learning combined with CNNs for dynamic target acquisition. While effective in basic targeting, precision remains a challenge, underscoring the need for further research. Future directions include stabilizing the training environment and integrating more sophisticated CNN architectures to improve both accuracy and efficiency in real-time decision-making contexts.
