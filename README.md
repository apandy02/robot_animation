# Robot Animation

This repository contains the code for the Heartificial Implementation of Expressive Robot behavior transfer from animation to simulation.

## Getting Started

### Setup

1. Install UV
   
   with pip:
   ```bash
   pip install uv
   ```
   or using wget:
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:
   ```bash
   git clone git@github.com:apandy02/robot_animation.git
   cd robot_animation
   ```

3. Create virtual environment and sync dependencies:
   ```bash
   uv venv
   uv sync
   ```

4. Set up Weights & Biases:
   
   a. Create a W&B account at https://docs.wandb.ai/quickstart/ if you don't have one
   
   b. Get your API key from your W&B account settings
   
   c. Add your API key to your shell configuration file:
   ```bash
   echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc  # or ~/.bash_profile or ~/.zshrc
   source ~/.bashrc  # or the appropriate file you modified
   ```
   
   d. Login to W&B:
   ```bash
   uv run wandb login
   ```

## Running the Project

To train the robot using PPO:

```bash
uv run src/robot_animation/ppo_training.py
```

## Linting and Formatting

To run the linter and formatter:

```bash
uvx isort .
uvx ruff check . --fix
```
