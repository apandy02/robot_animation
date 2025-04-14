# Robot Animation

This repository contains the code for the Heartificial Implementation of Expressive Robot behavior transfer from animation to simulation.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Setup](#setup)
- [Running the Project](#running-the-project)
- [Linting and Formatting](#linting-and-formatting)

## Project Structure

```
robot_animation/
├── src/                    # Source code directory
│   └── robot_animation/    # Main package containing core functionality
├── data/                   # Data directory for storing datasets and processed data
├── models/                 # Directory for storing trained models and checkpoints
├── experimentation/        # Directory for experimental code and notebooks
├── blender/               # Blender-related files and assets
├── robot_models/          # Robot model definitions and assets
└── pyproject.toml         # Project configuration and dependencies
```

## Getting Started

### Setup

1. Install UV
   
   with pip:
   ```bash
   pip install uv
   ```
   or using wget (recommended):
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
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

## Running PPO Training

### Using Stable Baselines3

To train the robot using PPO:

```bash
uv run src/robot_animation/train_ppo_sb3.py --total-timesteps 100000
```

### Using Puffer

To train the robot using Puffer:

```bash
uv run src/robot_animation/puffer/train_ppo_puffer.py --train.total-timesteps 100000
```

## Linting and Formatting

To run the linter and formatter:

```bash
uvx isort .
uvx ruff check . --fix
```
