# Dino RL - Chrome Dinosaur Game Agent

A reinforcement learning project that trains an AI agent to play the Chrome Dinosaur Game using Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms from the Stable Baselines3 library.

## Overview

This project demonstrates training a deep reinforcement learning agent to master the Chrome dinosaur game. The agent uses computer vision to capture game frames and learns optimal strategies through trial and error.

## Project Structure

```
dino-rl/
├── train.py              # Training script for DQN agent
├── eval.py               # Evaluation script for testing trained models
├── requirements.txt      # Python dependencies
├── configs/
│   └── game_url.json     # Configuration for game URL
├── envs/
│   └── dino_env.py       # Custom Gymnasium environment for the dinosaur game
├── game/                 # Chrome dinosaur game HTML/CSS/JS files
│   ├── index.html
│   ├── index.css
│   ├── index.js
│   ├── LICENSE
│   ├── README.md
│   └── assets/           # Game assets (sprites, etc.)
└── runs/                 # TensorBoard logs and model checkpoints
```

## Features

- **Custom Gymnasium Environment**: `DinoEnv` wraps the Chrome dinosaur game with RL-compatible interfaces
- **Deep Q-Network (DQN)**: Uses CNN policy for visual learning
- **Frame Stacking**: Processes 4 consecutive frames to capture motion
- **Computer Vision**: Captures game frames using Selenium and OpenCV
- **Automatic Model Checkpointing**: Saves best models during training
- **TensorBoard Integration**: Visualize training metrics and performance

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.8+
- PyTorch (for neural networks)
- Stable Baselines3 (RL algorithms)
- Selenium (browser automation)
- OpenCV (image processing)
- Gymnasium (RL environment API)

See `requirements.txt` for complete list of dependencies.

## Usage

### Training

Run the training script to train the DQN agent:

```bash
python train.py
```

**Training Configuration:**
- Algorithm: DQN with CNN Policy
- Total timesteps: 1,000,000
- Learning rate: 1e-4
- Replay buffer size: 50,000
- Batch size: 32
- Frame stack: 4 frames
- Gamma (discount factor): 0.99
- Exploration: 10% of training, final epsilon: 0.01

Training logs and model checkpoints are saved to the `runs/` directory.

### Evaluation

Test a trained model:

```bash
python eval.py
```

This script:
- Loads a trained PPO model
- Runs 10 evaluation episodes
- Displays the game window with the agent playing
- Prints rewards for each episode

## Architecture

### Environment (DinoEnv)

- **Observation Space**: 84×84 grayscale images (single frame)
- **Action Space**: 3 discrete actions
  - 0: Do nothing
  - 1: Jump
  - 2: Duck
- **Game Control**: Selenium WebDriver automates key presses to the game

### Model

- **Policy**: CNN (Convolutional Neural Network)
- **Frame Stack**: 4 frames of history
- **Network Architecture**: Standard DQN with dueling architecture

## Training

The agent learns through DQN with the following approach:
1. Agent observes current game state (84×84 grayscale image)
2. CNN policy network outputs Q-values for each action
3. Agent selects action using epsilon-greedy strategy
4. Environment executes action and returns new state + reward
5. Experience stored in replay buffer
6. Model trains on mini-batches from replay buffer
7. Target network updated periodically

## Performance

Training runs are logged to TensorBoard:

```bash
tensorboard --logdir runs/
```

View episode rewards, episode lengths, and other metrics over time.

## TensorBoard Logs

Training histories are stored in `runs/DQN_*/` directories with TensorFlow event files for visualization.

## System Requirements

- **Browser**: Chrome/Chromium (required for game rendering)
- **GPU**: Recommended (CUDA support configured in training script)
- **Display**: Required for browser automation and rendering

## Notes

- The game HTML file is loaded locally via `file://` protocol
- Selenium WebDriver handles browser automation
- Computer vision (screen capture) extracts game state from rendered frames
- Model training requires significant compute resources (GPU recommended)

## Future Improvements

- Experiment with other algorithms (A3C, PPO, SAC)
- Hyperparameter optimization
- Transfer learning from other Atari games
- Multi-agent learning
- Curriculum learning strategies

## License

See individual component licenses (Chrome game included in `game/LICENSE`)

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Chrome Dinosaur Game](https://chromedino.com/)
