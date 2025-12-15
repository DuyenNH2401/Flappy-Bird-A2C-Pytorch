# Reinforcement Learning Flappy Bird

## Description
This repository contains a reinforcement learning (RL) implementation of the classic Flappy Bird game using the Advantage Actor‑Critic (A2C) algorithm. The project demonstrates how to train an agent with PyTorch and Gymnasium, providing scripts for training, testing, and visualizing the learned policy.

## Installation
```bash
# Clone the repository
git clone <repository-url>
cd RL/Project

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Training
```bash
python src/train.py
```
The training script will output episode statistics and save the model checkpoint to `models/a2c_flappy_bird.pth`.

### Testing
```bash
python src/test.py
```
The test script loads the saved model and runs a few episodes with rendering enabled.

## Project Structure
```
project_root/
├── src/                # Source code (Python scripts)
│   ├── agent.py
│   ├── check_dim.py
│   ├── network.py
│   ├── train.py
│   └── test.py
├── models/             # Saved model checkpoints
│   └── a2c_flappy_bird.pth
├── .gitignore
├── requirements.txt
└── README.md
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new feature branch.
3. Ensure code style consistency (PEP8) and add tests if applicable.
4. Submit a pull request with a clear description of changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
