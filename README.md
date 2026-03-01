# Gomoku (Five in a Row) with Q-Learning AI

This project implements a simplified Gomoku game on a 10×10 board.  
An AI agent learns to play using Q-learning.

## Features
- 10×10 board, five in a row to win.
- Q-learning agent with ε-greedy policy.
- GUI with Tkinter (no extra installs).
- Two scripts: `play_gui.py` (human vs AI) and `train.py` (self-play training).

## Requirements
- Python 3.10–3.12
- NumPy

## Usage
1. Install NumPy: `pip install numpy`
2. Train the AI (optional): `python train.py`
3. Play against AI: `run_game.bat` or `python play_gui.py` (press G to start)

## Reward Function
- Win: +1
- Draw: 0
- Each move: -0.01
- Block opponent's three‑in‑a‑row: +0.5

## Project Structure
├── gomoku_game.py

├── q_learning_agent.py

├── train.py

├── run_game.bat

├── play_gui.py

├── .gitignore

└── README.md
