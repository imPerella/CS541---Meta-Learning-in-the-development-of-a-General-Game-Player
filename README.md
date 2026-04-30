# Meta-Learning for a General Game Player

## Project Overview
The goal of this project is to develop a general game-playing engine utilizing a **meta-learning approach**. The primary idea is for a Neural Network (NN) to retain "knowledge" from various game configurations and transfer it to unseen or differently configured games. The NN will map varied game parameters (such as board size, number of pieces, game type) to heuristic evaluation functions suitable for non-ML agents like **Minimax**.

## Current State & Dataset Generation
The current codebase focuses on the **Dataset Generation** phase. The data generated provides:
*   **X (Input):** Game configuration descriptors (e.g., [rows, cols, num_pieces, max_pieces_per_player, num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win, turns_per_block]).
*   **Y (Label):** Evaluation functions, formatted as means and standard deviations of five distinct heuristics evaluated over random games.

Each generated variant now also includes a **turn-pattern value** controlling how many consecutive turns each player receives before turn control switches to the opponent.

### Five Base Heuristics Evaluated:
1.  **Control** (Board presence/percentage owned)
2.  **Mobility** (Number of available legal moves)
3.  **Stability** (Tiles that are highly resistant to being taken)
4.  **Connectivity** (Largest contiguous grouping of player pieces)
5.  **Tension** (Percentage of moves causing significant and impactful board flips/captures/wins)

### Supported Game Variants
The system currently implements five games and can generate variants around board sizes and specific rules:
*   Tic-Tac-Toe
*   Connect-Four
*   Othello
*   Ataxx
*   Checkers

## Project Structure
*   `Game.py`: Abstract base class dictating game logic and heuristic evaluation loops.
*   `State.py`: Contains individual classes capturing internal game states via NumPy arrays.
*   `Generate_Datasets.py`: Main module calling variations for all implemented games to produce (X, Y) pairs.
*   `Generation_Functions.py`: Game runner generating random samples to calculate baseline heuristic stats.
*   `Player.py`: Minimax/alphabeta helper functions to convert a mean and standard deviation to a playable move
*   `Model.py`: Model architecture and training loop logic
*   `Evaluate_Model.py`: Trains/loads a meta model, runs inference to generate heuristics, converts those mean and standard deviations to playable moves, sends those to the corresponding game engine to play against brute-forced optimal heuristics or random moves
*   Individual Game Implementations (`Ataxx.py`, `Checkers.py`, `ConnectFour.py`, `Othello.py`, `TicTacToe.py`): Implementations of `Game.py` to act as a game engine

## How to run

### Dataset Generation

`python Generate_Datasets.py`

### Model Training

`python Model.py`

### Model Evaluation and Gameplay

`python Evaluate_Model.py`

## Dependencies
Instill requirements via pip install -r requirements.txt. The only current external dependency is numpy.
