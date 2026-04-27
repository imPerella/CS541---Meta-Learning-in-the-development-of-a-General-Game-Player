# Meta-Learning for a General Game Player

## Project Overview (README UPDATES NEEDED)
The goal of this project is to develop a general game-playing engine utilizing a **meta-learning approach**. The primary idea is for a Neural Network (NN) to retain "knowledge" from various game configurations and transfer it to unseen or differently configured games. The NN will map varied game parameters (such as board size, number of pieces, game type) to heuristic evaluation functions suitable for non-ML agents like **Minimax**.

## Current State & Dataset Generation
The current codebase focuses on the **Dataset Generation** phase. The data generated provides:
*   **X (Input):** Game configuration descriptors (e.g., [rows, cols, num_pieces, ..., in_a_row_to_win, turns_per_block]).
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
*   proposal.tex: Contains detailed formal plans and baseline assumptions for the project.
*   Game.py: Abstract base class dictating game logic and heuristic evaluation loops.
*   State.py: Contains individual classes capturing internal game states via NumPy arrays.
*   Generate_Datasets.py: Main module calling variations for all implemented games to produce (X, Y) pairs.
*   Generation_Functions.py: Game runner generating random samples to calculate baseline heuristic stats.
*   Individual Game Implementations (Ataxx.py, Checkers.py, ConnectFour.py, Othello.py, TicTacToe.py).

## How to Run Dataset Generation
To generate the training suite of game configurations and target heuristic values, import or run Generate_Datasets.py:

```python
from Generate_Datasets import dataset

# Returns combinations of X parameters and Y heuristic distributions.
# n_samples controls random rollout count used to estimate Y labels.
X, Y = dataset(variant_values=[10, 10, 10, 10, 10, 10], n_samples=1000)
```

## Dependencies
Instill requirements via pip install -r requirements.txt. The only current external dependency is numpy.
