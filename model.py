from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# One-hot flags for datasets to include in training.
# Order: TicTacToe, ConnectFour, Othello, Ataxx, Checkers
DATASETS = [1, 1, 1, 1, 1]

DATASET_INFO = [
	("tic_tac_toe", "tic_tac_toe.npz"),
	("connect_four", "connect_four.npz"),
	("othello", "othello.npz"),
	("ataxx", "ataxx.npz"),
	("checkers", "checkers.npz"),
]

FEATURE_NAMES = [ # must match order in X arrays, len=13
	"game_id",
	"rows",
	"cols",
	"num_pieces",
	"max_pieces_per_player",
	"num_unique_pieces",
	"placement_game",
	"captures",
	"space_game",
	"edge_unplayable_ratio",
	"inner_unplayable_ratio",
	"in_a_row_to_win",
	"turns_per_block",
]

HEURISTIC_NAMES = ["control", "mobility", "stability", "connectivity", "tension"]

DATASET_DIR = Path("generated_datasets")
OUTPUT_DIR = Path("trained_models")

# Hyperparams
SEED = 42
HIDDEN_SIZES = [64, 256, 64]
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32 # 32 has been tested to be better than 16 and 64
EPOCHS = 120
TEST_SPLIT = 0.2


class HeuristicMetaRegressor(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_sizes, dropout):
		super().__init__()
		layers = []
		current_dim = input_dim
		for size in hidden_sizes:
			layers.append(nn.Linear(current_dim, size))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			current_dim = size
		layers.append(nn.Linear(current_dim, output_dim))
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _prepare_targets(y_values):
	y_array = np.asarray(y_values, dtype=np.float32)
	if y_array.ndim == 3:
		return y_array.reshape(y_array.shape[0], -1) # flatten 5 mean and 5 std dimensions
	if y_array.ndim == 2:
		return y_array
	raise ValueError(f"Check Y shape?: {y_array.shape}")


def _standardize(values):
	mean = values.mean(axis=0)
	std = values.std(axis=0)
	std = np.where(std < 1e-8, 1.0, std)
	normalized = (values - mean) / std
	return normalized, mean, std


def load_selected_datasets(flags, dataset_dir, test_split=0.0):
	if len(flags) != len(DATASET_INFO):
		raise ValueError("datasets must have 5 elements in the expected order")

	x_groups = []
	y_groups = []
	used_names = []

	for dataset_id, (flag, (name, filename)) in enumerate(zip(flags, DATASET_INFO), start=1):
		if int(flag) == 0:
			continue
		path = dataset_dir / filename
		if not path.exists():
			raise FileNotFoundError(f"Missing dataset: {path}")
		data = np.load(path)
		x_values = data["X"]
		game_id = np.full((x_values.shape[0], 1), dataset_id, dtype=x_values.dtype)
		x_groups.append(np.concatenate([game_id, x_values], axis=1))
		y_groups.append(data["Y"])
		used_names.append(name)

	if not x_groups:
		raise ValueError("Select a dataset!")

	x_values = np.concatenate(x_groups, axis=0)
	y_values = np.concatenate(y_groups, axis=0)

	if test_split is None or test_split == 0.0:
		return x_values, y_values, used_names, None, None

	n_total = x_values.shape[0]

	indices = np.random.permutation(n_total)
	x_values = x_values[indices]
	y_values = y_values[indices]

	test_count = int(round(n_total * test_split))

	x_values_test = x_values[:test_count]
	y_values_test = y_values[:test_count]
	x_values_train = x_values[test_count:]
	y_values_train = y_values[test_count:]

	return x_values_train, y_values_train, used_names, x_values_test, y_values_test


def train():
	# set_seed(SEED)

	x_values_train, y_values_train, used_names, x_values_test, y_values_test = load_selected_datasets(DATASETS, DATASET_DIR, test_split=TEST_SPLIT)
	y_values_train = _prepare_targets(y_values_train) # flattening
	if y_values_test is not None:
		y_values_test = _prepare_targets(y_values_test)

	# Double checking
	x_values_train = x_values_train.astype(np.float32)
	y_values_train = y_values_train.astype(np.float32)
	if x_values_test is not None:
		x_values_test = x_values_test.astype(np.float32)
		y_values_test = y_values_test.astype(np.float32)

	x_norm_train, x_mean, x_std = _standardize(x_values_train)
	y_norm_train, y_mean, y_std = _standardize(y_values_train)
	if x_values_test is not None:
		x_norm_test = (x_values_test - x_mean) / x_std
		y_norm_test = (y_values_test - y_mean) / y_std

	input_dim = x_norm_train.shape[1]
	output_dim = y_norm_train.shape[1]

	if input_dim != len(FEATURE_NAMES):
		print(f"Warning: expected {len(FEATURE_NAMES)} features but got {input_dim}, given no modifications to our code, this should be 13")
	if output_dim != len(HEURISTIC_NAMES) * 2:
		print(f"Warning: expected {len(HEURISTIC_NAMES) * 2} targets but got {output_dim}, given no modifications to our code, this should be 10 (5 heuristic means and 5 heuristic stds)")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset_train = TensorDataset(
		torch.from_numpy(x_norm_train),
		torch.from_numpy(y_norm_train),
	)
	train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

	test_loader = None
	if x_values_test is not None:
		dataset_test = TensorDataset(
			torch.from_numpy(x_norm_test),
			torch.from_numpy(y_norm_test),
		)
		test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

	model = HeuristicMetaRegressor(input_dim, output_dim, HIDDEN_SIZES, DROPOUT).to(device)
	print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
	criterion = nn.MSELoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	for epoch in range(EPOCHS):
		model.train()
		running_loss = 0.0

		for batch_x, batch_y in train_loader:
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)

			optimizer.zero_grad()
			preds = model(batch_x)
			loss = criterion(preds, batch_y)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * batch_x.size(0)

		epoch_loss = running_loss / len(dataset_train)
		if test_loader is None:
			print(f"Epoch {epoch:4d} | Train Loss: {epoch_loss:.6f}")
		else:
			model.eval()
			test_running = 0.0
			with torch.no_grad():
				for batch_x, batch_y in test_loader:
					batch_x = batch_x.to(device)
					batch_y = batch_y.to(device)
					preds = model(batch_x)
					loss = criterion(preds, batch_y)
					test_running += loss.item() * batch_x.size(0)
			test_loss = test_running / len(dataset_test)
			print(f"Epoch {epoch:4d} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	flag_string = "".join(str(int(bool(flag))) for flag in DATASETS)
	model_name = f"model_{flag_string}.pt"
	output_path = OUTPUT_DIR / model_name

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"input_dim": input_dim,
			"output_dim": output_dim,
			"hidden_sizes": HIDDEN_SIZES,
			"dropout": DROPOUT,
			"x_mean": x_mean,
			"x_std": x_std,
			"y_mean": y_mean,
			"y_std": y_std,
			"dataset_flags": DATASETS,
			"dataset_names": used_names,
			"feature_names": FEATURE_NAMES,
			"heuristic_names": HEURISTIC_NAMES,
		},
		output_path,
	)

	print(f"Saved model to {output_path}")


if __name__ == "__main__":
	train()