from pathlib import Path

import numpy as np

from Generation_Functions import *


def _to_numpy_array(values):
   try:
      return np.asarray(values, dtype=np.float64)
   except (TypeError, ValueError):
      raise ValueError("FIX BEFORE RUNNING!")


def _save_group_dataset(output_dir, group_name, x_group, y_group):
   output_path = Path(output_dir)
   output_path.mkdir(parents=True, exist_ok=True)

   np.savez_compressed(
      output_path / f"{group_name}.npz",
      X=_to_numpy_array(x_group),
      Y=_to_numpy_array(y_group),
   )


def dataset(variant_values=[3, 3, 3, 3, 3], n_eval_samples=250, output_dir="generated_datasets"):
   if len(variant_values) != 5:
      raise ValueError("FIX BEFORE RUNNING!")

   var1, var2, var3, var4, var5 = variant_values

   X1, Y1 = generate_tic_tac_toe(num_variants=var1, n_samples=n_eval_samples)
   _save_group_dataset(output_dir, "tic_tac_toe", X1, Y1)
   X2, Y2 = generate_connect_four(num_variants=var2, n_samples=n_eval_samples)
   _save_group_dataset(output_dir, "connect_four", X2, Y2)
   X3, Y3 = generate_othello(num_variants=var3, n_samples=n_eval_samples)
   _save_group_dataset(output_dir, "othello", X3, Y3)
   X4, Y4 = generate_ataxx(num_variants=var4, n_samples=n_eval_samples)
   _save_group_dataset(output_dir, "ataxx", X4, Y4)
   X5, Y5 = generate_checkers(num_variants=var5, keep_pieces=True, n_samples=n_eval_samples)
   _save_group_dataset(output_dir, "checkers", X5, Y5)
   # X6, Y6 = generate_checkers(num_variants=var6, keep_pieces=False, n_samples=n_eval_samples)
   # _save_group_dataset(output_dir, "checkers_without_pieces", X6, Y6)

   X = X4 + X5
   Y = Y4 + Y5

   return X, Y


X, Y = dataset([3, 3, 3, 3, 3], n_eval_samples=250)
print("FINAL PRINT: DATASET SIZES")
print(len(X), len(Y))