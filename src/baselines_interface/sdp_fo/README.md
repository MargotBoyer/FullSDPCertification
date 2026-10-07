# SDP-FO adapter (Dathathri et al. 2020)

Bridges FastSDPCertification's networks/bounds to the official `jax_verify` implementation
of SDP-FO. See `baselines/README.md` for the dedicated `baselines` conda env setup.

## Why two environments

`jax_verify`'s SDP-FO code depends on a pinned old `jax` (`<0.6`, see `baselines/README.md`),
incompatible with the `certif` env's torch/MOSEK/CVXPY stack. The adapter is split across
the env boundary and the two sides talk through a file, not a shared process:

1. **`certif` env** (this project's usual stack): load the network (`networks/network.py`)
   and compute bounds (`bounds.py` / `bounds_crown.py`) exactly as `certification_problem.py`
   does, then call `instance_io.save_instance(...)` to write a single `.npz`.
2. **`baselines` env**: `run_sdp_fo.py` loads that `.npz` and runs the SDP-FO solver.

## Files

- `instance_io.py` — `save_instance`/`load_instance`: the `.npz` schema (`W_k`, `b_k` per
  layer, `L_k`/`U_k` per layer including the input ball, `label`, `target_label`,
  `input_bounds`). No jax/jax_verify dependency — importable from the `certif` env.
- `adapter.py` — `build_params`/`build_bounds`/`solve_sdp_fo`: converts to `jax_verify`'s
  `(W, b)`-pair / `IntBound` conventions and calls `solve_sdp_dual_simple`. Requires the
  `baselines` env.
- `run_sdp_fo.py` — CLI: `.npz` in, result `.json` out (`optimal_value`, `time`, `status`).

## Conventions carried over from the main pipeline

- **Bounds**: `L[0]`/`U[0]` are the input ball bounds; `L[k]`/`U[k]` for `k>=1` are
  **pre-activation** bounds (same convention as `bounds.py`). `adapter.build_bounds` turns
  these into `jax_verify`'s `IntBound(lb, ub, lb_pre, ub_pre)` the same way
  `jax_verify.extensions.sdp_verify.utils.boundprop` would — so this adapter uses *our*
  precomputed bounds instead of calling `jax_verify`'s own IBP/CROWN-IBP boundprop. This is
  what makes the comparison to TargetedSDP/UntargetedSDP apples-to-apples: same bounds,
  different SDP relaxation and solve method.
- **Sign convention**: `optimal_value = -verified_ub`, matching `results.csv`'s
  `min_j z^y - z^j` (>= 0 means certified robust for that target).
- **Weight layout**: `networks/network.py`'s `extract_weights()` returns `W[k]` as
  `(n_out, n_in)` (PyTorch `nn.Linear` convention); `jax_verify` expects `(n_in, n_out)`
  (`inputs @ W + b`) — `build_params` transposes.

## Known gotcha

`jax_verify.extensions.sdp_verify.sdp_verify.solve_sdp_dual_simple` defaults to
`optax.adam(1e3)` when no optimizer is passed — this learning rate diverges (verified by a
smoke test: loss explodes from O(1) to O(1e3) within a few hundred steps). `adapter.py`
always overrides this with `optax.adam(1e-3)` unless the caller passes `opt=` explicitly.

## Not yet wired up

- No integration into `certification_problem.py`'s run loop / `results.csv` schema yet —
  `run_sdp_fo.py` writes a standalone JSON per instance.
- No batching across samples/targets — one `.npz` per (sample, target_label) instance.
