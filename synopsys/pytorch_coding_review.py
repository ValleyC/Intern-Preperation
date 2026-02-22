"""
================================================================================
PYTHON & PYTORCH CODING REVIEW — Synopsys/Ansys ML Internship
================================================================================
Interview Date: February 24, 2026
Interviewer: Xin Xu (Principal R&D Engineer)

Redesigned for PRACTICAL assessment of Python + PyTorch fluency.
A senior R&D engineer will likely test:
  1. Can you write clean, correct Python? (data structures, OOP, generators)
  2. Can you manipulate tensors and data? (NumPy/PyTorch operations)
  3. Can you build a proper ML training pipeline? (Dataset, training loop, eval)
  4. Can you debug common issues? (shapes, devices, gradients, memory)

NOT likely: "Implement a Fourier Neural Operator from scratch"

This file is organized as:
  Part 1: Python Fundamentals (5 exercises)
  Part 2: NumPy / Tensor Operations (5 exercises)
  Part 3: PyTorch Practical Patterns (5 exercises)
  Part 4: Common Gotchas — Bug Spotting (10 items)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os


# ==============================================================================
# PART 1: PYTHON FUNDAMENTALS
# ==============================================================================
# These test clean Python — the kind of code you'd write daily at Synopsys.
# ==============================================================================


# --- Exercise 1: Data Processing with Dictionaries & Comprehensions -----------
# "We have simulation results stored as a list of dictionaries. Write a function
#  to group them by geometry type and compute the average error per group."

def group_and_average(
    results: List[Dict],
    group_key: str = "geometry",
    value_key: str = "error",
) -> Dict[str, float]:
    """
    Group simulation results by a key and compute the mean of a value field.

    >>> results = [
    ...     {"geometry": "dipole", "error": 0.05, "freq": 2.4e9},
    ...     {"geometry": "patch",  "error": 0.12, "freq": 5.0e9},
    ...     {"geometry": "dipole", "error": 0.03, "freq": 2.4e9},
    ...     {"geometry": "patch",  "error": 0.08, "freq": 5.0e9},
    ...     {"geometry": "horn",   "error": 0.02, "freq": 10e9},
    ... ]
    >>> group_and_average(results)
    {'dipole': 0.04, 'patch': 0.1, 'horn': 0.02}
    """
    groups = defaultdict(list)
    for r in results:
        groups[r[group_key]].append(r[value_key])
    return {k: sum(v) / len(v) for k, v in groups.items()}


# --- Exercise 2: Generator for Large File Processing -------------------------
# "We have a huge CSV of simulation parameters. Write a generator that yields
#  batches of N lines without loading the entire file into memory."

def batch_reader(filepath: str, batch_size: int = 32):
    """
    Yield batches of lines from a file. Memory-efficient for large files.

    Why a generator?
    - Simulation datasets can be GBs. Loading all into RAM is wasteful.
    - Generators produce items lazily — only one batch in memory at a time.
    - This is the same principle behind PyTorch's DataLoader with num_workers.
    """
    batch = []
    with open(filepath, "r") as f:
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:  # Don't forget the last incomplete batch!
        yield batch


# --- Exercise 3: Class Design — Simulation Result Container -------------------
# "Design a class to hold simulation results with proper validation."

class SimulationResult:
    """
    Container for a single simulation result with validation.

    Demonstrates:
    - __init__ with validation
    - __repr__ for debugging
    - __eq__ for comparison
    - Property for derived quantity
    - Class method as alternative constructor
    """

    def __init__(self, name: str, s_params: np.ndarray, freq_ghz: np.ndarray):
        """
        Args:
            name: Design identifier (e.g., "antenna_v3")
            s_params: S-parameter matrix, shape (n_freq, n_ports, n_ports), complex
            freq_ghz: Frequency points in GHz, shape (n_freq,)
        """
        if s_params.shape[0] != freq_ghz.shape[0]:
            raise ValueError(
                f"Frequency dimension mismatch: s_params has {s_params.shape[0]} "
                f"points but freq_ghz has {freq_ghz.shape[0]}"
            )
        self.name = name
        self.s_params = s_params
        self.freq_ghz = freq_ghz

    @property
    def n_ports(self) -> int:
        return self.s_params.shape[1]

    @property
    def s11_db(self) -> np.ndarray:
        """Return S11 in dB: 20 * log10(|S11|)"""
        return 20.0 * np.log10(np.abs(self.s_params[:, 0, 0]) + 1e-12)

    @property
    def resonant_freq_ghz(self) -> float:
        """Frequency where |S11| is minimized (resonance)."""
        return float(self.freq_ghz[np.argmin(self.s11_db)])

    @classmethod
    def from_touchstone(cls, filepath: str) -> "SimulationResult":
        """Alternative constructor: load from a .s2p touchstone file."""
        # In practice, you'd parse the file format here
        raise NotImplementedError("Touchstone parsing not implemented for demo")

    def __repr__(self):
        return (
            f"SimulationResult(name='{self.name}', "
            f"ports={self.n_ports}, "
            f"freq=[{self.freq_ghz[0]:.1f}-{self.freq_ghz[-1]:.1f}] GHz, "
            f"resonance={self.resonant_freq_ghz:.2f} GHz)"
        )

    def __eq__(self, other):
        if not isinstance(other, SimulationResult):
            return NotImplemented
        return (
            self.name == other.name
            and np.allclose(self.s_params, other.s_params)
            and np.allclose(self.freq_ghz, other.freq_ghz)
        )


# --- Exercise 4: Decorator for Timing Functions ------------------------------
# "Write a decorator to time any function. We use this to profile simulation
#  preprocessing, training, and inference."

import time
from functools import wraps

def timer(func):
    """
    Decorator that prints execution time of the wrapped function.

    Why @wraps(func)?
    - Preserves the original function's __name__, __doc__, etc.
    - Without it, debugging shows "wrapper" instead of the actual function name.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper


# --- Exercise 5: Error Handling & Defensive Programming -----------------------
# "Write a function that loads and validates simulation config from a dict.
#  Handle missing keys, wrong types, and invalid values gracefully."

def validate_config(config: Dict) -> Dict:
    """
    Validate and normalize a training configuration dictionary.

    Returns a clean config dict with defaults filled in.
    Raises ValueError with a clear message if something is wrong.
    """
    required_keys = ["model_type", "learning_rate", "num_epochs"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # Type checking with clear messages
    if not isinstance(config["learning_rate"], (int, float)):
        raise ValueError(
            f"learning_rate must be numeric, got {type(config['learning_rate']).__name__}"
        )

    lr = float(config["learning_rate"])
    if not (1e-8 <= lr <= 1.0):
        raise ValueError(f"learning_rate={lr} is outside valid range [1e-8, 1.0]")

    valid_models = {"mlp", "gnn", "fno", "siren"}
    if config["model_type"] not in valid_models:
        raise ValueError(
            f"model_type='{config['model_type']}' not in {valid_models}"
        )

    # Return clean config with defaults
    return {
        "model_type": config["model_type"],
        "learning_rate": lr,
        "num_epochs": int(config["num_epochs"]),
        "batch_size": config.get("batch_size", 32),
        "hidden_dim": config.get("hidden_dim", 128),
        "weight_decay": float(config.get("weight_decay", 1e-4)),
        "device": config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    }


# ==============================================================================
# PART 2: NUMPY / TENSOR OPERATIONS
# ==============================================================================
# Core data manipulation — the daily bread of ML research engineering.
# ==============================================================================


# --- Exercise 6: Broadcasting & Vectorized Operations -------------------------
# "Compute pairwise Euclidean distances between two sets of points WITHOUT loops."

def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between two point sets.

    Args:
        x: shape (N, D) — N points in D dimensions
        y: shape (M, D) — M points in D dimensions

    Returns:
        dist: shape (N, M) — dist[i,j] = ||x[i] - y[j]||

    Key concept: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    This avoids the O(NMD) loop and uses O(NM + ND + MD) with BLAS.
    """
    # Method 1: Using the expansion trick (numerically less stable but fast)
    xx = (x * x).sum(dim=1, keepdim=True)   # (N, 1)
    yy = (y * y).sum(dim=1, keepdim=True).T  # (1, M)
    xy = x @ y.T                              # (N, M)
    dist_sq = xx + yy - 2 * xy
    # Clamp to avoid negative values from numerical errors
    return torch.sqrt(dist_sq.clamp(min=0.0))

    # Method 2 (simpler, more memory): torch.cdist(x, y)


# --- Exercise 7: Advanced Indexing — Gather and Scatter -----------------------
# "Given node features and an edge list, gather source/target node features
#  for all edges, then scatter messages back to nodes."

def gather_scatter_demo(
    node_features: torch.Tensor,  # (num_nodes, feature_dim)
    edge_index: torch.Tensor,      # (2, num_edges) — [src; tgt]
) -> torch.Tensor:
    """
    GNN-style gather-scatter: collect neighbor info, aggregate per node.

    This is the fundamental operation behind ALL graph neural networks.
    Understanding indexing here is critical for mesh-based simulation ML.
    """
    src, tgt = edge_index[0], edge_index[1]
    num_nodes = node_features.shape[0]

    # GATHER: get features for each edge's source and target
    src_feat = node_features[src]  # (num_edges, feature_dim)
    tgt_feat = node_features[tgt]  # (num_edges, feature_dim)

    # Compute messages (simple example: difference of features)
    messages = src_feat - tgt_feat  # (num_edges, feature_dim)

    # SCATTER: aggregate messages back to target nodes (sum)
    aggregated = torch.zeros_like(node_features)
    aggregated.index_add_(0, tgt, messages)

    # Alternative: scatter_add from torch_scatter (PyG) or torch.scatter_add
    # aggregated = torch.zeros_like(node_features)
    # aggregated.scatter_add_(0, tgt.unsqueeze(1).expand_as(messages), messages)

    return aggregated


# --- Exercise 8: Masking & Boolean Indexing -----------------------------------
# "Filter simulation data: keep only samples where the error is below a threshold
#  and the frequency is within a range."

def filter_simulation_data(
    errors: torch.Tensor,       # (N,)
    frequencies: torch.Tensor,  # (N,)
    fields: torch.Tensor,       # (N, H, W) — field data
    max_error: float = 0.1,
    freq_range: Tuple[float, float] = (1e9, 10e9),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter data using boolean masks. Returns filtered errors, freqs, fields.

    Key concepts:
    - Boolean indexing: tensor[mask] returns elements where mask is True
    - Combining masks with & (and), | (or), ~ (not)
    - This is vectorized — no Python loops needed
    """
    # Build boolean masks
    error_mask = errors < max_error
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])

    # Combine masks
    valid_mask = error_mask & freq_mask

    # Apply mask to all tensors consistently
    return errors[valid_mask], frequencies[valid_mask], fields[valid_mask]


# --- Exercise 9: Reshaping & Einsum ------------------------------------------
# "Given a batch of 2D field predictions, compute the relative L2 error
#  per sample (not averaged across the batch)."

def per_sample_relative_l2(
    pred: torch.Tensor,   # (batch, H, W)
    target: torch.Tensor,  # (batch, H, W)
) -> torch.Tensor:
    """
    Compute relative L2 error for each sample: ||pred - target|| / ||target||

    Returns shape (batch,) — one error value per sample.

    Key concept: Use .reshape(batch, -1) to flatten spatial dims,
    then reduce over the flattened dim only (not the batch dim).
    """
    batch = pred.shape[0]
    # Flatten spatial dimensions
    p = pred.reshape(batch, -1)   # (batch, H*W)
    t = target.reshape(batch, -1)  # (batch, H*W)

    # L2 norm along the spatial dimension (dim=1), keep batch dim
    diff_norm = torch.norm(p - t, dim=1)    # (batch,)
    target_norm = torch.norm(t, dim=1)       # (batch,)

    # Avoid division by zero
    return diff_norm / target_norm.clamp(min=1e-8)


# --- Exercise 10: Efficient Batch Operations ----------------------------------
# "Normalize each sample in a batch independently (zero mean, unit variance)
#  along the spatial dimensions."

def per_sample_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each sample to zero mean and unit variance.

    Input:  (batch, channels, H, W)
    Output: (batch, channels, H, W) — each sample independently normalized

    This is InstanceNorm without learnable params.
    Useful for simulation data where each sample has different magnitude.
    """
    # Compute mean and std over spatial dims (H, W), keep batch and channel
    mean = x.mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
    std = x.std(dim=(-2, -1), keepdim=True)    # (B, C, 1, 1)
    return (x - mean) / std.clamp(min=1e-8)


# ==============================================================================
# PART 3: PYTORCH PRACTICAL PATTERNS
# ==============================================================================
# Building real ML pipelines — what you'd actually do on the job.
# ==============================================================================


# --- Exercise 11: Custom Dataset for Simulation Data --------------------------
# "Write a PyTorch Dataset for loading simulation field data from disk.
#  Each sample has different spatial resolution (variable-size meshes)."

class SimulationDataset(Dataset):
    """
    Custom Dataset for simulation field data stored as .npz files.

    Each file contains:
      - 'params': input parameters (e.g., geometry, frequency), shape (P,)
      - 'field':  output field values, shape (N_i, F) — N_i varies per sample
      - 'coords': node coordinates, shape (N_i, D)

    Key design decisions:
      1. Lazy loading: don't load all data into RAM at init
      2. Handle variable sizes: each mesh has different N_i
      3. Return dict (not tuple) for clarity
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Only store file paths at init — lazy loading
        self.file_paths = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.file_paths[idx])

        sample = {
            "params": torch.tensor(data["params"], dtype=torch.float32),
            "coords": torch.tensor(data["coords"], dtype=torch.float32),
            "field": torch.tensor(data["field"], dtype=torch.float32),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_variable_size(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-size simulation data.

    Since each sample has different number of nodes N_i, we can't just stack.
    Options:
      1. Pad to max size (shown here) — simple, works with standard PyTorch
      2. Concatenate with batch index (PyG-style) — more memory efficient
      3. Use nested tensors (PyTorch 2.0+)
    """
    # Fixed-size: stack normally
    params = torch.stack([s["params"] for s in batch])  # (B, P)

    # Variable-size: pad to max
    max_nodes = max(s["coords"].shape[0] for s in batch)
    coord_dim = batch[0]["coords"].shape[1]
    field_dim = batch[0]["field"].shape[1]

    padded_coords = torch.zeros(len(batch), max_nodes, coord_dim)
    padded_fields = torch.zeros(len(batch), max_nodes, field_dim)
    masks = torch.zeros(len(batch), max_nodes, dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s["coords"].shape[0]
        padded_coords[i, :n] = s["coords"]
        padded_fields[i, :n] = s["field"]
        masks[i, :n] = True  # True = valid node, False = padding

    return {
        "params": params,
        "coords": padded_coords,
        "field": padded_fields,
        "mask": masks,  # CRITICAL: loss/metrics must use this mask!
    }


# Usage:
# loader = DataLoader(dataset, batch_size=8, collate_fn=collate_variable_size)


# --- Exercise 12: Complete Training Loop with Best Practices ------------------
# "Write a training function with validation, early stopping, checkpointing,
#  and proper device management."

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "best_model.pt",
    device: str = "cuda",
):
    """
    Production-quality training loop with all standard practices.

    Features:
    - AdamW optimizer with weight decay
    - Cosine annealing LR schedule
    - Gradient clipping
    - Early stopping on validation loss
    - Model checkpointing (save best)
    - Proper train/eval mode switching
    - Mixed precision training (AMP)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda")  # For mixed precision

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # ---- TRAINING ----
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move data to device
            x = batch["input"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = F.mse_loss(pred, y)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()  # .item() to avoid memory leak!
            num_batches += 1

        train_loss /= num_batches
        scheduler.step()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():  # CRITICAL: save memory, prevent gradient leakage
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)

                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    loss = F.mse_loss(pred, y)

                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= num_val_batches

        # ---- EARLY STOPPING & CHECKPOINTING ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Best: {best_val_loss:.6f}"
            )

    # Load best model
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# --- Exercise 13: Model Definition with Flexible Architecture ----------------
# "Write a simple but flexible MLP that could serve as a surrogate model."

class SurrogateMLP(nn.Module):
    """
    Flexible MLP for surrogate modeling: params → field values.

    Demonstrates:
    - Dynamic layer construction from config
    - Residual connections (optional)
    - Multiple normalization options
    - Proper weight initialization
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "gelu",
        norm: str = "layer",
        dropout: float = 0.0,
    ):
        super().__init__()

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        norm_fn = {
            "layer": nn.LayerNorm,
            "batch": nn.BatchNorm1d,
            "none": lambda d: nn.Identity(),
        }[norm]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                norm_fn(h_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim)

        # Initialize weights (Kaiming for ReLU/GELU, Xavier for others)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, output_dim)"""
        return self.head(self.backbone(x))


# --- Exercise 14: Learning Rate Finder (Practical ML Tool) --------------------
# "Implement a simple LR range test to find the optimal learning rate."

@torch.no_grad()
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.

    Useful for: reporting in papers, estimating memory, comparing architectures.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_MB": total * 4 / (1024 ** 2),  # Assuming float32
    }


# --- Exercise 15: Inference with Uncertainty (Practical for Simulation) -------
# "Add prediction uncertainty using MC-Dropout. This is critical for simulation
#  surrogates — we need to know WHEN the model is uncertain."

def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Runs N forward passes with dropout ENABLED at inference time.
    The variance across predictions estimates epistemic uncertainty.

    Args:
        model: Model with Dropout layers (must have dropout > 0)
        x: Input tensor, shape (batch, ...)
        n_samples: Number of MC samples (30 is usually sufficient)

    Returns:
        mean: Mean prediction, shape same as model output
        std:  Standard deviation (uncertainty), same shape

    Why this matters for Synopsys:
    - A surrogate model that says "I don't know" is safer than one that
      silently gives wrong answers. High uncertainty → fall back to HFSS.
    """
    model.train()  # Keep dropout ON (this is the key trick!)
    x = x.to(device)

    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():  # Still no gradients needed
            pred = model(x)
        predictions.append(pred)

    predictions = torch.stack(predictions)  # (n_samples, batch, ...)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    model.eval()  # Restore to eval mode
    return mean, std


# ==============================================================================
# PART 4: BUG SPOTTING — COMMON PYTORCH GOTCHAS
# ==============================================================================
# "Can you spot the bug?" — quick-fire questions an interviewer might ask.
# ==============================================================================

GOTCHAS = """
================================================================================
PART 4: 10 PYTORCH GOTCHAS — "SPOT THE BUG"
================================================================================

━━━ GOTCHA 1: model.eval() does NOT disable gradients ━━━

BUG:
    model.eval()
    output = model(x)  # Still tracking gradients! Wastes memory.

FIX:
    model.eval()
    with torch.no_grad():
        output = model(x)

WHY: model.eval() only changes Dropout and BatchNorm behavior.
     torch.no_grad() disables autograd for memory/speed savings.

━━━ GOTCHA 2: In-place operations break autograd ━━━

BUG:
    x = torch.randn(3, requires_grad=True)
    x += 1            # In-place! Destroys grad graph.
    loss = x.sum()
    loss.backward()   # RuntimeError

FIX:
    y = x + 1         # New tensor — graph intact.

RULE: Any operation ending in _ (add_, mul_, zero_) is in-place.
      Never use on tensors that need gradients.

━━━ GOTCHA 3: Device mismatch ━━━

BUG:
    model = model.cuda()
    x = torch.randn(4, 10)   # CPU!
    output = model(x)         # RuntimeError

FIX:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)

TIP: Inside a model, create new tensors on the same device as input:
     mask = torch.ones(n, device=x.device)

━━━ GOTCHA 4: Forgetting model.train() after validation ━━━

BUG:
    model.eval()
    val_loss = validate(model)
    # ... continue training with Dropout disabled, BatchNorm frozen ...

FIX:
    model.eval()
    with torch.no_grad():
        val_loss = validate(model)
    model.train()  # ALWAYS switch back!

━━━ GOTCHA 5: Gradient accumulation (forgetting zero_grad) ━━━

BUG:
    for batch in dataloader:
        loss = model(batch).sum()
        loss.backward()       # Gradients ACCUMULATE from prior steps!
        optimizer.step()

FIX:
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()

NOTE: Accumulation is actually useful for simulating larger batch sizes:
    for i, batch in enumerate(dataloader):
        loss = model(batch).sum() / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

━━━ GOTCHA 6: Silent broadcasting bugs ━━━

BUG:
    pred = model(x)       # (batch, 10)
    target = get_target() # (batch,)
    loss = (pred - target) ** 2  # Broadcasts to (batch, 10) — WRONG!

FIX:
    target = target.unsqueeze(-1)  # (batch, 1) — explicit shape
    loss = (pred - target) ** 2

DEFENSE: Always assert shapes: assert pred.shape == target.shape

━━━ GOTCHA 7: Memory leak from storing loss tensors ━━━

BUG:
    losses = []
    for batch in dataloader:
        loss = model(batch).sum()
        losses.append(loss)  # Entire computation graph stays in memory!

FIX:
    losses.append(loss.item())   # .item() → Python float, graph freed
    # or
    losses.append(loss.detach()) # Tensor without grad history

━━━ GOTCHA 8: BatchNorm with batch_size=1 ━━━

BUG:
    model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
    x = torch.randn(1, 10)  # Single sample!
    model(x)                 # RuntimeError: variance is 0

FIX:
    Use LayerNorm (normalizes across features) or InstanceNorm.
    For simulation data with small batches, LayerNorm is usually better.

━━━ GOTCHA 9: DDP state_dict key mismatch ━━━

BUG:
    # Saved with DDP:   keys = "module.layer.weight"
    # Loaded without DDP: expects "layer.weight"

FIX (save correctly):
    torch.save(ddp_model.module.state_dict(), "model.pt")

FIX (load flexibly):
    state = torch.load("model.pt")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

━━━ GOTCHA 10: torch.Tensor vs torch.tensor ━━━

BUG:
    x = torch.Tensor(3)     # Allocates UNINITIALIZED tensor of size 3!
    y = torch.Tensor([1,2]) # Always float32, ignores input dtype

FIX:
    x = torch.tensor(3)     # Scalar tensor with value 3
    y = torch.tensor([1,2]) # Infers dtype (int64 here)

RULE: Always use lowercase torch.tensor() for creating from data.

================================================================================
BONUS: PATTERNS EVERY ML ENGINEER SHOULD KNOW
================================================================================

# 1. Proper training loop skeleton
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch.to(device)), target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

# 2. Proper evaluation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch.to(device))
    model.train()

# 3. Proper checkpointing
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
    }, 'checkpoint.pt')

    ckpt = torch.load('checkpoint.pt', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

# 4. Mixed precision (AMP)
    scaler = torch.amp.GradScaler("cuda")
    with torch.amp.autocast("cuda"):
        loss = model(x)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 5. Gradient accumulation
    for i, batch in enumerate(loader):
        loss = model(batch) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
"""


# ==============================================================================
# TESTS
# ==============================================================================

def test_part1():
    """Test Python fundamentals."""
    print("--- Part 1: Python Fundamentals ---")

    # Exercise 1
    results = [
        {"geometry": "dipole", "error": 0.05},
        {"geometry": "patch",  "error": 0.12},
        {"geometry": "dipole", "error": 0.03},
        {"geometry": "patch",  "error": 0.08},
    ]
    avg = group_and_average(results)
    assert abs(avg["dipole"] - 0.04) < 1e-10
    assert abs(avg["patch"] - 0.10) < 1e-10
    print("  [PASS] Exercise 1: group_and_average")

    # Exercise 3
    s_params = np.random.randn(10, 2, 2) + 1j * np.random.randn(10, 2, 2)
    freqs = np.linspace(1.0, 10.0, 10)
    result = SimulationResult("test_antenna", s_params, freqs)
    assert result.n_ports == 2
    assert 1.0 <= result.resonant_freq_ghz <= 10.0
    print(f"  [PASS] Exercise 3: SimulationResult — {result}")

    # Exercise 5
    config = {"model_type": "gnn", "learning_rate": 1e-3, "num_epochs": 50}
    clean = validate_config(config)
    assert clean["batch_size"] == 32  # Default filled in
    try:
        validate_config({"model_type": "invalid", "learning_rate": 1e-3, "num_epochs": 10})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [PASS] Exercise 5: validate_config")


def test_part2():
    """Test tensor operations."""
    print("--- Part 2: Tensor Operations ---")

    # Exercise 6
    x = torch.randn(5, 3)
    y = torch.randn(7, 3)
    dist = pairwise_distances(x, y)
    assert dist.shape == (5, 7)
    assert (dist >= 0).all()
    # Verify against torch.cdist
    expected = torch.cdist(x, y)
    assert torch.allclose(dist, expected, atol=1e-5)
    print("  [PASS] Exercise 6: pairwise_distances")

    # Exercise 7
    node_feat = torch.randn(4, 8)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    agg = gather_scatter_demo(node_feat, edges)
    assert agg.shape == (4, 8)
    print("  [PASS] Exercise 7: gather_scatter")

    # Exercise 8
    errors = torch.tensor([0.05, 0.15, 0.03, 0.20, 0.08])
    freqs = torch.tensor([2e9, 5e9, 3e9, 15e9, 8e9])
    fields = torch.randn(5, 4, 4)
    e, f, fd = filter_simulation_data(errors, freqs, fields)
    assert len(e) == 3  # 0.05, 0.03, 0.08 pass error; 0.03 and 0.08 pass freq
    print("  [PASS] Exercise 8: filter_simulation_data")

    # Exercise 9
    pred = torch.randn(4, 16, 16)
    target = torch.randn(4, 16, 16) + 5  # Offset so target_norm > 0
    rel_err = per_sample_relative_l2(pred, target)
    assert rel_err.shape == (4,)
    assert (rel_err >= 0).all()
    print("  [PASS] Exercise 9: per_sample_relative_l2")

    # Exercise 10
    x = torch.randn(2, 3, 8, 8) * 100 + 50  # Arbitrary scale
    normed = per_sample_normalize(x)
    # Check zero mean per sample/channel
    means = normed.mean(dim=(-2, -1))
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    print("  [PASS] Exercise 10: per_sample_normalize")


def test_part3():
    """Test PyTorch practical patterns."""
    print("--- Part 3: PyTorch Patterns ---")

    # Exercise 13
    model = SurrogateMLP(input_dim=10, output_dim=5, hidden_dims=[64, 64])
    x = torch.randn(4, 10)
    y = model(x)
    assert y.shape == (4, 5)
    y.sum().backward()
    params = count_parameters(model)
    print(f"  [PASS] Exercise 13: SurrogateMLP — {params['total']:,} params ({params['total_MB']:.2f} MB)")

    # Exercise 15
    model_with_dropout = SurrogateMLP(
        input_dim=10, output_dim=5, hidden_dims=[64, 64], dropout=0.1
    )
    mean, std = predict_with_uncertainty(model_with_dropout, x, n_samples=10)
    assert mean.shape == (4, 5)
    assert std.shape == (4, 5)
    assert (std >= 0).all()
    print("  [PASS] Exercise 15: MC-Dropout uncertainty")


if __name__ == "__main__":
    print("=" * 70)
    print("PYTHON & PYTORCH CODING REVIEW — Synopsys ML Internship")
    print("=" * 70)
    print()

    test_part1()
    print()
    test_part2()
    print()
    test_part3()
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print()
    print("To review gotchas, read the GOTCHAS string in this file,")
    print("or see pytorch_coding_prep.md for the full study guide.")
