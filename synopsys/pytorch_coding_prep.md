# Python & PyTorch Coding Preparation — Synopsys ML Internship

**Interview:** Monday, February 24, 2026, with Xin Xu (Principal R&D Engineer)
**Companion file:** `pytorch_coding_review.py` (runnable code exercises)

---

## What They'll Actually Test (and What They Won't)

### Likely Assessment Areas

| Area | Why | How They'd Test |
|------|-----|-----------------|
| **Python fluency** | You'll write Python every day | "Process this data," "Design this class," list/dict comprehensions |
| **Tensor manipulation** | Core of ML research engineering | "Compute X without loops," reshaping, broadcasting, indexing |
| **PyTorch training pipeline** | Can you build real experiments? | "Walk me through your training loop," "What's your Dataset look like?" |
| **Debugging instinct** | Saves weeks of wasted compute | "Why is my model not learning?" "Spot the bug in this code" |
| **Data handling** | Simulation data is messy | Variable-size meshes, normalization, collation, I/O |

### Unlikely (Over-prepared in old version)

| Area | Why Unlikely |
|------|-------------|
| Implement FNO layer from scratch | Too specialized — they'd explain their architecture |
| Implement SIREN from scratch | They'd give you a reference implementation |
| Implement diffusion training step | You'd discuss this verbally, not code it live |
| Implement GRU cell from scratch | PyTorch has `nn.GRU` — nobody reimplements this |

---

## Part 1: Python Quick Reference

### Data Structures — Know These Cold

```python
# List comprehension with condition
errors = [r["error"] for r in results if r["geometry"] == "dipole"]

# Dict comprehension
error_by_name = {r["name"]: r["error"] for r in results}

# defaultdict — avoid KeyError when grouping
from collections import defaultdict
groups = defaultdict(list)
for r in results:
    groups[r["type"]].append(r)

# Counter — frequency counting
from collections import Counter
type_counts = Counter(r["type"] for r in results)

# enumerate — when you need index + value
for i, item in enumerate(items):
    print(f"Processing {i}/{len(items)}: {item}")

# zip — iterate multiple sequences together
for name, error in zip(names, errors):
    print(f"{name}: {error:.4f}")

# sorted with key — custom sorting
results.sort(key=lambda r: r["error"])  # In-place, ascending
top_5 = sorted(results, key=lambda r: r["error"])[:5]
```

### Generators — Memory-Efficient Processing

```python
# Generator function — yields items one at a time
def batch_reader(filepath, batch_size=32):
    batch = []
    with open(filepath) as f:
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:  # Don't forget the last batch!
        yield batch

# Generator expression — like list comprehension but lazy
total = sum(r["error"] for r in results)  # No list created in memory

# WHY: Simulation datasets can be GBs. Generators process one chunk at a time.
```

### OOP Patterns — Clean Class Design

```python
class SimResult:
    def __init__(self, name, data):
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got {data.ndim}D")
        self.name = name
        self.data = data

    @property                    # Computed attribute (no parentheses to access)
    def mean_error(self):
        return float(self.data.mean())

    @classmethod                 # Alternative constructor
    def from_file(cls, path):
        data = np.load(path)
        return cls(name=path, data=data)

    def __repr__(self):          # For debugging: print(result) shows useful info
        return f"SimResult('{self.name}', shape={self.data.shape})"

    def __len__(self):           # len(result) works
        return self.data.shape[0]
```

### Decorators — Common Interview Topic

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)  # Preserves function name/docstring
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timer
def train_epoch(model, loader):
    ...
```

### Error Handling — Defensive Programming

```python
# Validate inputs early with clear messages
def train(config):
    if "learning_rate" not in config:
        raise ValueError("Missing 'learning_rate' in config")
    lr = config["learning_rate"]
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"learning_rate must be positive, got {lr}")

# Use .get() for optional keys with defaults
batch_size = config.get("batch_size", 32)

# try/except — catch specific exceptions
try:
    data = np.load(filepath)
except FileNotFoundError:
    print(f"Warning: {filepath} not found, skipping")
    return None
```

---

## Part 2: NumPy / Tensor Operations Quick Reference

### Shapes, Reshaping, Broadcasting

```python
x = torch.randn(8, 3, 64, 64)  # (batch, channels, H, W)

# Reshape: flatten spatial dims
x_flat = x.reshape(8, 3, -1)   # (8, 3, 4096) — -1 infers the size
x_flat = x.view(8, -1)         # (8, 12288) — flatten everything but batch

# Permute: swap dimensions (e.g., channels-last ↔ channels-first)
x_nhwc = x.permute(0, 2, 3, 1)  # (8, 64, 64, 3) — for visualization

# Unsqueeze / squeeze: add / remove dimensions of size 1
bias = torch.randn(3)           # (3,)
bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, 3, 1, 1)
# Now bias broadcasts with (8, 3, 64, 64)

# Broadcasting rules (right-aligned):
# (8, 3, 64, 64) + (3, 1, 1) → (8, 3, 64, 64)  ✓
# (8, 3, 64, 64) + (64,)     → ERROR             ✗ (ambiguous dim)
# (8, 3, 64, 64) + (64, 64)  → (8, 3, 64, 64)   ✓
```

### Indexing — The Most Important Skill

```python
# Basic indexing
x[0]          # First sample
x[:, 0]       # First channel of all samples
x[..., -1]    # Last column (... = fill remaining dims)

# Boolean masking
mask = errors < 0.1
filtered = data[mask]  # Only rows where error < 0.1

# Advanced indexing (gather)
indices = torch.tensor([0, 2, 4])
selected = data[indices]  # Rows 0, 2, 4

# Scatter (inverse of gather) — critical for GNNs
aggregated = torch.zeros(num_nodes, dim)
aggregated.index_add_(0, target_indices, messages)
# aggregated[target_indices[i]] += messages[i]
```

### Common Operations

```python
# Pairwise distances (don't write a loop!)
dist = torch.cdist(x, y)          # (N, M) pairwise L2

# Or manually: ||a-b||² = ||a||² + ||b||² - 2a·b
xx = (x**2).sum(-1, keepdim=True) # (N, 1)
yy = (y**2).sum(-1, keepdim=True).T  # (1, M)
dist = torch.sqrt((xx + yy - 2 * x @ y.T).clamp(min=0))

# Relative L2 error per sample
diff = (pred - target).reshape(batch, -1)
rel_err = diff.norm(dim=1) / target.reshape(batch, -1).norm(dim=1).clamp(min=1e-8)

# Normalize per sample (InstanceNorm without learnable params)
mean = x.mean(dim=(-2, -1), keepdim=True)
std = x.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
x_normed = (x - mean) / std

# Einsum — readable tensor contractions
# Matrix multiply: C = A @ B
C = torch.einsum("ij,jk->ik", A, B)
# Batch matrix multiply
C = torch.einsum("bij,bjk->bik", A, B)
# Dot product per sample
dots = torch.einsum("bi,bi->b", x, y)
```

---

## Part 3: PyTorch Practical Patterns

### 3.1 Custom Dataset

```python
class SimulationDataset(Dataset):
    """Key points an interviewer wants to see:
    1. Lazy loading (don't load everything in __init__)
    2. Returns dict (not tuple) for clarity
    3. Handle variable sizes
    """
    def __init__(self, data_dir, transform=None):
        self.files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        sample = {
            "params": torch.tensor(data["params"], dtype=torch.float32),
            "field": torch.tensor(data["field"], dtype=torch.float32),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
```

**Handling variable-size data (meshes with different node counts):**

```python
def collate_variable_size(batch):
    """Pad variable-size tensors to max size in batch."""
    params = torch.stack([s["params"] for s in batch])

    max_nodes = max(s["field"].shape[0] for s in batch)
    padded = torch.zeros(len(batch), max_nodes, batch[0]["field"].shape[1])
    masks = torch.zeros(len(batch), max_nodes, dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s["field"].shape[0]
        padded[i, :n] = s["field"]
        masks[i, :n] = True  # True = valid, False = padding

    return {"params": params, "field": padded, "mask": masks}

# Usage:
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_variable_size)
```

### 3.2 Complete Training Loop

```python
def train(model, train_loader, val_loader, epochs=100, lr=1e-3, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch["input"].to(device), batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()  # .item() to avoid memory leak!

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["input"].to(device), batch["target"].to(device)
                val_loss += F.mse_loss(model(x), y).item()

        # --- Checkpoint ---
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best.pt")

    model.load_state_dict(torch.load("best.pt", weights_only=True))
    return model
```

### 3.3 Model Definition

```python
class SurrogateMLP(nn.Module):
    """Flexible MLP — the kind you'd actually write at work."""
    def __init__(self, in_dim, out_dim, hidden=[256, 256], dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))
```

### 3.4 Saving & Loading Models

```python
# --- SAVE (include everything needed to resume) ---
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_loss": best_loss,
    "config": config,  # So you know how to reconstruct the model
}, "checkpoint.pt")

# --- LOAD ---
ckpt = torch.load("checkpoint.pt", weights_only=True)
model = SurrogateMLP(**ckpt["config"])  # Reconstruct from config
model.load_state_dict(ckpt["model_state_dict"])

# --- LOAD DDP model into non-DDP ---
state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(state)
```

### 3.5 MC-Dropout for Uncertainty

```python
def predict_with_uncertainty(model, x, n_samples=30):
    """
    Keep dropout ON at inference → multiple stochastic forward passes.
    Variance across predictions ≈ epistemic uncertainty.

    High uncertainty → fall back to full HFSS simulation.
    """
    model.train()  # Keeps dropout active (the key trick)
    preds = torch.stack([model(x).detach() for _ in range(n_samples)])
    model.eval()
    return preds.mean(0), preds.std(0)
```

---

## Part 4: 10 PyTorch Gotchas — Quick Review

| # | Bug | Fix | One-liner |
|---|-----|-----|-----------|
| 1 | `model.eval()` alone | Add `with torch.no_grad():` | eval() changes Dropout/BN, no_grad() saves memory |
| 2 | `x += 1` on grad tensor | `y = x + 1` (new tensor) | In-place ops (`_` suffix) break autograd |
| 3 | Model on GPU, data on CPU | `.to(device)` everything | Define `device` once, use it everywhere |
| 4 | Forget `model.train()` after val | Always switch back | Dropout stays off, BN uses running stats |
| 5 | Missing `zero_grad()` | Call before each backward | Gradients accumulate by default |
| 6 | `(B,10) - (B,)` broadcasts wrong | `unsqueeze(-1)` to `(B,1)` | Always assert shapes match |
| 7 | `losses.append(loss)` leaks memory | Use `loss.item()` | Tensor retains entire computation graph |
| 8 | BatchNorm with batch=1 | Use LayerNorm instead | BN needs variance across batch dim |
| 9 | DDP adds `module.` prefix | Save `model.module.state_dict()` | Strip prefix when loading |
| 10 | `torch.Tensor(3)` | Use `torch.tensor(3)` | Capital-T creates uninitialized tensor of size 3 |

---

## Part 5: Questions They Might Ask Verbally

### "How would you handle variable-size simulation meshes in a dataset?"

> "Three approaches. **Padding**: pad to the max size in each batch with a boolean mask — simple, works with standard PyTorch, but wastes memory on small samples. **Concatenation**: concatenate all nodes into one big graph with a batch index vector — this is what PyTorch Geometric does, memory-efficient but needs custom code. **Interpolation**: resample everything to a fixed grid — enables standard CNNs/FNO but loses mesh resolution. I'd default to padding with masks for prototyping, then switch to PyG-style concatenation if memory is a bottleneck."

### "Your model trains fine but gives bad predictions on new geometries. What do you try?"

> "I'd diagnose in order:
> 1. **Visualize failures** — are they random or systematic? If only certain geometry types fail, it's a distribution gap.
> 2. **Check normalization** — are input coordinates and field values normalized per-sample? Absolute coordinates break generalization.
> 3. **Use relative features** — edge-based features (x_i - x_j) instead of absolute positions. This gives spatial equivariance and is what MeshGraphNets does.
> 4. **Add more diverse training data** — possibly with augmentation (rotations, reflections) if the physics has those symmetries.
> 5. **Add uncertainty estimation** — MC-Dropout or ensembles to flag when the model is outside its training distribution."

### "Walk me through how you'd set up an experiment for this internship."

> "Step 1: **Data pipeline** — understand the HFSS output format, write a Dataset class, establish train/val/test splits stratified by geometry type. Step 2: **Baseline** — simplest possible model (small MLP) to validate the pipeline end-to-end, establish that learning is happening. Step 3: **Iterate** — try architectures matched to the data structure (GNN for meshes, FNO for regular grids), track experiments with a config file and logging. Step 4: **Evaluate** — not just MSE, but physics-informed metrics (PDE residual, S-parameter accuracy) and out-of-distribution tests. Step 5: **Document** — clear figures, reproducible configs, written summary for the team."

### "How would you add uncertainty estimation to a surrogate model?"

> "Three options in increasing complexity:
> 1. **MC-Dropout** — keep dropout on at inference, run N forward passes, use variance as uncertainty. Cheapest to implement, works with any model that has dropout.
> 2. **Deep Ensemble** — train 3-5 models with different random seeds, use prediction disagreement. More compute but better calibrated than MC-Dropout.
> 3. **Evidential / heteroscedastic output** — model predicts both mean AND variance directly. Single forward pass at inference but needs careful loss design (negative log-likelihood).
>
> For a simulation surrogate at Synopsys, I'd start with MC-Dropout and use it to trigger HFSS fallback when uncertainty exceeds a threshold. This is the approach I recommended in my JESTIE paper — knowing when the model doesn't know is as important as the prediction itself."

### "Explain distributed training in PyTorch."

> "The standard approach is **DistributedDataParallel (DDP)**. Each GPU gets a full model copy and processes different data. After each backward pass, gradients are all-reduced (averaged) across GPUs using NCCL, then each GPU takes the same optimizer step. Key implementation details: use `DistributedSampler` to partition data, call `sampler.set_epoch(epoch)` for proper shuffling, save with `model.module.state_dict()` to strip the DDP wrapper, and scale the learning rate linearly with the number of GPUs. Launch with `torchrun --nproc_per_node=N train.py`."

---

## Part 6: Quick Verbal Python Concepts

If asked "explain X in Python":

| Concept | 15-Second Answer |
|---------|-----------------|
| **List vs Tuple** | Lists are mutable (append, modify), tuples are immutable (hashable, can be dict keys). Use tuples for fixed collections, lists for dynamic ones. |
| **`*args, **kwargs`** | `*args` collects positional args as a tuple, `**kwargs` collects keyword args as a dict. Used for flexible function signatures and forwarding arguments. |
| **Generator vs List** | Generator yields items lazily (one at a time, low memory). List stores everything in RAM. Use generators for large datasets. |
| **`@property`** | Makes a method accessible like an attribute (no parentheses). Useful for computed/derived values that look like data. |
| **`@classmethod`** | Alternative constructor. Gets `cls` (the class) as first arg instead of `self`. E.g., `SimResult.from_file(path)`. |
| **`@staticmethod`** | Function that lives in a class namespace but doesn't access `self` or `cls`. Rarely needed — usually just use a module-level function. |
| **`__repr__` vs `__str__`** | `__repr__` is for developers (unambiguous, shows class name). `__str__` is for end users (pretty printing). |
| **Context manager (`with`)** | Guarantees cleanup (file.close(), lock.release()). Implement with `__enter__`/`__exit__` or `@contextmanager`. |
| **GIL** | Global Interpreter Lock — only one Python thread runs at a time. Use multiprocessing (not threading) for CPU parallelism. PyTorch DataLoader uses multiprocessing via `num_workers`. |
| **`is` vs `==`** | `is` checks identity (same object in memory), `==` checks equality (same value). Use `is` only for `None`: `if x is None`. |

---

## Study Schedule

| When | What | Time |
|------|------|------|
| **Day 1** | Read through `pytorch_coding_review.py`, run it, make sure you understand every line | 2 hrs |
| **Day 2** | Practice the verbal questions out loud (Part 5 above). Time yourself — each answer should be 60-90 seconds. | 1.5 hrs |
| **Day 3** | Review the 10 gotchas. For each one, can you explain WHY it's a bug without looking? | 1 hr |
| **Day 4** | Light review: skim the quick reference tables. Do NOT study new material. | 30 min |

---

*This prep covers the practical Python/PyTorch skills a Principal R&D Engineer would assess. The deep ML architecture knowledge is covered in `synopsys_final_prep.md`.*
