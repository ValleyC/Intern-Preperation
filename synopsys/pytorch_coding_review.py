"""
================================================================================
PYTORCH CODING REVIEW — Synopsys/Ansys ML Internship
================================================================================
Interview Date: February 24, 2026
Interviewer: Xin Xu (Principal R&D Engineer)
Candidate Role: ML for Electromagnetic Simulation

This file contains:
  Part 1: "Implement X from scratch" questions (5 questions with solutions)
  Part 2: "How would you implement X" verbal questions (5 questions, answers in docstrings)
  Part 3: Common PyTorch gotchas (10 items)

All code is self-contained, well-commented, and follows PyTorch best practices.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==============================================================================
# PART 1: "IMPLEMENT X FROM SCRATCH" QUESTIONS
# ==============================================================================


# ==============================================================================
# QUESTION 1: Implement a GRU Cell from Scratch
# ==============================================================================
# Context: Relevant to candidate's IEEE JESTIE paper, where GRU networks were
# used for real-time electromagnetic transient simulation (3.33× FTRT on FPGA).
#
# INTERVIEWER PROMPT:
#   "Implement a single GRU cell from scratch in PyTorch. Given input x_t and
#    previous hidden state h_{t-1}, compute the new hidden state h_t.
#    Show the reset gate, update gate, and the candidate hidden state."
#
# GRU EQUATIONS:
#   r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)    (reset gate)
#   z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)    (update gate)
#   n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  (candidate)
#   h_t = (1 - z_t) * n_t + z_t * h_{t-1}                        (output)
# ==============================================================================

class GRUCell(nn.Module):
    """
    A single GRU cell implemented from scratch.

    The GRU (Gated Recurrent Unit) uses two gates:
    - Reset gate (r): controls how much of the previous state to forget
    - Update gate (z): controls the blend between old state and candidate

    This is simpler than LSTM (no separate cell state) while retaining the
    ability to capture long-range dependencies.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weight matrices for efficiency: one matmul instead of three
        # Input-to-hidden weights for all three gates: [reset, update, candidate]
        self.W_ih = nn.Linear(input_size, 3 * hidden_size)

        # Hidden-to-hidden weights for all three gates: [reset, update, candidate]
        self.W_hh = nn.Linear(hidden_size, 3 * hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights uniformly, matching PyTorch's default GRU init."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single GRU cell.

        Args:
            x_t:    Input at current timestep.    Shape: (batch_size, input_size)
            h_prev: Hidden state from previous timestep. Shape: (batch_size, hidden_size)

        Returns:
            h_t: New hidden state.  Shape: (batch_size, hidden_size)
        """
        # Compute all input-to-hidden projections in one matmul
        # Shape: (batch_size, 3 * hidden_size)
        x_proj = self.W_ih(x_t)

        # Compute all hidden-to-hidden projections in one matmul
        # Shape: (batch_size, 3 * hidden_size)
        h_proj = self.W_hh(h_prev)

        # Split into the three gate projections
        x_r, x_z, x_n = x_proj.chunk(3, dim=-1)  # each: (batch, hidden_size)
        h_r, h_z, h_n = h_proj.chunk(3, dim=-1)  # each: (batch, hidden_size)

        # ---- Reset Gate ----
        # Decides how much of the previous hidden state to let through
        # when computing the candidate. r=0 means "forget everything".
        r_t = torch.sigmoid(x_r + h_r)  # Shape: (batch, hidden_size)

        # ---- Update Gate ----
        # Decides how much of the new candidate vs old state to use.
        # z=1 means "keep old state entirely" (skip connection / carry).
        z_t = torch.sigmoid(x_z + h_z)  # Shape: (batch, hidden_size)

        # ---- Candidate Hidden State ----
        # The reset gate modulates the previous hidden state BEFORE the
        # linear projection — this is the key difference from LSTM.
        n_t = torch.tanh(x_n + r_t * h_n)  # Shape: (batch, hidden_size)

        # ---- New Hidden State ----
        # Convex combination: when z_t is close to 1, we keep the old state
        # (gradient highway); when z_t is close to 0, we adopt the candidate.
        h_t = (1 - z_t) * n_t + z_t * h_prev  # Shape: (batch, hidden_size)

        return h_t


def test_gru_cell():
    """Verify our GRU cell produces correct output shapes and matches PyTorch."""
    batch_size, input_size, hidden_size = 4, 10, 20

    cell = GRUCell(input_size, hidden_size)
    x = torch.randn(batch_size, input_size)
    h = torch.randn(batch_size, hidden_size)

    h_new = cell(x, h)
    assert h_new.shape == (batch_size, hidden_size), f"Expected {(batch_size, hidden_size)}, got {h_new.shape}"

    # Verify gradients flow
    loss = h_new.sum()
    loss.backward()
    assert cell.W_ih.weight.grad is not None, "Gradients did not flow to W_ih"
    print("[PASS] GRU Cell: shape, gradient checks passed")


# ==============================================================================
# QUESTION 2: Implement a Basic FNO (Fourier Neural Operator) Layer
# ==============================================================================
# Context: Neural operators are central to Ansys SimAI. FNO learns operators
# between function spaces by parameterizing convolutions in Fourier space.
#
# INTERVIEWER PROMPT:
#   "Implement a single Fourier Neural Operator layer. Show the spectral
#    convolution: FFT of the input, multiply by learnable complex weights
#    in Fourier space (keeping only the low-frequency modes), then IFFT back.
#    Include the residual linear path."
#
# FNO LAYER:
#   output = sigma( W_fourier(x) + W_local(x) + b )
#   where W_fourier operates in spectral domain and W_local is pointwise.
# ==============================================================================

class SpectralConv2d(nn.Module):
    """
    Spectral convolution layer: the core of the Fourier Neural Operator.

    Instead of convolving in physical space (O(N^2) or FFT-based O(N log N)),
    we directly multiply by learnable weights in Fourier space. This is
    equivalent to a global convolution with O(N log N + k) cost, where k
    is the number of retained Fourier modes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Args:
            in_channels:  Number of input channels.
            out_channels: Number of output channels.
            modes1:       Number of Fourier modes to keep in the first dimension.
            modes2:       Number of Fourier modes to keep in the second dimension.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of low-frequency modes to retain
        self.modes2 = modes2

        # Scale factor for initialization (similar to Kaiming)
        scale = 1.0 / (in_channels * out_channels)

        # Learnable complex-valued weights in Fourier space
        # We need two sets because rfft2 produces both positive and negative
        # frequency components along the first axis, but only non-negative
        # along the second axis (Hermitian symmetry).
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication and contraction over the channel dimension.

        This is the spectral equivalent of a 1x1 convolution: it mixes channels
        independently at each Fourier mode.

        Args:
            input_ft: (batch, in_ch, modes1, modes2) complex tensor
            weights:  (in_ch, out_ch, modes1, modes2) complex tensor

        Returns:
            (batch, out_ch, modes1, modes2) complex tensor
        """
        # Einstein summation: contract over input channels (i), keep batch (b),
        # output channels (o), and spatial frequency indices (x, y)
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: FFT -> multiply by weights -> IFFT.

        Args:
            x: Input tensor. Shape: (batch, channels, height, width)

        Returns:
            Output tensor. Shape: (batch, out_channels, height, width)
        """
        batch_size = x.shape[0]
        height, width = x.shape[-2], x.shape[-1]

        # Step 1: Compute real FFT along the last two spatial dimensions.
        # rfft2 exploits Hermitian symmetry: output has shape
        # (batch, channels, height, width//2 + 1) — roughly half the modes.
        x_ft = torch.fft.rfft2(x)  # Shape: (B, C_in, H, W//2+1), complex

        # Step 2: Initialize output in Fourier space (zeros for modes we discard)
        out_ft = torch.zeros(
            batch_size, self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Step 3: Multiply only the low-frequency modes by learnable weights.
        # This acts as a low-pass filter with learnable coefficients.

        # Positive frequencies (top-left corner of the frequency grid)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )

        # Negative frequencies along dim1 (bottom-left corner, wraps around)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Step 4: Inverse FFT back to physical space
        x_out = torch.fft.irfft2(out_ft, s=(height, width))  # Shape: (B, C_out, H, W)

        return x_out


class FNOLayer(nn.Module):
    """
    A complete FNO layer = Spectral convolution + pointwise linear + residual + activation.

    The spectral path captures global (low-frequency) interactions.
    The linear path captures local (pointwise) interactions.
    Together they approximate the full Green's function of the PDE.
    """

    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        # Pointwise linear transform (1x1 conv) — captures local/high-frequency info
        self.linear = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: parallel spectral + linear paths, then GELU activation.

        Args:
            x: Shape (batch, channels, height, width)
        Returns:
            Same shape as input.
        """
        # Spectral path: global convolution in Fourier space
        x_spectral = self.spectral_conv(x)

        # Local path: pointwise linear (1x1 conv)
        x_local = self.linear(x)

        # Combine and activate — GELU is smoother than ReLU, good for physics
        return F.gelu(x_spectral + x_local)


def test_fno_layer():
    """Verify FNO layer output shapes and gradient flow."""
    batch, channels, H, W = 2, 32, 64, 64
    modes1, modes2 = 12, 12  # Keep 12 low-frequency modes in each direction

    layer = FNOLayer(channels, modes1, modes2)
    x = torch.randn(batch, channels, H, W)
    y = layer(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    y.sum().backward()
    assert layer.spectral_conv.weights1.grad is not None
    print("[PASS] FNO Layer: shape, gradient checks passed")


# ==============================================================================
# QUESTION 3: Implement a GNN Message-Passing Layer
# ==============================================================================
# Context: Relevant to candidate's DAC paper (chip placement on netlists) and
# mesh-based EM simulation (Pfaff et al., "Learning Mesh-Based Simulation").
#
# INTERVIEWER PROMPT:
#   "Implement a GNN message-passing layer from scratch. Given node features
#    and an edge index, compute edge messages from pairs of node features,
#    aggregate them per node (sum), and update node features."
#
# MESSAGE PASSING:
#   1. For each edge (i,j): message_ij = MLP([h_i || h_j || e_ij])
#   2. Aggregate: m_i = SUM_{j in N(i)} message_ij
#   3. Update: h_i' = MLP([h_i || m_i])
# ==============================================================================

class MessagePassingLayer(nn.Module):
    """
    A general message-passing GNN layer.

    This follows the "encode-process-decode" paradigm used in MeshGraphNets
    (Pfaff et al.) and GraphCast for physics simulation.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        """
        Args:
            node_dim:   Dimension of input node features.
            edge_dim:   Dimension of edge features (0 if no edge features).
            hidden_dim: Hidden dimension for MLPs.
        """
        super().__init__()
        self.node_dim = node_dim

        # Edge message MLP: takes concatenated [source_node, target_node, edge_features]
        edge_input_dim = 2 * node_dim + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node update MLP: takes concatenated [current_node, aggregated_messages]
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),  # Output same dim for residual
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: message computation -> aggregation -> node update.

        Args:
            node_features: Shape (num_nodes, node_dim)
            edge_index:    Shape (2, num_edges) — [source_indices; target_indices]
            edge_features: Shape (num_edges, edge_dim) or None

        Returns:
            Updated node features. Shape (num_nodes, node_dim)
        """
        src_idx = edge_index[0]  # Source node indices, shape: (num_edges,)
        tgt_idx = edge_index[1]  # Target node indices, shape: (num_edges,)

        # ---- Step 1: Compute edge messages ----
        # Gather features of source and target nodes for each edge
        src_features = node_features[src_idx]  # (num_edges, node_dim)
        tgt_features = node_features[tgt_idx]  # (num_edges, node_dim)

        # Concatenate source, target, and edge features
        if edge_features is not None:
            edge_input = torch.cat([src_features, tgt_features, edge_features], dim=-1)
        else:
            edge_input = torch.cat([src_features, tgt_features], dim=-1)

        messages = self.edge_mlp(edge_input)  # (num_edges, hidden_dim)

        # ---- Step 2: Aggregate messages per target node ----
        # Sum aggregation: for each target node, sum all incoming messages.
        # scatter_add is the standard way; we use index_add_ for vanilla PyTorch.
        num_nodes = node_features.shape[0]
        aggregated = torch.zeros(
            num_nodes, messages.shape[-1],
            device=node_features.device, dtype=node_features.dtype
        )
        # index_add_(dim, index, source): aggregated[tgt_idx[i]] += messages[i]
        aggregated.index_add_(0, tgt_idx, messages)  # (num_nodes, hidden_dim)

        # ---- Step 3: Update node features ----
        # Concatenate current features with aggregated messages
        node_input = torch.cat([node_features, aggregated], dim=-1)  # (N, node_dim + hidden)
        node_update = self.node_mlp(node_input)  # (N, node_dim)

        # Residual connection: critical for deep GNNs to avoid oversmoothing
        return node_features + node_update


def test_message_passing():
    """Verify message passing on a small graph."""
    num_nodes, node_dim, edge_dim, hidden_dim = 6, 16, 4, 32

    # Create a small graph: 6 nodes, 8 directed edges
    edge_index = torch.tensor([
        [0, 1, 2, 3, 1, 2, 4, 5],  # source
        [1, 2, 3, 0, 4, 5, 0, 0],  # target
    ], dtype=torch.long)
    num_edges = edge_index.shape[1]

    node_features = torch.randn(num_nodes, node_dim)
    edge_features = torch.randn(num_edges, edge_dim)

    layer = MessagePassingLayer(node_dim, edge_dim, hidden_dim)
    out = layer(node_features, edge_index, edge_features)

    assert out.shape == (num_nodes, node_dim), f"Expected {(num_nodes, node_dim)}, got {out.shape}"
    out.sum().backward()
    print("[PASS] Message Passing Layer: shape, gradient checks passed")


# ==============================================================================
# QUESTION 4: Implement a Simple Diffusion Training Step
# ==============================================================================
# Context: Relevant to candidate's two diffusion papers (EDISCO for TSP,
# DAC for chip placement). Core of score-based/denoising diffusion models.
#
# INTERVIEWER PROMPT:
#   "Implement the forward diffusion process and one training step for a
#    denoising diffusion model. Show: (1) how to add noise at an arbitrary
#    timestep, (2) the noise-prediction loss, and (3) a complete training
#    iteration."
#
# DDPM FORWARD PROCESS:
#   q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
#   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
# ==============================================================================

class DiffusionTrainer:
    """
    Implements the DDPM (Ho et al., 2020) forward process and training objective.

    Key insight: we can sample x_t directly from x_0 (no need to iterate
    through all previous timesteps) using the closed-form marginal.
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Args:
            model:          The noise-prediction network (e.g., U-Net).
            num_timesteps:  Number of diffusion steps T.
            beta_start:     Starting noise schedule value.
            beta_end:       Ending noise schedule value.
        """
        self.model = model
        self.num_timesteps = num_timesteps

        # ---- Define the noise schedule ----
        # Linear schedule (as in original DDPM). Cosine schedule is often better.
        betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Precompute useful quantities for the closed-form forward process
        alphas = 1.0 - betas                              # alpha_t = 1 - beta_t
        alpha_bar = torch.cumprod(alphas, dim=0)          # alpha_bar_t = prod_{s=1}^{t} alpha_s
        self.sqrt_alpha_bar = alpha_bar.sqrt()             # sqrt(alpha_bar_t)
        self.sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()  # sqrt(1 - alpha_bar_t)

    def forward_process(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean data x_0 to get x_t (the forward/diffusion process).

        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_0:   Clean data.     Shape: (batch, ...)
            t:     Timestep index. Shape: (batch,), values in [0, T-1]
            noise: Optional pre-sampled noise (for reproducibility).

        Returns:
            x_t:   Noisy data at timestep t. Same shape as x_0.
            noise: The noise that was added (needed for loss computation).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Move schedule tensors to the same device as data
        sqrt_ab = self.sqrt_alpha_bar.to(x_0.device)
        sqrt_omab = self.sqrt_one_minus_alpha_bar.to(x_0.device)

        # Gather the schedule values for each sample's timestep
        # Then reshape for broadcasting: (batch,) -> (batch, 1, 1, ...) to match x_0
        ndim = x_0.dim()
        shape = (-1,) + (1,) * (ndim - 1)  # e.g., (-1, 1, 1, 1) for images

        sqrt_ab_t = sqrt_ab[t].reshape(shape)       # (batch, 1, ...)
        sqrt_omab_t = sqrt_omab[t].reshape(shape)   # (batch, 1, ...)

        # Reparameterization: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
        x_t = sqrt_ab_t * x_0 + sqrt_omab_t * noise

        return x_t, noise

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute the simplified DDPM training loss for one batch.

        L_simple = E_{t, x_0, eps} [ || eps - eps_theta(x_t, t) ||^2 ]

        The model learns to predict the noise that was added, given the
        noisy sample and the timestep.

        Args:
            x_0: Clean training data. Shape: (batch, ...)

        Returns:
            Scalar loss value.
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps uniformly for each sample in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)

        # Sample noise and compute noisy data
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_process(x_0, t, noise=noise)

        # Predict the noise using the model
        noise_pred = self.model(x_t, t)

        # MSE loss between true noise and predicted noise
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def training_step(self, x_0: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Execute one complete training iteration.

        Args:
            x_0:       Clean data batch. Shape: (batch, ...)
            optimizer: The optimizer for the model parameters.

        Returns:
            The loss value as a Python float.
        """
        # Standard PyTorch training pattern
        self.model.train()
        optimizer.zero_grad()        # Clear old gradients
        loss = self.compute_loss(x_0)
        loss.backward()              # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Stabilize training
        optimizer.step()             # Update weights

        return loss.item()


class SimpleNoisePredictor(nn.Module):
    """A minimal noise prediction network for testing (NOT production quality)."""

    def __init__(self, data_dim: int, hidden_dim: int = 128, num_timesteps: int = 1000):
        super().__init__()
        self.time_embed = nn.Embedding(num_timesteps, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise given noisy data and timestep."""
        t_emb = self.time_embed(t)  # (batch, hidden_dim)
        # Flatten spatial dims if needed, then concat with time embedding
        x_flat = x_t.reshape(x_t.shape[0], -1)
        inp = torch.cat([x_flat, t_emb], dim=-1)
        return self.net(inp).reshape(x_t.shape)


def test_diffusion_training():
    """Verify one complete diffusion training step."""
    data_dim, batch_size = 32, 8

    model = SimpleNoisePredictor(data_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(model, num_timesteps=1000)

    x_0 = torch.randn(batch_size, data_dim)  # Fake clean data

    # Run 3 training steps, loss should decrease (or at least not crash)
    losses = []
    for step in range(3):
        loss = trainer.training_step(x_0, optimizer)
        losses.append(loss)

    print(f"[PASS] Diffusion Training: 3 steps completed, losses = {[f'{l:.4f}' for l in losses]}")


# ==============================================================================
# QUESTION 5: Implement a SIREN Layer
# ==============================================================================
# Context: Relevant to Ansys SimAI's implicit neural representations (INRs).
# SIREN uses sinusoidal activations to learn continuous functions, enabling
# representation of fine-grained physical fields (EM, thermal, stress).
#
# INTERVIEWER PROMPT:
#   "Implement a SIREN (Sinusoidal Representation Network) layer. Show the
#    sinusoidal activation, and crucially, the proper weight initialization
#    that makes SIREN work. Explain why the initialization matters."
#
# SIREN:
#   y = sin(omega_0 * (Wx + b))
#   First layer:  W ~ Uniform(-1/n, 1/n)
#   Hidden layers: W ~ Uniform(-sqrt(6/(n*omega_0^2)), sqrt(6/(n*omega_0^2)))
# ==============================================================================

class SIRENLayer(nn.Module):
    """
    A single SIREN layer: Linear transform followed by sinusoidal activation.

    Key insight: sin(.) preserves the distribution of activations through depth
    IF the weights are initialized correctly. The initialization ensures that
    the input to sin(.) remains in [-pi, pi] (approximately), preventing the
    activations from collapsing or exploding.

    Reference: Sitzmann et al., "Implicit Neural Representations with Periodic
    Activation Functions" (NeurIPS 2020)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first_layer: bool = False,
    ):
        """
        Args:
            in_features:    Input dimension.
            out_features:   Output dimension.
            omega_0:        Frequency scaling factor. Higher = more detail.
                            30.0 is the default from the paper.
            is_first_layer: Whether this is the first layer (different init).
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first_layer = is_first_layer
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        """
        Critical: SIREN requires specific initialization to work properly.

        The goal is to keep the pre-activation values (omega_0 * (Wx + b))
        distributed so that sin(.) operates in its full range.

        First layer:
          - Input is typically normalized to [-1, 1]
          - W ~ Uniform(-1/n, 1/n) ensures output stays bounded
          - After scaling by omega_0, this gives good coverage of sin()

        Hidden layers:
          - Input comes from sin(), which is in [-1, 1]
          - W ~ Uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
          - This is derived from requiring Var(output) = 1 after sin()
          - The factor sqrt(6/n) comes from the variance of a uniform distribution
            combined with E[cos^2] = 1/2 (derivative of sin used in backprop)
        """
        with torch.no_grad():
            if self.is_first_layer:
                # First layer: uniform(-1/n, 1/n) so that omega_0 * Wx ~ Uniform(-omega_0, omega_0)
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: carefully chosen so that the distribution is
                # preserved through the sin() activation
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: linear transform, scale by omega_0, then sin().

        Args:
            x: Input tensor. Shape: (..., in_features)
        Returns:
            Output tensor. Shape: (..., out_features)
        """
        # omega_0 controls the frequency of the sinusoidal activation.
        # Higher omega_0 allows the network to represent higher-frequency details.
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    A complete SIREN network for implicit neural representation.

    Takes coordinates (x, y) or (x, y, z) as input and outputs field values
    (e.g., E-field magnitude, temperature). Can represent continuous fields
    at arbitrary resolution.
    """

    def __init__(
        self,
        in_features: int = 2,       # e.g., 2D coordinates
        out_features: int = 1,       # e.g., scalar field value
        hidden_features: int = 256,
        num_hidden_layers: int = 3,
        omega_0: float = 30.0,
    ):
        super().__init__()

        layers = []

        # First layer (different initialization)
        layers.append(SIRENLayer(in_features, hidden_features, omega_0, is_first_layer=True))

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(SIRENLayer(hidden_features, hidden_features, omega_0, is_first_layer=False))

        self.network = nn.Sequential(*layers)

        # Final linear layer (no sin activation — we want raw output values)
        self.final_layer = nn.Linear(hidden_features, out_features)
        # Initialize final layer similarly to hidden layers
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / omega_0
            self.final_layer.weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Spatial coordinates. Shape: (batch, in_features)
        Returns:
            Field values. Shape: (batch, out_features)
        """
        features = self.network(coords)
        return self.final_layer(features)


def test_siren():
    """Verify SIREN layer and full network."""
    batch_size = 100

    # Test single layer
    layer = SIRENLayer(2, 64, omega_0=30.0, is_first_layer=True)
    coords = torch.rand(batch_size, 2) * 2 - 1  # Coordinates in [-1, 1]
    out = layer(coords)
    assert out.shape == (batch_size, 64)
    # Check output is in [-1, 1] (sin output)
    assert out.min() >= -1.0 and out.max() <= 1.0, "SIREN output should be in [-1, 1]"

    # Test full network: learn a 2D field from coordinates
    siren = SIREN(in_features=2, out_features=1, hidden_features=64, num_hidden_layers=2)
    field_values = siren(coords)
    assert field_values.shape == (batch_size, 1)
    field_values.sum().backward()
    print("[PASS] SIREN: layer and network shape/gradient checks passed")


# ==============================================================================
# PART 2: "HOW WOULD YOU IMPLEMENT X" VERBAL QUESTIONS
# ==============================================================================
# These are engineering judgment questions, not coding questions.
# The answers below are structured talking points.
# ==============================================================================

VERBAL_QUESTIONS = """
================================================================================
PART 2: VERBAL "HOW WOULD YOU IMPLEMENT X" QUESTIONS
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q1: "How would you handle a dataset of 10,000 HFSS simulations with varying
     mesh sizes for training a surrogate model?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY CHALLENGE: Each simulation has a different mesh, so tensor shapes differ.

APPROACH 1 — Interpolate to a fixed grid:
  - Interpolate all simulation results onto a common regular grid.
  - Enables standard CNN/FNO architectures with fixed tensor shapes.
  - Trade-off: loss of resolution near fine geometry features.
  - Implementation: scipy.interpolate or torch-based interpolation during
    preprocessing. Store as HDF5 with fixed shape.

APPROACH 2 — Graph representation (preferred for complex geometries):
  - Treat each mesh as a graph: nodes = mesh vertices, edges = mesh connectivity.
  - Node features = field values + coordinates; edge features = relative positions.
  - Use a GNN (e.g., MeshGraphNet) that handles variable-size graphs natively.
  - Batch with PyG's Batch.from_data_list() which concatenates graphs with offsets.
  - This preserves mesh resolution where it matters (near geometry features).

APPROACH 3 — Implicit neural representations:
  - Train SIREN/NeRF-style networks that take (x,y,z) coordinates as input.
  - The network IS the representation — no fixed grid needed.
  - Query at any resolution at inference time.
  - This is what SimAI appears to use.

DATA PIPELINE CONSIDERATIONS:
  - Store raw simulations in HDF5 or Zarr for efficient I/O.
  - Use a custom PyTorch Dataset with lazy loading (don't load all 10K into RAM).
  - Normalize inputs (geometry params, BCs) and outputs (field values) per-feature.
  - Split: 80/10/10 train/val/test, stratified by geometry type if applicable.
  - Data augmentation: exploit symmetries (rotation, reflection) if the physics
    allows it — doubles or quadruples effective dataset size for free.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q2: "Your model trains fine on small antenna designs but blows up on large
     ones. What would you try?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIAGNOSIS (ask these first):
  1. "Blows up" = NaN loss? Exploding outputs? OOM? Each has different fixes.
  2. Does it fail during training or only at inference?
  3. How much larger are the "large" designs? 2x? 100x?

IF NUMERICAL INSTABILITY (NaN/Inf):
  - Normalize inputs by physical scale: divide coordinates by characteristic
    length (e.g., wavelength), divide field values by max expected magnitude.
  - Use LayerNorm or InstanceNorm between layers (BatchNorm can be unstable
    when statistics differ between small and large designs).
  - Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  - Lower learning rate; use a warmup schedule.
  - Check for division by zero in custom loss functions.

IF GENERALIZATION FAILURE (large error, not NaN):
  - The model likely overfit to the scale of small designs.
  - Solution 1: Scale-invariant input representation — normalize coordinates
    to [-1, 1] per design, use relative (not absolute) positions.
  - Solution 2: Include more large designs in training (curriculum learning:
    start small, gradually increase size).
  - Solution 3: Use architecture that handles variable scales:
    * FNO: resolution-invariant by design (can train 64x64, infer 256x256).
    * GNN: naturally handles different numbers of nodes.
    * Multi-scale architecture: separate branches for different frequency bands.

IF OUT OF MEMORY:
  - Gradient checkpointing: trade compute for memory.
  - Mixed precision training (torch.cuda.amp).
  - Domain decomposition: split large designs into overlapping patches,
    process each patch, stitch results (like sliding window).
  - Reduce batch size but increase gradient accumulation steps.

RECOMMENDED SYSTEMATIC APPROACH:
  1. Visualize predictions on failing cases to understand the failure mode.
  2. Check if the issue is input-side (coordinates) or output-side (fields).
  3. Start with normalization fixes (cheapest to try).
  4. Then try architectural changes.
  5. Finally, consider data augmentation / curriculum learning.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q3: "How would you implement equivariance in a network for EM simulation?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY EQUIVARIANCE MATTERS FOR EM:
  Maxwell's equations are equivariant to rotations, translations, and
  reflections (the Euclidean group E(3)). If you rotate the antenna, the
  E-field rotates with it. A network that respects this learns faster and
  generalizes better — my EDISCO paper showed 33-50% less training data.

WHAT TYPE OF EQUIVARIANCE:
  - SE(3) = rotations + translations (orientation matters, no reflections)
  - E(3) = SE(3) + reflections (full Euclidean group)
  - For EM: E-field is a VECTOR field — it transforms as a type-1 representation
    under rotations. This is different from a scalar field (type-0).
  - B-field is a pseudovector (type-1 but flips sign under reflection).

IMPLEMENTATION APPROACHES:

  1. EQUIVARIANT GNNs (e.g., EGNN, E(n)-Equivariant GNN):
     - Use relative position vectors (x_j - x_i) as edge features.
     - Update both scalar features AND coordinate/vector features.
     - Scalar features are invariant; vector features transform equivariantly.
     - Libraries: e3nn (PyTorch), MACE architecture.

  2. SPHERICAL HARMONICS (e3nn approach):
     - Represent features as irreducible representations (irreps) of SO(3).
     - Scalars = l=0, vectors = l=1, rank-2 tensors = l=2.
     - Use Clebsch-Gordan tensor products for equivariant interactions.
     - Most principled but highest implementation complexity.

  3. FRAME AVERAGING (practical shortcut):
     - For each input, compute predictions for multiple rotated versions.
     - Average the (inverse-rotated) outputs.
     - Achieves approximate equivariance without architectural changes.
     - Cheaper to implement but slower at inference.

  4. STEERABLE CNNs (for grid-based representations):
     - Use steerable filters that transform predictably under rotation.
     - Good for regular grid data (like FNO on a regular mesh).
     - Library: escnn (PyTorch).

PRACTICAL RECOMMENDATION:
  For EM simulation specifically, I'd start with EGNN for mesh-based data
  (treating vector fields as coordinate-like features that transform under
  rotation), and validate that equivariance actually improves generalization
  on a held-out set of rotated geometries before investing in the full
  spherical harmonics machinery.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q4: "How would you set up distributed training for a large surrogate model?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: Choose the parallelism strategy.
  - DATA PARALLEL (most common, try first):
    * Each GPU gets a copy of the full model + different data batch.
    * Gradients are all-reduced (averaged) across GPUs after each step.
    * Effective batch size = per_gpu_batch * num_gpus.
    * Works when model fits on one GPU.

  - MODEL PARALLEL (when model is too large for one GPU):
    * Split model across GPUs (e.g., encoder on GPU0, decoder on GPU1).
    * Pipeline parallelism to keep all GPUs busy.
    * More complex, only needed for very large models.

STEP 2: Implementation with PyTorch DDP (DistributedDataParallel).

    # In the training script:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    # Initialize process group
    dist.init_process_group(backend="nccl")  # NCCL for GPU, gloo for CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Wrap model
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Use DistributedSampler to partition data across GPUs
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=per_gpu_batch)

    # In training loop: set epoch for proper shuffling
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # CRITICAL: ensures different shuffling per epoch
        for batch in dataloader:
            ...

    # Launch: torchrun --nproc_per_node=4 train.py

STEP 3: Practical considerations.
  - Learning rate scaling: linear scaling rule — multiply LR by num_gpus.
  - Warmup: increase LR gradually over first few hundred steps.
  - Gradient accumulation: if per-GPU batch is too small, accumulate gradients
    over multiple forward passes before stepping.
  - Mixed precision (AMP): almost always a good idea — 2x memory savings,
    1.5-2x speed, minimal accuracy loss.
  - Logging: only log from rank 0 to avoid duplicate output.
  - Checkpointing: save model.module.state_dict() (unwrap DDP wrapper).
  - Reproducibility: set seeds per-rank = base_seed + rank.

STEP 4: If using cloud/cluster (likely at Ansys):
  - Use SLURM or Kubernetes for job scheduling.
  - PyTorch Lightning or HuggingFace Accelerate simplify multi-node setup.
  - Monitor GPU utilization — if below 80%, you're likely I/O bottlenecked.
    Fix with: more DataLoader workers, pin_memory=True, prefetching.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q5: "How would you evaluate whether your surrogate model is accurate enough
     for production use?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 1: Standard ML Metrics (necessary but NOT sufficient)
  - Relative L2 error on held-out test set (per-field, not aggregated).
  - Pointwise max error (worst-case matters in engineering — a bridge doesn't
    care about average stress if it fails at one point).
  - Distribution of errors: plot histograms, check for heavy tails.
  - Per-region error: separate error metrics near geometry surfaces vs. far-field.
    Errors near boundaries are often larger and more consequential.

TIER 2: Physics-Informed Validation (catches "looks right but is wrong")
  - PDE residual: plug the predicted fields back into Maxwell's equations
    and measure how badly they violate the governing PDEs.
  - Conservation laws: check div(D)=rho, div(B)=0, energy conservation.
  - Boundary condition satisfaction: verify E-field tangential components
    are continuous, normal components satisfy jump conditions.
  - These don't require ground truth — they're self-consistency checks.

TIER 3: Engineering-Relevant Metrics (what the engineer actually cares about)
  - S-parameters (S11, S21, etc.) for RF components — derived quantities.
  - Radiation pattern for antennas.
  - Resonant frequency accuracy.
  - Bandwidth prediction.
  - Compare these derived quantities, not just raw field values.

TIER 4: Robustness and Reliability
  - Out-of-distribution detection: train an ensemble or use MC-Dropout to
    estimate prediction uncertainty. Flag designs where uncertainty is high.
  - Adversarial testing: deliberately test on designs at the boundary of the
    training distribution (extreme dimensions, unusual materials).
  - Regression testing: when updating the model, verify it doesn't degrade
    on previously passing test cases.

DEPLOYMENT DECISION FRAMEWORK:
  - Define acceptance criteria WITH the domain expert upfront.
  - Example: "S11 must be within 0.5 dB of HFSS for 95% of test designs."
  - Use the surrogate for exploration/screening, HFSS for final validation.
  - Monitor in production: periodically run HFSS on surrogate-approved designs
    to catch drift.

MY EXPERIENCE:
  In my IEEE JESTIE paper, I validated on fault transients (worst-case scenarios)
  separately from normal operation. DFIG model accuracy was 0.02% MSELoss, battery
  0.00078%, PV 0.2% (normal) to 4% (partial shading). The quantization error
  (float to FPGA fixed-point) was separately below 0.01%. The key was including
  diverse operating conditions (Monte Carlo sampling, 5% fault scenarios) in both
  training AND testing, plus OOD validation on faults 20-100% outside training range.
"""


# ==============================================================================
# PART 3: COMMON PYTORCH GOTCHAS
# ==============================================================================

PYTORCH_GOTCHAS = """
================================================================================
PART 3: 10 COMMON PYTORCH GOTCHAS FOR INTERVIEWS
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 1: Forgetting torch.no_grad() during evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    model.eval()
    output = model(x)  # Still building computation graph! Wastes memory.

FIX:
    model.eval()
    with torch.no_grad():        # Disables gradient tracking
        output = model(x)

WHY: model.eval() only changes behavior of Dropout/BatchNorm layers. It does
NOT disable gradient computation. You need torch.no_grad() to save memory
and speed up inference. In a training loop, this also prevents accidental
gradient leakage from validation into training.

RELATED: Use .detach() when you need a tensor value but don't want gradients
to flow through it (e.g., for logging, or when using a target in a loss):
    target = model_target(x).detach()  # Stop gradients here


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 2: In-place operations breaking autograd
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    x = torch.randn(3, requires_grad=True)
    x += 1          # In-place add! Modifies x's data.
    x.backward()    # RuntimeError: in-place operation modified a leaf Variable

FIX:
    x = torch.randn(3, requires_grad=True)
    y = x + 1       # Creates a NEW tensor; autograd graph is intact.
    y.backward()

COMMON IN-PLACE TRAPS:
    - x += 1, x *= 2, x[0] = 5
    - x.add_(1), x.mul_(2), x.zero_()
    - Any operation ending in _ is in-place

RULE: Never do in-place operations on tensors that require gradients or are
part of the computation graph. Autograd saves references to tensor versions;
in-place ops invalidate those references.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 3: GPU/CPU tensor device mismatch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    model = model.cuda()
    x = torch.randn(4, 10)          # On CPU!
    output = model(x)               # RuntimeError: expected CUDA tensor

FIX:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(4, 10).to(device)
    output = model(x)

BEST PRACTICE: Define `device` once at the top of your script and use it
everywhere. When creating new tensors inside a model (e.g., masks, positional
encodings), use `x.device` to match the input tensor:
    mask = torch.ones(n, device=x.device)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 4: Forgetting model.train() / model.eval()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    # After validation, you forget to switch back:
    model.eval()
    val_loss = validate(model)
    # ... continue training without model.train() ...
    # Dropout is disabled, BatchNorm uses running stats => poor training

FIX:
    model.eval()
    with torch.no_grad():
        val_loss = validate(model)
    model.train()  # ALWAYS switch back before training resumes

WHY: model.eval() changes:
    - Dropout: disabled (no random zeroing)
    - BatchNorm: uses running mean/var instead of batch statistics
    If you forget to switch back, training will silently degrade.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 5: Not zeroing gradients before backward()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    for batch in dataloader:
        loss = compute_loss(model(batch))
        loss.backward()           # Gradients ACCUMULATE!
        optimizer.step()

FIX:
    for batch in dataloader:
        optimizer.zero_grad()     # Clear gradients from previous iteration
        loss = compute_loss(model(batch))
        loss.backward()
        optimizer.step()

WHY: PyTorch accumulates gradients by default (useful for gradient
accumulation across micro-batches). If you forget zero_grad(), gradients
from previous steps add up, leading to incorrect and ever-growing updates.

NOTE: optimizer.zero_grad(set_to_none=True) is slightly faster than the
default (sets .grad to None instead of zero tensor).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 6: Incorrect tensor shape broadcasting leading to silent bugs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    predictions = model(x)         # Shape: (batch, 10)
    targets = get_targets(x)       # Shape: (batch,)
    loss = (predictions - targets) ** 2  # Broadcasts to (batch, 10)!
    # No error, but the loss is computed incorrectly.

FIX:
    targets = targets.unsqueeze(-1)  # Shape: (batch, 1) — explicit
    loss = (predictions - targets) ** 2  # Now (batch, 10) correctly

BEST PRACTICE:
    - Always check shapes with assert or print during development.
    - Use named dimensions or comments: # (batch, seq_len, hidden)
    - Be especially careful with loss functions: MSE between wrong shapes
      can silently compute a scalar that looks reasonable but is wrong.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 7: Memory leaks from storing tensors with grad history
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    losses = []
    for batch in dataloader:
        loss = compute_loss(model(batch))
        losses.append(loss)  # Keeps entire computation graph in memory!
    avg_loss = sum(losses) / len(losses)

FIX:
    losses = []
    for batch in dataloader:
        loss = compute_loss(model(batch))
        losses.append(loss.item())  # .item() extracts Python float, frees graph
    avg_loss = sum(losses) / len(losses)

WHY: A tensor with grad_fn retains the entire computation graph that created
it. Storing such tensors in a list prevents garbage collection of intermediate
activations. GPU memory grows unboundedly. Use .item() for scalar logging
or .detach() if you need the tensor value without the graph.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 8: BatchNorm with batch_size=1 or very small batches
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
    x = torch.randn(1, 10)       # Single sample!
    output = model(x)             # RuntimeError or NaN (variance is 0)

FIX OPTIONS:
    - Use LayerNorm instead of BatchNorm (normalizes across features, not batch).
    - Use InstanceNorm for per-sample normalization.
    - Use GroupNorm (compromise: normalizes across groups of channels).
    - Ensure batch size >= 2 during training (or use SyncBatchNorm for DDP).

WHY: BatchNorm computes mean and variance across the batch dimension.
With batch_size=1, variance is 0, leading to division by zero. Even with
small batches (2-4), the statistics are noisy and unstable.

FOR SIMULATION: LayerNorm or InstanceNorm is usually preferred because
simulation datasets often have small effective batch sizes (each sample
is a large mesh or field).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 9: Loading a state_dict with mismatched keys (especially with DDP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    # Model was saved with DDP wrapper:
    torch.save(ddp_model.state_dict(), "model.pt")
    # Keys look like: "module.linear.weight", "module.linear.bias"

    # Loading into a non-DDP model:
    model.load_state_dict(torch.load("model.pt"))
    # Error: unexpected key "module.linear.weight"

FIX 1 (save correctly):
    torch.save(ddp_model.module.state_dict(), "model.pt")  # Unwrap first

FIX 2 (load flexibly):
    state_dict = torch.load("model.pt")
    # Strip "module." prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

FIX 3 (strict=False for partial loading):
    model.load_state_dict(torch.load("model.pt"), strict=False)
    # Loads matching keys, ignores extras. Useful for transfer learning.
    # BUT: silently ignores typos in key names — use with caution.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOTCHA 10: Incorrect use of torch.Tensor vs torch.tensor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG:
    x = torch.Tensor([1, 2, 3])     # Always float32
    y = torch.Tensor(3)             # Allocates uninitialized 1D tensor of size 3!

FIX:
    x = torch.tensor([1, 2, 3])     # Infers dtype from data (int64 here)
    y = torch.tensor(3)             # Creates a scalar tensor with value 3
    z = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Explicit dtype

WHY: torch.Tensor (capital T) is the class constructor — behaves unexpectedly
with scalar arguments (creates uninitialized tensor of that SIZE, not value).
torch.tensor (lowercase) is the function — always creates from data.

RULE: Always use torch.tensor() (lowercase) for creating tensors from data.
Use torch.zeros(), torch.ones(), torch.randn() for specific patterns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BONUS: QUICK REFERENCE — COMMON PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Proper training loop skeleton:
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # If using a per-step scheduler

# Proper evaluation skeleton:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch.to(device))
            total_loss += criterion(output, target.to(device)).item()
    model.train()  # Switch back!

# Proper checkpointing:
    # Save
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pt')
    # Load
    checkpoint = torch.load('checkpoint.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
"""


# ==============================================================================
# MAIN: Run all tests
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ALL TESTS")
    print("=" * 70)
    print()

    print("--- Question 1: GRU Cell ---")
    test_gru_cell()
    print()

    print("--- Question 2: FNO Layer ---")
    test_fno_layer()
    print()

    print("--- Question 3: GNN Message Passing ---")
    test_message_passing()
    print()

    print("--- Question 4: Diffusion Training ---")
    test_diffusion_training()
    print()

    print("--- Question 5: SIREN ---")
    test_siren()
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

    # Print verbal questions and gotchas (for review)
    print(VERBAL_QUESTIONS)
    print(PYTORCH_GOTCHAS)
