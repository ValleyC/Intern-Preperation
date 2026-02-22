# Synopsys ML Internship — FINAL Interview Preparation

**Interview:** Monday, February 24, 2026, with Xin Xu (Principal R&D Engineer)
**Prepared:** February 20, 2026

---

## Table of Contents

- [Study Plan: 4 Days to Interview](#study-plan-4-days-to-interview)
- [1. Interviewer Intelligence: Xin Xu](#1-interviewer-intelligence-xin-xu)
  - [Key HFSS Concepts to Know](#key-hfss-concepts-to-know)
- [2. ML-for-Simulation Technical Knowledge](#2-ml-for-simulation-technical-knowledge)
  - [2.1 Fourier Neural Operator (FNO)](#21-fourier-neural-operator-fno)
  - [2.2 DeepONet](#22-deeponet)
  - [2.3 Physics-Informed Neural Networks (PINNs)](#23-physics-informed-neural-networks-pinns)
  - [2.4 Implicit Neural Representations (INRs / SIREN)](#24-implicit-neural-representations-inrs--siren)
  - [2.5 Graph Neural Networks for Mesh-Based Simulation](#25-graph-neural-networks-for-mesh-based-simulation)
  - [2.6 Reduced-Order Models (ROMs)](#26-reduced-order-models-roms)
  - [2.7 Deep Dive: MeshGraphNets](#27-deep-dive-meshgraphnets-pfaff-et-al-icml-2021--the-most-relevant-external-paper)
- [3. Research Walkthrough — Presenting Your Papers](#3-research-walkthrough--presenting-your-papers)
  - [Paper 1: IEEE JESTIE (Lead with this)](#paper-1-ieee-jestie-published-2024--lead-with-this)
  - [Paper 2: DAC 2026](#paper-2-dac-2026-under-review--1-minute)
  - [Paper 3: EDISCO — ICML 2026](#paper-3-edisco--icml-2026-under-review--45-seconds)
  - [The Connecting Thread](#the-connecting-thread--have-this-ready)
- [4. Paper Defense: Tough Questions & Best Answers](#4-paper-defense-tough-questions--best-answers)
  - [Paper 1: IEEE JESTIE](#paper-1-ieee-jestie-ml-for-emt-simulation--your-strongest-card)
  - [Paper 2: DAC (Chip Placement)](#paper-2-dac-chip-placement-with-diffusion)
  - [Paper 3: EDISCO (TSP)](#paper-3-edisco-equivariant-diffusion-for-tsp)
  - [Cross-Cutting Questions](#cross-cutting-questions)
- [5. Latest Ansys/Synopsys AI Developments (2025-2026)](#5-latest-ansyssynopsys-ai-developments-2025-2026)
- [6. Your Value Proposition](#6-your-value-proposition-memorize-this)
- [7. Questions to Ask Xin Xu](#7-questions-to-ask-xin-xu-pick-3-4)
- [8. Key Numbers — Quick Reference Card](#8-key-numbers--quick-reference-card)
- [9. Behavioral Questions — STAR Stories](#9-behavioral-questions--star-stories)
- [10. Files in This Folder](#10-files-in-this-folder)
- [11. Final Reminders](#11-final-reminders)

---

## Study Plan: 4 Days to Interview

### Day 1 (Friday Feb 21) — FOUNDATIONS
**Morning: Technical Knowledge (~3 hours)**
- [ ] Read this document end-to-end
- [ ] Review the 6 ML-for-simulation concepts (Section 2) — practice saying each "interview-ready explanation" out loud
- [ ] Read the Ansys SimAI blog post: https://www.ansys.com/blog/explaining-simai

**Afternoon: Paper Defense (~3 hours)**
- [ ] Read Paper Defense section (Section 3) — for each of your 3 papers, practice answering the top 5 tough questions out loud
- [ ] Time yourself: each answer should be 60-90 seconds max
- [ ] Pay special attention to the "red flags to avoid" — these are common mistakes

### Day 2 (Saturday Feb 22) — SYSTEM DESIGN
**Morning: System Design Scenarios (~3 hours)**
- [ ] Read `synopsys_ml_system_design_scenarios.md` in full
- [ ] Practice Scenario 1 (Mesh Refinement) and Scenario 4 (Diffusion for EM) — these are the most likely
- [ ] Practice the META-framework: Clarify → Scope → Data → Model → Training → Validation → Integration

**Afternoon: PyTorch Coding (~2 hours)**
- [ ] Review `pytorch_coding_review.py` — understand each implementation
- [ ] Run the file: `python pytorch_coding_review.py` to verify everything works
- [ ] Practice implementing the GRU cell and GNN layer from memory (most likely to be asked)
- [ ] Review the 10 PyTorch gotchas

### Day 3 (Sunday Feb 23) — MOCK INTERVIEW
**Morning: Full Mock Interview (~2 hours)**
- [ ] Do a full mock interview with a friend or record yourself
- [ ] Opening: 60-second elevator pitch
- [ ] "Tell me about your most relevant research" → lead with IEEE JESTIE
- [ ] "How would you accelerate EM simulation with ML?" → Scenario 2 or 4
- [ ] One tough paper defense question per paper
- [ ] "What questions do you have for me?" → pick 3-4 from Section 6

**Afternoon: Review & Polish (~2 hours)**
- [ ] Review any weak areas identified during mock
- [ ] Re-read the interviewer context (Section 1) — know who you're talking to
- [ ] Review key numbers (Section 7) — commit to memory
- [ ] Prepare your physical setup (quiet room, good internet, papers accessible)

### Day 4 (Monday Feb 24) — INTERVIEW DAY
**Morning: Light Review (~1 hour)**
- [ ] Re-read elevator pitch and "tell me about yourself" answer
- [ ] Skim key numbers one more time
- [ ] Review your 3-4 questions to ask
- [ ] DO NOT study anything new — you know this material

---

## 1. Interviewer Intelligence: Xin Xu

### The Job Description — What They Actually Want
The four responsibilities, in order of importance for your interview:

| JD Responsibility | What It Really Means | Your Direct Match |
|---|---|---|
| **"Apply ML to make predictions on decisions during simulation"** | Predict solver decisions: where to refine mesh, when to stop iterating, which solver strategy to use | JESTIE: you predicted which architecture (GRU vs MLP) fits each component's physics; your decomposition IS a solver decision |
| **"Dynamically refine predictions as more information becomes available"** | Iterative improvement during the solve — not one-shot prediction | DAC: dense per-step energy feedback during diffusion = refining as info arrives; EDISCO: continuous-time diffusion = progressive refinement |
| **"Study and summarize existing ML algorithms and frameworks"** | Literature review + understanding team's existing codebase | Your breadth across GRU/MLP/diffusion/equivariant GNN + experience reading and building on prior work |
| **"Summarize findings and present research achievements"** | Communication and paper-writing | 3 papers (1 published, 2 under review at top venues) |

**Key phrase to echo in your answers:** "predictions on decisions during simulation" — this is their language, use it.

### What We Know
- **Title:** Principal R&D Engineer at Ansys (now Synopsys post-acquisition)
- **LinkedIn:** linkedin.com/in/xin-xu-1a794225
- **No public publications found** — this is typical for senior engineers at simulation companies whose work is embedded in proprietary software

### What We Infer (from HFSS R&D job postings and team structure)
**Most likely technical focus: HFSS core solver development**

The Ansys HFSS R&D team works on:
- **Finite Element Method (FEM)** — the foundational solver technology
- **Adaptive mesh refinement** — HFSS's hallmark auto-adaptive meshing
- **Domain decomposition methods (DDM)** — distributing large EM problems across compute nodes
- **Hybrid solvers** — FE-BI, integral equations, SBR+
- **Mesh Fusion** — patented tech (2021) allowing different meshing algorithms in different geometric regions
- **GPU acceleration** — CUDA-enabled NVIDIA GPU support
- **HPC scalability** — multi-node parallelism

### What This Means for You
| He Will Care About | How to Address |
|---|---|
| Physical accuracy and reliability | Always mention validation, safety mechanisms, solver fallback |
| Solver efficiency, not just ML metrics | Frame speedup in terms of solver iterations saved, not just ML loss |
| Practical deployment in production software | Mention integration with existing HFSS workflow |
| Whether you understand FEM and meshing | Show familiarity with adaptive tetrahedral meshes, S-parameter convergence |
| Whether ML is substance or hype | Be honest about limitations, never oversell |

### Key HFSS Concepts to Know

| Term | What It Is |
|------|-----------|
| **Adaptive mesh refinement** | HFSS iteratively refines tetrahedral mesh until S-parameters converge (Delta-S criterion) |
| **S-parameters** | Scattering parameters (S11, S21, ...) — what engineers use to characterize RF components |
| **Delta-S** | The change in S-parameters between consecutive mesh refinement passes — convergence criterion |
| **Tetrahedral mesh** | 3D mesh of tetrahedra used by HFSS FEM solver |
| **FEM** | Finite Element Method — discretize Maxwell's equations on mesh, solve sparse linear system |
| **Domain decomposition** | Split large problems into sub-domains, solve in parallel, iterate on boundaries |
| **Mesh Fusion** | Multiple meshing algorithms (Classic, TAU, Phi) applied to different regions of the same design |
| **SBR+** | Shooting and Bouncing Rays — asymptotic method for large-scale radar/antenna problems |

---

## 2. ML-for-Simulation Technical Knowledge

### 2.1 Fourier Neural Operator (FNO)

**Core idea:** Learns mappings between function spaces by parameterizing convolutions in Fourier space. FFT the input, multiply by learnable weights at truncated frequency modes, inverse FFT back.

**Key properties:**
- Resolution-invariant (train on 64x64, infer on 256x256)
- Captures global interactions in O(N log N)
- Requires structured/regular grids (major limitation for HFSS tetrahedral meshes)

**Interview-ready explanation:**
> "FNO learns operator mappings between function spaces, making it resolution-invariant. It works by performing learned convolutions in Fourier space — you FFT the input, multiply by learnable weights at truncated frequency modes, then inverse FFT back. This captures global interactions efficiently, but the reliance on FFT means it requires structured grids, which is a significant limitation for EM simulation where we typically use unstructured adaptive meshes."

### 2.2 DeepONet

**Core idea:** Branch net encodes the input function at sensor locations; trunk net encodes the output query coordinate. Their dot product gives the solution at any continuous point.

**Key difference from FNO:** Works on unstructured grids and irregular domains. More flexible but doesn't inherently capture global interactions.

**Interview-ready explanation:**
> "DeepONet splits operator learning into a branch net (encodes the input function) and a trunk net (encodes the output query coordinate). Unlike FNO, it does not require structured grids, making it naturally suited for irregular geometries. The trunk net learns adaptive basis functions while the branch net learns the coefficients — it's a learned generalization of classical basis expansion methods."

### 2.3 Physics-Informed Neural Networks (PINNs)

**Core idea:** Embed PDE residuals in the loss function. Train by penalizing violations of governing equations at collocation points.

**Known failure modes:**
- Spectral bias (slow to learn high-frequency components)
- Loss balancing between PDE residual, BCs, and data terms
- Must be retrained per new geometry (not amortized)

**Why Ansys likely prefers data-driven over PINNs:**
- Data-driven surrogates give millisecond inference (PINNs need retraining per problem)
- Ansys already has high-fidelity solvers to generate training data
- PINNs have unpredictable convergence (bad for commercial products)

**Interview-ready explanation:**
> "PINNs embed the PDE residual directly into the loss function — automatic differentiation computes derivatives of the network output, and we penalize violations of the governing equations. This means you can train without simulation data. However, for industrial EM applications, PINNs have critical limitations: spectral bias with high-frequency fields, retraining per geometry, and slow training. That's why SimAI uses a data-driven approach — train on existing HFSS data once, get millisecond inference for new designs."

### 2.4 Implicit Neural Representations (INRs / SIREN)

**Core idea:** Represent continuous signals as coordinate-to-value neural networks: f(x,y,z) → field values. SIREN uses sinusoidal activations.

**Why sinusoidal activations matter:** Derivatives of sines are cosines — the network's derivatives inherit the same expressive power. Critical for EM fields where we need curl and divergence.

**Connection to SimAI:** SimAI's description as "learning continuous functions" strongly suggests INR-like components.

**Interview-ready explanation:**
> "INRs parameterize a continuous signal as a coordinate-to-value mapping. SIREN uses sinusoidal activations, which is critical because derivatives of sines are cosines — the network's derivatives inherit the same expressive power. For EM simulation, this means curl and divergence operators are well-behaved, unlike ReLU networks where second derivatives vanish."

### 2.5 Graph Neural Networks for Mesh-Based Simulation

**Core idea:** Encode the simulation mesh as a graph. Message passing lets each node aggregate information from neighbors through learned functions.

**Why relevant for HFSS:** Operates directly on unstructured tetrahedral meshes. Handles adaptive refinement naturally (graph just gains more nodes/edges).

**Interview-ready explanation:**
> "MeshGraphNets encode the simulation mesh as a graph — nodes are mesh vertices, edges follow mesh connectivity. Message passing lets each node aggregate information from neighbors, learning the local physics update rule. This is particularly relevant for HFSS because it handles unstructured adaptive tetrahedral meshes natively, unlike FNO which requires regular grids."

### 2.6 Reduced-Order Models (ROMs)

**Core idea:** Compress high-dimensional simulation states into a low-dimensional latent space. Classical POD uses SVD on solution snapshots; autoencoder-based ROMs generalize to nonlinear manifolds.

**Connection to SimAI:** SimAI essentially builds data-driven compressed representations from simulation data, then predicts in that compressed space.

**Interview-ready explanation:**
> "Reduced-order models compress high-dimensional simulation states into a low-dimensional latent space. Classical POD uses SVD on solution snapshots — it's like PCA on simulation data. Autoencoder-based ROMs generalize this to nonlinear manifolds. This is fundamentally what SimAI does: build a compressed representation from HFSS data, then predict in that compressed space for orders-of-magnitude speedup."

### Quick Comparison Table

| Method | Grid Type | Physics? | Data Needs | Inference Speed |
|--------|-----------|----------|------------|-----------------|
| FNO | Uniform/structured | No (data-driven) | High | Very fast |
| DeepONet | Any | Optional | Moderate | Fast |
| PINNs | Meshfree | Yes (PDE in loss) | Low/None | Slow (retrain) |
| SIREN/INR | Continuous | Optional | Per-signal | Fast (after training) |
| GNN/MeshGraphNets | Unstructured | Optional | High | Moderate |
| ROM (POD/AE) | From full solver | Optional | Moderate | Very fast |

---

### 2.7 Deep Dive: MeshGraphNets (Pfaff et al., ICML 2021) — THE Most Relevant External Paper

> **Why this paper matters more than any other:** HFSS solves Maxwell's equations on adaptive tetrahedral meshes. MeshGraphNets learns physics simulation by treating the FEM mesh as a graph and running message-passing GNNs over it — the exact same data structure. If Xin Xu's team is applying ML to HFSS, this paper is almost certainly in their reading list.

**Paper:** "Learning Mesh-Based Simulation with Graph Networks"
**Authors:** Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia (DeepMind)
**Venue:** ICML 2021 | **arXiv:** 2010.03409

---

#### Step 1: Problem & Key Insight

> "Classical simulation solvers (FEM, FVM) are accurate but slow. The key insight is that simulation meshes ARE graphs — so a GNN that operates on this graph can learn the physics update rule directly, running 1-2 orders of magnitude faster than the original solver."

**[WHITEBOARD: the core idea]**
```
  Classical FEM Pipeline:
  ┌──────────┐    ┌───────────┐    ┌──────────┐
  │ Geometry │ →  │ Mesh      │ →  │ Solve    │ → Solution fields
  │          │    │ (tetra/   │    │ Ku = f   │   (E, H, S-params)
  │          │    │  triangle) │    │          │
  └──────────┘    └───────────┘    └──────────┘
                       ↓
  MeshGraphNets:       │ Treat mesh as graph!
                       ↓
  ┌──────────┐    ┌───────────┐    ┌──────────┐
  │ Mesh     │ →  │ GNN       │ →  │ Predict  │ → Approximate fields
  │ as Graph │    │ (15-step  │    │ next     │   (11-289× faster)
  │ V, E     │    │  msg pass)│    │ timestep │
  └──────────┘    └───────────┘    └──────────┘
```

---

#### Step 2: Dual-Space Graph Representation

> "The most important architectural decision: the model operates in TWO spaces simultaneously — mesh-space (rest-state topology) and world-space (current physical configuration). This is what makes it work for both material dynamics and collision detection."

**[WHITEBOARD: dual-space representation]**
```
  MESH-SPACE (rest geometry)         WORLD-SPACE (deformed geometry)
  ┌─────────────────────┐           ┌─────────────────────┐
  │  A───B              │           │     A'              │
  │  │ / │              │           │    / \              │
  │  │/  │    Mesh      │           │   /   \   World    │
  │  C───D    edges     │           │  C'────D'  edges   │
  │           (E^M)     │           │  ↕ ↕      (E^W)    │
  │  Encodes: strain,   │           │  B'  Encodes:      │
  │  rest connectivity  │           │  proximity,        │
  │                     │           │  collisions         │
  └─────────────────────┘           └─────────────────────┘

  WHY BOTH?
  ─────────
  Mesh-space only → Can't detect collisions (cloth folding onto itself)
  World-space only → Can't encode material rest state (strain = deformation)
  Both together → Complete physical description

  Ablation proof:
  • Remove world edges → 51-92% RMSE increase (collisions fail)
  • Remove mesh-space features → Can't learn elastic materials at all
```

**Graph construction details:**

```
  Node features:
  ┌────────────────────────────────────────────┐
  │ • Dynamical quantities (velocity/momentum) │
  │ • One-hot node type (normal/wall/obstacle)  │
  │ • Next-step velocity (for kinematic nodes)  │
  └────────────────────────────────────────────┘

  Mesh-edge features (E^M):
  ┌────────────────────────────────────────────┐
  │ • u_ij = u_i - u_j  (mesh-space relative)  │
  │ • |u_ij|            (mesh-space distance)   │
  │ • x_ij = x_i - x_j  (world-space relative) │
  │ • |x_ij|            (world-space distance)  │
  └────────────────────────────────────────────┘

  World-edge features (E^W) — only for Lagrangian systems:
  ┌────────────────────────────────────────────┐
  │ • x_ij = x_i - x_j  (world-space relative) │
  │ • |x_ij|            (world-space distance)  │
  │ • Created when |x_i - x_j| < r_W           │
  │ • Excluded if already a mesh edge           │
  └────────────────────────────────────────────┘
```

---

#### Step 3: Encoder-Processor-Decoder Architecture

> "The architecture is simple but powerful: encode features into 128-dim latent vectors, run 15 rounds of message passing, decode to physical predictions. All components are just 2-layer MLPs."

**[WHITEBOARD: full architecture]**
```
  ┌─────────────────────────── ENCODER ───────────────────────────┐
  │                                                                │
  │  Node features ──→ ε_V (MLP: 2-layer, 128-hidden) ──→ v_i    │
  │  Mesh-edge feat ──→ ε_M (MLP: 2-layer, 128-hidden) ──→ e^M_ij│
  │  World-edge feat ─→ ε_W (MLP: 2-layer, 128-hidden) ──→ e^W_ij│
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
                              ↓
  ┌───────────────── PROCESSOR (×15 blocks) ──────────────────────┐
  │                                                                │
  │  For each block l = 1, ..., 15:                                │
  │                                                                │
  │  1. Edge update (mesh):  e'^M_ij ← f^M([e^M_ij; v_i; v_j])   │
  │  2. Edge update (world): e'^W_ij ← f^W([e^W_ij; v_i; v_j])   │
  │  3. Node update:         v'_i ← f^V([v_i; Σ_j e'^M_ij;       │
  │                                             Σ_j e'^W_ij])     │
  │                                                                │
  │  All f^M, f^W, f^V are 2-layer MLPs with:                     │
  │    • Residual connections: output = input + MLP(input)         │
  │    • LayerNorm on output                                       │
  │    • Each block has its OWN parameters (not shared)            │
  │                                                                │
  │  15 blocks → information travels 15 hops through the mesh     │
  └────────────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────── DECODER ───────────────────────────┐
  │                                                                │
  │  p_i = δ_V(v_i)   (MLP: 2-layer, 128-hidden, NO LayerNorm)   │
  │                                                                │
  │  Output = derivative (acceleration or rate of change)          │
  │                                                                │
  │  Integration (forward-Euler, Δt = 1):                          │
  │  • 1st-order (fluids):    q^{t+1} = p_i + q^t                 │
  │  • 2nd-order (cloth):     q^{t+1} = p_i + 2q^t - q^{t-1}     │
  │                           (Störmer-Verlet style)               │
  └────────────────────────────────────────────────────────────────┘
```

**Architecture config:**
```
  ┌────────────────────────────────────────────┐
  │ Latent dimension:         128              │
  │ MLP hidden layers:        2                │
  │ MLP hidden size:          128              │
  │ Activation:               ReLU             │
  │ Normalization:            LayerNorm        │
  │                           (except decoder) │
  │ Message-passing blocks:   15               │
  │ Optimizer:                Adam             │
  │ LR schedule:              1e-4 → 1e-6     │
  │                     (exp decay over 5M)    │
  │ Total training steps:     10M              │
  │ Hardware:                 1× V100 GPU      │
  │ Memory:                   1-2.5 GB         │
  └────────────────────────────────────────────┘
```

---

#### Step 4: Training — Noise Injection for Rollout Stability

> "This is the most clever training trick. During training, they only supervise ONE step at a time (no expensive multi-step rollout loss). But they add noise to the input to simulate the kind of errors the model will see during autoregressive rollout. The target is adjusted so the model learns to CORRECT errors, not just predict the next state."

**[WHITEBOARD: noise injection mechanism]**
```
  Problem: At inference, the model feeds its OWN predictions back as input.
           Small errors compound → trajectory diverges.

  Solution: During training, add noise ε to the input state:

  FIRST-ORDER SYSTEMS:
  ┌──────────────────────────────────────────────────────┐
  │ Input:  q̃ = q^t + ε        (ε ~ N(0, σ²))          │
  │ Target: p̃ = p - ε          (adjusted to correct!)   │
  │                                                      │
  │ After integration: q̃^{t+1} = p̃ + q̃                  │
  │                            = (p - ε) + (q^t + ε)    │
  │                            = p + q^t                 │
  │                            = q^{t+1}  ← TRUE value! │
  └──────────────────────────────────────────────────────┘

  SECOND-ORDER SYSTEMS (cloth):
  ┌──────────────────────────────────────────────────────┐
  │ Adding noise to position corrupts velocity estimate  │
  │ (ẋ = x^t - x^{t-1}), creating a dilemma:            │
  │                                                      │
  │ Position-corrected target: ẍ_P  (corrects position)  │
  │ Velocity-corrected target: ẍ_V  (corrects velocity)  │
  │                                                      │
  │ Blended: ẍ̃ = γ·ẍ_P + (1-γ)·ẍ_V                      │
  │                                                      │
  │ Optimal: γ = 0.1  (90% velocity correction)          │
  └──────────────────────────────────────────────────────┘

  Loss function: L = Σ_i || p_i - p̄_i ||²   (per-node L2)

  Noise magnitudes per domain:
  ┌───────────────────┬────────────┬────────────┐
  │ Domain            │ Variable   │ σ          │
  │───────────────────│────────────│────────────│
  │ FlagSimple        │ position   │ 1×10⁻³    │
  │ FlagDynamic       │ position   │ 3×10⁻³    │
  │ SphereDynamic     │ position   │ 1×10⁻³    │
  │ DeformingPlate    │ position   │ 3×10⁻³    │
  │ CylinderFlow      │ momentum   │ 2×10⁻²    │
  │ Airfoil           │ momentum   │ 1×10¹     │
  │ Airfoil           │ density    │ 1×10⁻²    │
  └───────────────────┴────────────┴────────────┘

  WHY IT WORKS: Analogous to scheduled sampling / exposure bias
  correction in NLP. The model learns to "snap back" to the true
  trajectory even when its input is slightly wrong.
```

**Connection to your work:** "This solves the same error accumulation problem I discuss in my JESTIE paper. My approach was to recommend a conventional model monitor as a safety fallback. MeshGraphNets takes a different but complementary approach — building robustness directly into training. For HFSS, I'd combine both: noise-augmented training PLUS uncertainty monitoring."

---

#### Step 5: Learned Adaptive Remeshing

> "This is the feature most relevant to HFSS. The model doesn't just predict physics — it also learns WHERE the mesh needs to be refined. This is exactly 'predictions on decisions during simulation.'"

**[WHITEBOARD: learned remeshing]**
```
  Sizing Field Prediction:
  ┌────────────────────────────────────────────────────────┐
  │ Additional decoder output: S_i ∈ R^{2×2}              │
  │ (symmetric positive-definite tensor per node)          │
  │                                                        │
  │ Defines desired local resolution as an ELLIPSE:        │
  │                                                        │
  │ Edge (i,j) is valid iff:  u_ij^T · S_i · u_ij ≤ 1    │
  │                                                        │
  │ Too long → SPLIT    Just right → KEEP    Too short → COLLAPSE│
  └────────────────────────────────────────────────────────┘

  Remeshing Operations (in order):
  ┌────────────────────────────────────────────────────────┐
  │ 1. SPLIT:    u_ij^T S u_ij > 1 → insert midpoint     │
  │ 2. FLIP:     anisotropic Delaunay criterion            │
  │ 3. COLLAPSE: edge too short, no new invalid edges     │
  │ 4. FLIP:     final quality improvement                 │
  └────────────────────────────────────────────────────────┘

  When ground-truth sizing unavailable:
  → Estimate via MINIDISK algorithm:
    S_i = argmax Σ_{j∈N(i)} u_ij^T S_i u_ij
    s.t.  u_ij^T S_i u_ij ≤ 1  ∀ j ∈ N(i)
    (= minimum-area ellipse containing all neighbor vectors)

  HFSS Connection:
  ┌────────────────────────────────────────────────────────┐
  │ HFSS adaptive meshing loop:                            │
  │   Solve → Error estimate → Refine mesh → Re-solve     │
  │                                                        │
  │ MeshGraphNets can learn the "Error estimate → Refine"  │
  │ step, predicting WHERE to refine from the current      │
  │ field state — without running the full solve first.     │
  │                                                        │
  │ This IS "predictions on decisions during simulation"   │
  └────────────────────────────────────────────────────────┘
```

---

#### Step 6: Results — 6 Physics Domains, 11-289× Speedup

**[WHITEBOARD: comprehensive results]**
```
  Domains tested (all with ONE architecture, no domain-specific changes):

  LAGRANGIAN (deforming meshes):
  ┌────────────────────────────────────────────────────────────┐
  │ Domain          │ Solver  │ Mesh      │ ~Nodes │ System    │
  │─────────────────│─────────│───────────│────────│───────────│
  │ FlagSimple      │ ArcSim  │ Triangle  │ 1,579  │ Cloth     │
  │ FlagDynamic     │ ArcSim  │ Tri+Remesh│ 2,767  │ Cloth     │
  │ SphereDynamic   │ ArcSim  │ Tri+Remesh│ 1,373  │ Cloth+Obj │
  │ DeformingPlate  │ COMSOL  │ Tetrahedral│ 1,271 │ Elastic   │
  └────────────────────────────────────────────────────────────┘

  EULERIAN (fixed meshes):
  ┌────────────────────────────────────────────────────────────┐
  │ CylinderFlow    │ COMSOL  │ Triangle  │ 1,885  │ Incomp NS │
  │ Airfoil         │ SU2     │ Triangle  │ 5,233  │ Comp NS   │
  └────────────────────────────────────────────────────────────┘

  Speedup & Accuracy:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Domain         │ 1-step RMSE │ 50-step RMSE │ GPU     │ Speedup │
  │                │ (×10⁻³)     │ (×10⁻³)      │ ms/step │         │
  │────────────────│─────────────│──────────────│─────────│─────────│
  │ FlagSimple     │ 1.08 ± 0.02 │ 92.6 ± 5.0  │   19    │ 214.7×  │
  │ FlagDynamic    │ 1.57 ± 0.02 │ 72.4 ± 4.3  │   43    │  31.3×  │
  │ SphereDynamic  │ 0.29 ± 0.005│ 11.5 ± 0.9  │   32    │  11.5×  │
  │ DeformingPlate │ 0.25 ± 0.05 │  1.8 ± 0.5  │   24    │  89.0×  │
  │ CylinderFlow   │ 2.34 ± 0.12 │  6.3 ± 0.7  │   21    │  35.3×  │
  │ Airfoil        │  314 ± 36   │  582 ± 37    │   37    │ 289.1×  │
  └──────────────────────────────────────────────────────────────────┘

  Also runs on CPU (4-22× speedup, no GPU required).
  Models stay STABLE for 40,000+ step rollouts.
```

---

#### Step 7: Ablations — What Matters Most

**[WHITEBOARD: ablation insights]**
```
  ┌──────────────────────────────────────────────────────────┐
  │ Ablation                   │ Effect                      │
  │────────────────────────────│─────────────────────────────│
  │ Remove world edges         │ 51-92% RMSE increase        │
  │ (collisions fail)          │ (critical for Lagrangian)   │
  │                            │                             │
  │ Remove mesh-space features │ Elastic materials break     │
  │ (no rest-state info)       │ (GNS particle failure mode) │
  │                            │                             │
  │ Use absolute coordinates   │ Airfoil RMSE: 11.5 → 26.5  │
  │ instead of relative edges  │ (breaks spatial equivariance)│
  │                            │                             │
  │ Reduce msg-pass to L=5    │ Significant accuracy drop   │
  │                            │ (receptive field too small)  │
  │                            │                             │
  │ Add history h > 1          │ Overfitting (diminishing    │
  │                            │ returns, worse generalize)   │
  └──────────────────────────────────────────────────────────┘

  Baseline comparisons:
  ┌──────────────────────────────────────────────────────────┐
  │ Method              │ Problem                            │
  │─────────────────────│────────────────────────────────────│
  │ GNS (particles)     │ Cannot encode rest state →         │
  │                     │ FAILS on cloth (entangled meshes)  │
  │                     │                                    │
  │ GCN                 │ No edge-level computation →        │
  │                     │ Unstable rollouts                  │
  │                     │                                    │
  │ UNet (CNN on grid)  │ Undersamples critical regions,     │
  │                     │ 4× more cells needed, oscillations │
  │                     │ Cannot handle irregular domains    │
  └──────────────────────────────────────────────────────────┘
```

---

#### Step 8: Generalization — Why This Isn't Just Memorization

**[WHITEBOARD: generalization evidence]**
```
  Parameter Extrapolation (Airfoil):
  ┌──────────────────────────────────────────────────────┐
  │ Test Condition           │ RMSE   │ Degradation      │
  │──────────────────────────│────────│──────────────────│
  │ In-distribution          │ 11.5   │ baseline         │
  │ Angle: ±25° → ±35°     │ 12.4   │ only 8%          │
  │ Mach: 0.2-0.7 → 0.7-0.9│ 13.1   │ only 14%         │
  └──────────────────────────────────────────────────────┘

  Shape Generalization (Cloth):
  • Train on rectangular flags (~2,700 nodes)
  • Test on fish-shaped flag → stable, accurate
  • Test on windsock with tassels → 20,000 nodes
    (10× larger, non-flat starting state never seen)
    → still produces stable predictions!

  Resolution Independence:
  • Single model generalizes to 10× larger meshes
  • Fundamental advantage of mesh-based over grid-based

  WHY: Relative edge features + message passing =
       translation/rotation-invariant local physics rules.
       The model learns "how neighbors interact" not
       "what happens at coordinate (x,y)."
```

---

#### Step 9: Limitations & Honest Assessment

```
  ┌──────────────────────────────────────────────────────────┐
  │ 1. CHAOTIC DIVERGENCE: Rollouts stay stable but         │
  │    diverge from specific trajectory after ~50 steps     │
  │    in chaotic systems (cloth). Statistics remain correct.│
  │                                                          │
  │ 2. NO CONSERVATION GUARANTEES: Does not enforce          │
  │    energy/momentum conservation. Physics-based loss      │
  │    terms could help (future work).                       │
  │                                                          │
  │ 3. REMESHING BOTTLENECK: CPU-based remesher dominates   │
  │    inference time (43ms model vs 837ms total for         │
  │    FlagDynamic). GPU remesher would help.                │
  │                                                          │
  │ 4. NOISE TUNING: σ varies over 4 orders of magnitude    │
  │    across domains. No principled selection method.        │
  │                                                          │
  │ 5. TRIANGULAR MESHES ONLY: Generic remesher handles     │
  │    triangles. HFSS uses tetrahedra → need new remesher. │
  │                                                          │
  │ 6. 1000 TRAINING TRAJECTORIES: Expensive if each        │
  │    trajectory requires hours of full simulation.         │
  │                                                          │
  │ 7. NO CROSS-DOMAIN TRANSFER: One model per physics      │
  │    domain. Unlike EDISCO's cross-distribution robustness.│
  └──────────────────────────────────────────────────────────┘
```

---

#### How to Talk About This Paper in the Interview

> **If asked "What related work have you studied?"**
>
> "MeshGraphNets by Pfaff et al. is the paper I think is most relevant to this role. It learns mesh-based simulation by treating the FEM mesh as a graph — nodes are mesh vertices, edges follow mesh connectivity — then runs 15 rounds of message passing to predict the next-step physics update. It achieves 11-289× speedup over classical solvers across six physics domains with a single architecture.
>
> Three aspects are particularly relevant to HFSS. First, it operates on the same unstructured mesh data structure that HFSS uses. Second, it learns adaptive remeshing — predicting a sizing field that determines where the mesh needs refinement — which is exactly 'predictions on decisions during simulation.' Third, it uses relative edge features for spatial equivariance, which connects to my EDISCO work on E(2)-equivariant GNNs.
>
> Where I see room for improvement: it doesn't guarantee conservation of physical quantities like energy or momentum, and it requires ~1000 training trajectories per domain. My EDISCO work addresses the first issue through equivariant architectures that preserve geometric symmetries by construction, and my DAC work addresses the second through unsupervised training from physics-based energy functions — no labeled data needed."

> **Mapping MeshGraphNets concepts to your papers:**

```
  ┌──────────────────────────┬──────────────────────────────────┐
  │ MeshGraphNets            │ Your Work                        │
  │──────────────────────────│──────────────────────────────────│
  │ GNN on simulation mesh   │ JESTIE: component surrogates     │
  │                          │ (both replace expensive solves)   │
  │                          │                                  │
  │ Learned remeshing        │ DAC: dense per-step energy       │
  │ (sizing field prediction)│ (both predict at each iteration) │
  │                          │                                  │
  │ Relative edge features   │ EDISCO: E(2)-equivariant GNN    │
  │ (spatial equivariance)   │ (principled geometric symmetry)  │
  │                          │                                  │
  │ Training noise for       │ JESTIE: error accumulation       │
  │ rollout stability        │ discussion + monitor fallback    │
  │                          │                                  │
  │ Multi-physics (1 arch)   │ JESTIE: GRU for stateful,       │
  │                          │ MLP for stateless (match arch    │
  │                          │ to physics structure)            │
  │                          │                                  │
  │ No conservation laws     │ EDISCO: equivariance enforces    │
  │ (limitation)             │ symmetry by construction (your   │
  │                          │ advantage)                       │
  └──────────────────────────┴──────────────────────────────────┘
```

---

## 3. Research Walkthrough — Presenting Your Papers

> **When this gets asked:** "Tell me about your research," "Walk me through your papers," or "What have you been working on?" This is almost certainly the first technical question. Lead with JESTIE, then briefly cover DAC and EDISCO, then tie them together. Aim for ~5 minutes total if uninterrupted.

### Paper 1: IEEE JESTIE (Published, 2024) — Lead with this

**Opening (30 sec):**
> "My most relevant published work is in IEEE JESTIE, on using neural networks to accelerate electromagnetic transient simulation for large-scale renewable energy systems. The core problem is that real-time hardware-in-the-loop testing of power grids requires solving complex nonlinear dynamics at 50-microsecond timesteps — and when you have hundreds of renewable energy devices, the traditional solver simply can't keep up."

---

#### Step 1: What We're Replacing — Traditional IBR Models

> "First, let me explain what the ML models are learning to replace. Each IBR type has a traditional computational model that's accurate but expensive."

**[WHITEBOARD: draw the three traditional models side-by-side]**
```
┌──────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│  (a) DFIG Wind       │  │  (b) Li-ion Battery   │  │  (c) PV Panel         │
│      Turbine         │  │                       │  │                       │
│                      │  │  Thévenin Equivalent:  │  │  Single-Diode Model:  │
│  Wound-Rotor IG      │  │                       │  │                       │
│  ┌──────────────┐    │  │  VBat ──[ZB]──┬── Vt  │  │  Iph ──┬──[Rs]── Iout│
│  │ Stator ↔ Grid│    │  │              │       │  │        │             │
│  │ Rotor ↔ RSC  │    │  │  VBat = E₀   │       │  │       [Rsh]          │
│  └──────────────┘    │  │    + Epol     │       │  │        │             │
│  ┌─────┐  ┌─────┐   │  │    + Eexp     │       │  │  Iph = (Sirr/S*irr)  │
│  │ GSC │←→│ RSC │   │  │    + ScEchg   │       │  │    × I*ph(1+αT·ΔT)  │
│  │Ctrl │  │Ctrl │   │  │    +(1-Sc)Edsc│       │  │                       │
│  └─────┘  └─────┘   │  │              │       │  │  Norton equiv:        │
│                      │  │  Norton equiv:│       │  │  JPV current source   │
│  Controls P,Q        │  │  IBeq=VBat·GB │       │  │  + conductance GPV    │
│  independently       │  │              │       │  │                       │
└──────────────────────┘  └───────────────────────┘  └───────────────────────┘
         ↓                          ↓                          ↓
   Complex nonlinear          Time-dependent             Static nonlinear
   + time-dependent           (SOC, thermal)             I-V relationship
   (rotor inertia,            → needs GRU                → MLP suffices
    flux dynamics)
   → needs GRU
```

> "The DFIG is the most complex: a wound-rotor induction generator with back-to-back converters (GSC and RSC) that independently control active and reactive power. The battery follows a Thévenin equivalent with state-of-charge dynamics. The PV cell is a single-diode model with irradiance-dependent current. All three are converted to Norton equivalent circuits (current source + conductance) for EMT nodal analysis."

> "The key insight driving the architecture choice: **DFIG and battery have temporal state** (rotor inertia, SOC) → GRU. **PV is a memoryless I-V curve** → MLP."

---

#### Step 2: System Decomposition — Component-Level Surrogates

**[WHITEBOARD: draw the full system decomposition]**
```
┌──────────────────────────────────────────────────────────────────────┐
│                    IEEE 118-Bus AC Network                            │
│                  (Transient Stability, Δt = 5ms)                     │
│                                                                      │
│     Bus 25 ←── 25kV/138kV Transformer ←── IBR Microgrid             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    ML-Based IBR Microgrid                       │ │
│  │                                                                 │ │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │ │
│  │  │   Wind Farm      │  │   Solar Farm      │  │  BESS Station │  │ │
│  │  │   270 MW         │  │   62.5 kW         │  │   50 MW       │  │ │
│  │  │                  │  │                   │  │               │  │ │
│  │  │  30 GRU models   │  │  20 MLP models    │  │  1 GRU model  │  │ │
│  │  │  × 6 turbines    │  │  × 4×4 panels     │  │  3×4 battery  │  │ │
│  │  │  = 180 turbines  │  │  = 320 panels     │  │  array        │  │ │
│  │  │                  │  │                   │  │               │  │ │
│  │  │  30 × 9MW        │  │  5 parallel ×     │  │  Scaled to    │  │ │
│  │  │  = 270MW total   │  │  4 series ×       │  │  50 MW        │  │ │
│  │  │                  │  │  3.125kW = 62.5kW │  │               │  │ │
│  │  │  EMT: 50µs       │  │  EMT: 50µs        │  │  EMT: 50µs    │  │ │
│  │  └─────────────────┘  └──────────────────┘  └───────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

Why component-level instead of monolithic?
  ✓ Match architecture to physics (GRU for stateful, MLP for memoryless)
  ✓ Batch-parallel on FPGA (30 identical GRU copies run simultaneously)
  ✓ Preserve individual unit characteristics (each turbine gets unique wind)
  ✓ Scale by adding copies, not retraining
```

> "Instead of one giant model for the whole system, I matched each component type to the right architecture. The wind farm is 30 copies of the same GRU model — each representing 6 turbines in parallel — running in batch-parallel on FPGA. This means scaling from 180 to 1800 turbines just means 10x more copies, no retraining."

---

#### Step 3: GRU Architecture — The Core ML Contribution

> "Let me walk through the GRU mechanics since this is central to the ML contribution."

**[WHITEBOARD: GRU cell equations and data flow]**
```
GRU Cell at timestep t:
═══════════════════════
                    ┌──────────────────────────────────────────┐
  xt (input) ──────→│                                          │
                    │  Update gate:  zt = σ(Wz·xt + Uz·h(t-1)) │
  h(t-1) ─────────→│  Reset gate:   rt = σ(Wr·xt + Ur·h(t-1)) │
  (prev hidden)     │  Candidate:    nt = tanh(Wn·xt            │
                    │                    + rt ⊙ (Un·h(t-1)))    │
                    │  Output:       ht = (1-zt)⊙nt + zt⊙h(t-1)│
                    │                                          │
                    │  zt → "how much to keep from past"        │
                    │  rt → "how much past to expose"           │
                    │  nt → "what new info to add"              │
                    └────────────────────────┬─────────────────┘
                                             │
                                         ht (output)

  Key: σ = sigmoid, ⊙ = element-wise multiply
  Only 2 gates (vs LSTM's 3) → fewer params → fits on FPGA
```

**[WHITEBOARD: DFIG GRU model — detailed I/O with recursive feedback]**
```
                     DFIG GRU Model (per 6-turbine bundle)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Sequence: t-4, t-3, t-2, t-1, t  (seq_len = 5)           │
    │                                                             │
    │  ┌─── Mechanical ───┐  ┌──── Electrical ────┐              │
    │  │ Vw  (wind speed)  │  │ Vabc (grid voltage) │              │
    │  │ Tm  (torque)    ◄─┤  │ P    (active power) │              │
    │  │ ωr  (rotor spd) ◄─┤  │ Q    (reactive pwr) │              │
    │  └───────────────────┘  └────────────────────┘              │
    │          │ recursive          │                              │
    │          │ features           │                              │
    │          ▼                    ▼                              │
    │  ┌──────────────────────────────────────────┐               │
    │  │        GRU Hidden Layer                   │               │
    │  │        hidden_size = 30                   │               │
    │  │        1 layer, dropout = 0.2             │               │
    │  └──────────────────┬───────────────────────┘               │
    │                     │                                       │
    │                     ▼                                       │
    │  ┌──── Output ──────────────────────────────┐               │
    │  │ Iabc (3-phase grid current) ──→ to grid  │               │
    │  │ Tm   (torque)    ──→ fed back as input ──┤───┐           │
    │  │ ωr   (rotor spd) ──→ fed back as input ──┤───┤           │
    │  │ P    (active power)                      │   │           │
    │  │ Q    (reactive power)                    │   │           │
    │  └──────────────────────────────────────────┘   │           │
    │         ▲                                       │           │
    │         └───── recursive feedback ──────────────┘           │
    │                (used as input at t+1)                        │
    │                                                             │
    │  Norton equiv output: Iabc → 3-phase current source to grid │
    └─────────────────────────────────────────────────────────────┘

    Why recursive features?
    ─────────────────────
    Tm (torque) and ωr (rotor speed) are INTERNAL to the turbine —
    not directly measurable from the grid during real-time emulation.
    The model must predict them AND use them as inputs for the next step.
    This creates the autoregressive loop that captures rotor dynamics.
```

**[WHITEBOARD: Battery GRU and PV MLP for comparison]**
```
    Battery GRU Model (3×4 array)          PV MLP Model (4×4 panel array)
    ─────────────────────────────          ─────────────────────────────
    ┌─ Thermal ──┐ ┌─ Electrical ┐        ┌─ Input ─────────────────┐
    │ dQ/dT       │ │ Vt (voltage) │        │ Irr1, Irr2, ... Irr16  │
    │ dE/dT       │ │ Iout         │        │ (16 panel irradiances)  │
    │ Ta (ambient) │ │ Ii  ◄────┐  │        │ Vt (terminal voltage)   │
    └─────────────┘ │ SOC ◄────┤  │        └────────────┬────────────┘
                    └──────────┘  │                     │
          ↓ recursive features    │          ┌──────────▼──────────┐
    ┌──────────────────┐          │          │  4 Hidden Layers     │
    │  GRU Layer        │          │          │  × 64 neurons each   │
    │  hidden = 20      │          │          │  ReLU activation     │
    │  seq_len = 3      │          │          └──────────┬──────────┘
    └────────┬─────────┘          │                     │
             ▼                    │          ┌──────────▼──────────┐
    ┌─── Output ───────┐          │          │  Output: Iout        │
    │ Iout              │          │          │  (panel current)     │
    │ Ii  ──────────────┤──────┘  │          └─────────────────────┘
    │ SOC ──────────────┤──────┘  │
    └──────────────────┘          │          No recursive features!
                                             PV is memoryless: I = f(V, Irr)
    Recursive: SOC & Ii fed back             Static nonlinear mapping
    (battery state evolves over time)
```

---

#### Step 4: Hyperparameter Selection — Data-Driven Decisions

> "The model architecture was carefully tuned, not guessed."

**[WHITEBOARD: hyperparameter tuning results from paper Fig. 4]**
```
    DFIG Hidden Size Selection              DFIG Sequence Length Selection
    (Fig. 4a from paper)                    (Fig. 4b from paper)

    MSELoss(%)                              MSELoss(%)
    0.16 │  ╲ hidden=5                      0.07 │  ╲ seq=2
    0.14 │   ╲                              0.06 │   ╲
    0.12 │    ╲ hidden=10                   0.05 │    ╲ seq=3
    0.10 │     ╲                            0.04 │     ╲ seq=4
    0.08 │      ╲──hidden=15               0.03 │      ╲───seq=5,6,7... (plateau)
    0.06 │        ╲─hidden=20,25                │         ─────────────────
    0.04 │          ╲─hidden=30,35          0.02 │
    0.02 │            ────────────────           │
         └────────────────────────          └────────────────────────
          0    20    40    60   100                0    20    40    60   100
                  Epoch                                    Epoch

    Decision: hidden=30 (diminishing          Decision: seq_len=5 (diminishing
    returns beyond 30, and 30 fits            returns beyond 5, and longer
    FPGA DSP budget)                          sequences need more BRAM storage)
```

**Training configuration summary:**
```
    ┌────────────────┬──────────────┬──────────────┬──────────────┐
    │                │  DFIG (GRU)  │ Battery(GRU) │   PV (MLP)   │
    ├────────────────┼──────────────┼──────────────┼──────────────┤
    │ Hidden size    │     30       │     20       │   64 × 4     │
    │ Layers         │   1 GRU      │   1 GRU      │  4 hidden    │
    │ Seq length     │     5        │     3        │    N/A       │
    │ Dropout        │    0.2       │    0.2       │    N/A       │
    │ Learning rate  │   0.001      │   0.001      │ 1e-5→5e-6   │
    │ Batch size     │   1000       │   1000       │   1000       │
    │ Epochs         │   1000       │    100       │   1600       │
    │ Optimizer      │   Adam       │   Adam       │   Adam       │
    │ Activation     │ sigmoid+tanh │ sigmoid+tanh │    ReLU      │
    │ Final MSELoss  │   0.02%      │  0.00078%    │  ~0.01%      │
    ├────────────────┼──────────────┼──────────────┼──────────────┤
    │ Recursive feats│  Tm, ωr      │  SOC, Ii     │    None      │
    │ Data source    │ Simulink MC  │ Simulink MC  │ C++ sim MC   │
    │ Data distrib.  │ N(10,3) m/s  │    varied    │N(1000,300)W/m²│
    │ Fault data     │ 5% of dataset│    N/A       │    N/A       │
    └────────────────┴──────────────┴──────────────┴──────────────┘
    MC = Monte Carlo sampling
```

> "A few things worth highlighting: the variable learning rate for PV (starting at 1e-5, halved to 5e-6 at epoch 800) was needed because the I-V curve has sharp nonlinearities near the maximum power point. The 5% fault scenarios in DFIG training data ensure the model handles grid faults it hasn't seen — validated on 80ms and 200ms fault durations (20% shorter and 100% longer than training)."

---

#### Step 5: Multi-Time-Step Coupling & FPGA Synchronization

> "A key contribution is the hybrid EMT-TS simulation with a novel synchronization mechanism on FPGA."

**[WHITEBOARD: Multi-time-step architecture and FPGA sync]**
```
    Multi-Time-Step Hybrid EMT-TS Architecture
    ═══════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────┐
    │  IEEE 118-Bus Network (TS solver, Δt = 5ms)              │
    │  → 1 calculation per system step                         │
    │  → Latency: 30µs (3000 FPGA cycles @ 10ns clock)        │
    └──────────┬───────────────────────────────────────────────┘
               │ data exchange every 5ms
               │ (bus voltages ↓, currents ↑)
    ┌──────────▼───────────────────────────────────────────────┐
    │  ML-Based IBR Microgrid (EMT, Δt = 50µs)                │
    │  → 100 calculations per system step  (5ms / 50µs = 100) │
    │                                                          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
    │  │ Wind GRU │  │  PV MLP  │  │ Batt GRU │               │
    │  │ Lat: 15µs│  │ Lat: 6µs │  │ Lat: 10µs│               │
    │  │ ×100     │  │ ×100     │  │ ×100     │               │
    │  │=1500µs   │  │=600µs    │  │=1000µs   │               │
    │  └──────────┘  └──────────┘  └──────────┘               │
    └──────────────────────────────────────────────────────────┘

    FPGA Data Synchronization (1 system time-step):
    ═══════════════════════════════════════════════
    Global Clock: 1 cycle = 10 ns

    DFIG:      |████████ 100×1500 = 150,000 cycles ████████|
    PV:        |█████ 100×1000 = 100,000 cycles █████|wait |
    Battery:   |███ 100×600 = 60,000 cycles ███|  wait     |
    IEEE 118:  |█ 1×3000 = 3000 cycles █|      wait        |
               ├───────────────────────────────────────┤───┤
               0                              1.5ms    Data  Next
                                            (=150K×10ns) Xchg Step

    System Equivalent Time-Step = 1.5 ms
    (dominated by wind farm: 100 × 15µs = 1500µs = 1.5ms)

    FTRT Ratio = 5ms / 1.5ms = 3.33 (faster-than-real-time!)
```

> "The synchronization is critical: each ML model computes at its own rate, storing outputs in buffers. A global clock governs control logic. The wind farm is the bottleneck (15µs × 100 = 1.5ms), so the system equivalent time-step is 1.5ms — yielding a 3.33× faster-than-real-time ratio for the overall system. This is analogous to multi-scale simulation in EM, where you might have fine resolution near an antenna and coarser resolution in the far field."

> "Between data exchanges, the bus voltage from the 118-bus network stays constant for 100 EMT steps. During this period, the ML models keep iterating over their recursive features (Tm, ωr for wind; SOC, Ii for battery), producing the transient dynamics."

---

#### Step 6: FPGA Deployment Pipeline

**[WHITEBOARD: Full deployment pipeline with quantization details]**
```
    Training (PyTorch, Float32)          Deployment (FPGA, Fixed-Point)
    ═══════════════════════════          ════════════════════════════════

    ┌──────────────┐                     ┌─────────────────────────────┐
    │ Float32 Model │                     │  Step 1: Quantization       │
    │ (PyTorch)     │ ──────────────────→ │                             │
    └──────────────┘                     │  GRU: Dynamic quantization  │
                                         │    Q(x) = round(x/Δ) + Z   │
                                         │    Δ,Z computed at runtime  │
                                         │    (activations change per  │
                                         │     input sequence)         │
                                         │                             │
                                         │  MLP: Static quantization   │
                                         │    Q(x) = round(x/Δ) + Z   │
                                         │    Δ,Z fixed after training │
                                         │    (feedforward = constant  │
                                         │     activation ranges)      │
                                         │                             │
                                         │  Data type: ap_int<8>       │
                                         │  (HLS 8-bit fixed-point)    │
                                         └────────────┬────────────────┘
                                                      │
                                         ┌────────────▼────────────────┐
                                         │  Step 2: LUT Activations    │
                                         │                             │
                                         │  sigmoid LUT:               │
                                         │    range [-10, 10]          │
                                         │    step = 0.001             │
                                         │    = 20,001 entries         │
                                         │                             │
                                         │  tanh LUT:                  │
                                         │    range [-6, 6]            │
                                         │    step = 0.001             │
                                         │    = 12,001 entries         │
                                         │                             │
                                         │  Pre-computed input→output  │
                                         │  (eliminates exp() on FPGA) │
                                         └────────────┬────────────────┘
                                                      │
                                         ┌────────────▼────────────────┐
                                         │  Step 3: FPGA Optimization  │
                                         │                             │
                                         │  • Loop pipelining          │
                                         │    (not unrolling — balance │
                                         │     resources vs latency)   │
                                         │  • Vitis HLS synthesis      │
                                         │  • Buffered data exchange   │
                                         └────────────┬────────────────┘
                                                      │
                                         ┌────────────▼────────────────┐
                                         │  Xilinx VCU118 Board        │
                                         │  UltraScale+ XCVU9P FPGA   │
                                         │                             │
                                         │  Resources Used:            │
                                         │  DSP:  93.3% of 6,840      │
                                         │  FF:   24.42% of 2.36M     │
                                         │  LUT:  61.3% of 1.18M      │
                                         │  BRAM: 4,320 available      │
                                         └─────────────────────────────┘

    Quantization Error: < 0.01% (float vs fixed-point output)
```

**[WHITEBOARD: Resource savings — ML vs Traditional on FPGA]**
```
    FPGA Resource Utilization: ML-Based vs Traditional Models
    ══════════════════════════════════════════════════════════

             DFIG                   Battery                  PV
    DSP:  1.3% vs 7.3%      DSP: 2.0% vs 10.53%    DSP: 1.3% vs 23.6%
    FF:   0.42% vs 2.0%     FF:  0.37% vs 1.9%     FF:  0.42% vs 18.72%
    LUT:  0.9% vs 6.0%      LUT: 0.9% vs 6.77%     LUT: 0.9% vs 38.64%

    → ML models use 3-30x fewer FPGA resources per component
    → This is WHY we can fit 30 wind + 20 PV + battery + 118-bus on ONE chip
```

---

#### Step 7: Results Summary

> "The wind farm model runs at 15 microseconds per timestep on FPGA, achieving 3.33x faster-than-real-time (50µs timestep / 15µs latency). DFIG accuracy is 0.02% MSELoss. And the approach scales massively:"

**[WHITEBOARD: Scalability — the key advantage]**
```
    Execution Time per 50µs Timestep vs Number of Wind Turbines
    ════════════════════════════════════════════════════════════

    Exec Time (µs)
    10⁶ │                                          ╱ Traditional
        │                                        ╱   (linear growth)
    10⁵ │                                      ╱
        │                                    ╱     456,400 µs
    10⁴ │                                  ╱
        │                                ╱         4,464 µs
    10³ │                              ╱
        │               ╱────────────────────────  ML-based
    10² │             ╱                            (nearly flat!)
        │     ╱──────    55.7 µs
    10¹ │───── 15 µs                               55.7 µs
        │
        └──────────┬──────┬──────┬──────┬──────
                   1     10    100   1000  10000
                         Number of Wind Turbines

    At 10,000 turbines: 456,400 / 55.7 = 8,193× speedup!

    Why ML scales better:
    Traditional → solve coupled nonlinear equations (O(n) per turbine)
    ML-based → batch-parallel identical GRU copies (O(1) per batch)
```

**Key result numbers to remember:**
```
    ┌─────────────────────────────────┬─────────────┐
    │ Metric                          │ Value        │
    ├─────────────────────────────────┼─────────────┤
    │ Wind FTRT ratio                 │ 3.33×        │
    │ Battery FTRT ratio              │ 5.0×         │
    │ PV FTRT ratio                   │ 8.33×        │
    │ System equivalent timestep      │ 1.5ms        │
    │ DFIG MSELoss                    │ 0.02%        │
    │ Battery MSELoss                 │ 0.00078%     │
    │ PV error (normal)               │ 0.2%         │
    │ PV error (partial shade)        │ 4%           │
    │ Quantization error              │ < 0.01%      │
    │ Scalability at 10K turbines     │ 8,193× faster│
    │ Wind error near transitions     │ < 1%         │
    └─────────────────────────────────┴─────────────┘
```

---

### Paper 2: DAC 2026 (Under Review) — ~2 minutes

> "My second line of work is on diffusion models for optimization. At DAC, I apply diffusion to chip placement — the problem of positioning circuit modules on a canvas to minimize wirelength. This is directly in Synopsys's EDA domain."

---

#### Step 1: Problem Formulation — Chip Placement as Energy Minimization

> "Chip placement finds optimal 2D positions for circuit components to minimize wirelength while avoiding overlaps. I formulate it as energy minimization over continuous coordinates."

**[WHITEBOARD: problem setup and energy function]**
```
    Problem: Given netlist G = (V, E) with N components and nets,
    find positions x = {(xi, yi)} on canvas W×H minimizing energy E(x)

    All positions normalized to [-1, 1] × [-1, 1] canvas

    Energy Function (3 differentiable terms):
    ════════════════════════════════════════

    E(x) = E_wire(x) + λ_overlap · E_overlap(x) + λ_bound · E_bound(x)
            ────────     ─────────────────────     ────────────────────
            HPWL         Pairwise overlap          Canvas boundary
            (wirelength)  penalty                   penalty

    E_wire = Σ [max(xi) - min(xi) + max(yi) - min(yi)]   (per net)
              e∈E

    E_overlap = Σ  [min(0, d_ij(x))]²     d_ij = signed distance
               i<j                          between rectangles
                                            (< 0 when overlapping)

    E_bound = Σ [max(0, |xi|+wi/2 - 1)² + max(0, |yi|+hi/2 - 1)²]
              i

    Selected weights: λ_overlap = 2.0, λ_bound = 1.0
    (tuned via empirical sweep — see ablation)

    Quality metrics:
    • Legality = Au/As (union area / sum area), target = 1.0
    • HPWL = half-perimeter wirelength (proxy for routed wirelength)
```

---

#### Step 2: Core Contribution — Dense Per-Step Energy Feedback

> "The key problem with existing approaches: RL methods (CircuitTraining) place components sequentially and can't revert bad early decisions. Supervised diffusion (ChipDiffusion) needs expensive optimal placement examples. My approach solves both."

**[WHITEBOARD: Sparse vs Dense reward — the key innovation]**
```
    Prior work (SDDS, DDPO): Sparse Terminal Reward
    ════════════════════════════════════════════════
    t=T          t=T-1    ...    t=1          t=0
    (noise) ───→ ───────────────→ ──────────→ (final)
                                               │
                                          R(x₀) = -E(x₀)
                                          ONE reward signal
                                          must propagate back
                                          through T=1000 steps
                                          via GAE decay (γλ)^l
                                          → HIGH variance
                                          → SLOW convergence

    Our approach: Dense Per-Step Energy Feedback
    ════════════════════════════════════════════
    t=T          t=T-1    ...    t=1          t=0
    (noise) ───→ ───────────────→ ──────────→ (final)
      │            │                │            │
    -E(xT)      -E(xT-1)        -E(x1)      -E(x0)
    compute      compute          compute      compute
    energy       energy           energy       energy
    HERE         HERE             HERE         HERE

    Per-step reward:
    r_t = -E(x_t)                    ← immediate quality feedback
        + α_noise · log p(xt|xt-1)   ← stay close to forward kernel
        + α_ent · H(qθ(xt-1|xt))     ← maintain exploration

    Result: 3.2× lower gradient variance, 2.8× fewer epochs to converge
```

> "Each temporal difference δt = rt + γV(xt+1) - V(xt) now incorporates immediate placement quality rather than relying solely on value function estimates. This is why dense feedback is so powerful — the advantage function Ât gets real signal at every step, not just exponentially decayed terminal reward."

---

#### Step 3: Diffusion Framework — From Discrete to Continuous

> "I extend the SDDS framework from discrete binary states to continuous 2D coordinates, treating the reverse diffusion as an RL policy."

**[WHITEBOARD: The RL-diffusion connection]**
```
    Diffusion ↔ Reinforcement Learning Mapping
    ═══════════════════════════════════════════

    Diffusion Concept          RL Concept
    ──────────────────         ──────────────────
    Reverse model qθ(xt-1|xt)  Policy π(a|s)
    Trajectory xT:0            Episode
    Single denoising step      Action
    Current noisy state xt     State
    Energy E(xt)               (Negative) reward
    All T steps                Horizon

    Forward process (adds noise):
    xt = √(1-βt) · xt-1 + √βt · ε,    ε ~ N(0,I)
    (cosine schedule, βt ∈ [0.0001, 0.02])

    Reverse process (learned policy):
    qθ(xt-1|xt, G, t) = N(μθ(xt, G, t), σ²θ(xt, G, t) · I)

    Training objective (PPO):
    J(θ) = E[Σ rt(xt, xt-1)]   maximized via clipped surrogate

    Key PPO hyperparameters:
    ε_clip = 0.2, λ_GAE = 0.95, γ = 0.99
    4 PPO epochs per batch, minibatch = 8
    gradient clipping max norm = 0.5
```

> "Unlike sequential RL (CircuitTraining) that places one component at a time and can't undo mistakes, diffusion refines ALL components simultaneously at every step. It's a parallel optimization over the full placement."

---

#### Step 4: Architecture — GNN Encode-Process-Decode

**[WHITEBOARD: Model architecture]**
```
    Encode-Process-Decode Architecture
    ═══════════════════════════════════

    ┌─── INPUT ────────────────────────────────────────────────┐
    │  Per node: [wi, hi, xt, yt, sin(t), cos(t)]  → 6D      │
    │  Edges: net connectivity from netlist G                  │
    └──────────────────────────┬───────────────────────────────┘
                               │
    ┌──────────────────────────▼───────────────────────────────┐
    │  ENCODER (MLP layers)                                    │
    │  6D → 256D embeddings                                    │
    └──────────────────────────┬───────────────────────────────┘
                               │
    ┌──────────────────────────▼───────────────────────────────┐
    │  PROCESSOR (GNN Message Passing)                         │
    │  16 layers × 256 hidden dimensions                       │
    │  Mean aggregation over neighbors                         │
    │  z^(l+1) = f(z^l, m)  where m = mean({z^l_j : j ∈ N(i)})│
    │                                                          │
    │  Why GNN? Circuit is a graph (components = nodes,        │
    │  nets = hyperedges). Message passing captures both       │
    │  local neighborhoods AND global circuit structure.        │
    └──────────────────────────┬───────────────────────────────┘
                               │
    ┌──────────────────────────▼───────────────────────────────┐
    │  DECODER (256D → outputs)                                │
    │                                                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
    │  │ μθ(x,y)       │  │ σ²θ(x,y)     │  │ Vθ(state)      │ │
    │  │ mean positions │  │ variances     │  │ value function │ │
    │  │ (per component)│  │ (per comp.)   │  │ (shared encoder│ │
    │  └──────────────┘  └──────────────┘  │  separate MLP)  │ │
    │                                      └────────────────┘ │
    │  Sample: xt-1 ~ N(μθ, σ²θ · I)                          │
    └──────────────────────────────────────────────────────────┘
```

---

#### Step 5: Synthetic Training Data — No Optimal Examples Needed

> "A key advantage: we train entirely on synthetic data without needing any optimal placements."

**[WHITEBOARD: Training data pipeline]**
```
    Synthetic Data Generation (20,000 circuits)
    ════════════════════════════════════════════

    Step 1: Generate random components
    ┌──────────────────────────────────────────┐
    │  N ~ Uniform(200, 1000) components       │
    │  Sizes: si ~ ClippedExp(λ, 0.01, 1.0)   │
    │    λ_size ~ Uniform(0.04, 0.08)          │
    │  → heavy-tailed: mostly small cells,     │
    │    occasional large macros (realistic)    │
    │  Density: Σ(wi·hi) ≤ ρ·A_canvas          │
    │    ρ ~ Uniform(0.75, 0.9)                │
    └──────────────────────────────────────────┘
                    │
    Step 2: Create temporary legal placement (collision detection)
                    │
    Step 3: Generate realistic netlist connectivity
    ┌──────────────────────────────────────────┐
    │  Local proximity (60%):                  │
    │    k=2-4 nearest neighbors               │
    │  Hierarchical clusters (30%):            │
    │    k-means, inter-cluster connections    │
    │  Long-range (10%):                       │
    │    random pairs (critical paths)         │
    │  30% of 2-pin nets merged into           │
    │    multi-pin (60% 3-pin, 30% 4-pin,      │
    │    10% 5-pin)                             │
    └──────────────────────────────────────────┘
                    │
    Step 4: DESTROY the legal placement → random positions
    ┌──────────────────────────────────────────┐
    │  xi, yi ~ Uniform(-1, 1)                 │
    │  Creates overlaps + poor wirelength       │
    │  Model must learn to fix this via         │
    │  energy-guided denoising                 │
    └──────────────────────────────────────────┘

    Why this works for zero-shot transfer:
    • Covers realistic size distributions, density, connectivity
    • No optimal examples needed — learns from energy function
    • 5K circuits already beats DREAMPlace (Table 5 in paper)
```

---

#### Step 6: Legalization Decoder — Post-Processing

> "The diffusion model produces near-legal placements (legality 0.953 before legalization). A lightweight force-directed decoder resolves remaining overlaps."

```
    Force-Directed Legalization
    ═══════════════════════════
    • Spatial hashing: partition canvas into grid cells
    • Only check overlaps with neighboring cells (efficient)
    • Repulsive forces proportional to penetration depth
    • Boundary violations resolved via gradient descent on E_bound
    • Adaptive step size: η = min(0.1, 1/√iteration)
    • Converges in ~23 iterations on average (15-50 range)
    • HPWL degradation from legalization: only 1.1%

    vs ChipDiffusion: 20,000 gradient-based optimization steps
    with carefully tuned hyperparameters → our decoder is MUCH simpler
```

---

#### Step 7: Results Summary

**[WHITEBOARD: Key results tables]**
```
    IBM Benchmarks (HPWL averages):
    ═══════════════════════════════
    ┌──────────────────┬──────────┬────────┬────────┬──────────┐
    │ Method           │ Clustered│ Macro  │ Mixed  │ Time     │
    │                  │ (×10⁷)  │ (×10⁵) │ (×10⁶) │ (min)    │
    ├──────────────────┼──────────┼────────┼────────┼──────────┤
    │ DREAMPlace       │  3.72    │   –    │  23.6  │  0.48    │
    │ ChiPFormer       │  3.15    │  7.33  │  28.9  │  124     │
    │ ChipDiffusion    │  2.98    │  2.49  │  22.7  │  4.4     │
    │ Ours             │  2.84    │  2.35  │  21.8  │  4.4     │
    ├──────────────────┼──────────┼────────┼────────┼──────────┤
    │ Improvement      │ -4.6%    │ -5.6%  │ -4.0%  │  same    │
    │ vs ChipDiffusion │          │        │        │          │
    └──────────────────┴──────────┴────────┴────────┴──────────┘

    ISPD2005 Average HPWL (×10⁵):
    ChiPFormer: 116  →  WireMask: 154  →  ChipDiffusion: 45.9  →  Ours: 44.2
    (3.7% better than ChipDiffusion, zero-shot from synthetic data)

    Key Ablation Results:
    ┌────────────────────────────────────────────────────────────┐
    │ Dense vs Sparse:    3.2× lower gradient variance           │
    │                     2.8× fewer epochs (1000 vs 2800)       │
    │ Legalization:       only 1.1% HPWL degradation             │
    │                     raw model legality already 0.953       │
    │ Data efficiency:    5K circuits → 3.15 (beats DREAMPlace)  │
    │                     20K circuits → 2.84 (best result)      │
    │ Penalty weights:    λ_overlap=2.0, λ_bound=1.0 optimal     │
    │                     too low → poor legality (0.892)         │
    │                     too high → +8.2% HPWL                  │
    └────────────────────────────────────────────────────────────┘
```

**Connection to this role:**
> "The relevance here is twofold: it's directly in Synopsys's EDA domain, and it demonstrates I can train generative models with physics-based reward signals instead of labeled data — which matters when optimal simulation examples are expensive to obtain. The dense per-step feedback idea maps directly to HFSS: compute a quality metric (e.g., field error, mesh quality) at every iteration of an adaptive solver, not just at convergence."

---

### Paper 3: EDISCO — ICML 2026 (Under Review)

> "My ICML submission tackles the Traveling Salesman Problem with the first E(2)-equivariant diffusion model. The insight is that rotating or reflecting a set of cities doesn't change the optimal tour, so the neural network should respect that symmetry. Let me walk through the methodology."

---

#### Step 1: Problem Formulation — TSP as Generative Modeling

> "Given n cities with coordinates c_i in R^2, TSP asks for a binary adjacency matrix X in {0,1}^{n×n} that forms the shortest Hamiltonian cycle. We reformulate this as learning a conditional distribution p(X | {c_i})."

**Key Idea — [WHITEBOARD: edge-based representation]**
```
  TSP with 4 cities:              Binary Adjacency Matrix X:

    c1 ──── c2                     j=1  j=2  j=3  j=4
    │        │                i=1 [ 0    1    0    1  ]
    │        │                i=2 [ 1    0    1    0  ]
    c4 ──── c3                i=3 [ 0    1    0    1  ]
                              i=4 [ 1    0    0    0  ]
  Tour: 1→2→3→4→1
                              Constraints:
  Objective:                    • Each city degree = 2
  X* = argmin_X f(X, {c_i})    • Selected edges form connected cycle
  s.t. X ∈ C                   • X_ij ∈ {0, 1}
```

> "The key observation is that TSP solutions are invariant under E(2) transformations — rotations, reflections, and translations. If you rotate all cities by 90°, the optimal tour is the same set of edges, just rotated. Existing methods ignore this and waste capacity learning this invariance from data."

**[WHITEBOARD: E(2) invariance]**
```
  Original          Rotated 90°       Reflected          Translated
  • → • → •         •                 • → • → •          • → • → •
  ↑       ↓         ↓                 ↓       ↑          ↑       ↓
  • ← • ← •         • → • → •        • ← • ← •          • ← • ← •

  ALL have the SAME optimal tour (same edge set)!

  E(2) = translations (R²) ⋊ rotations+reflections (O(2))
       = 3 continuous degrees of freedom (2 translation + 1 rotation)
       + discrete Z₂ reflection

  Equivariant model: f(g·x) = g·f(x) for all g ∈ E(2)
  → Output edges INVARIANT under coordinate transforms
```

---

#### Step 2: Continuous-Time Categorical Diffusion Framework

> "Unlike image diffusion that works with continuous pixels, TSP edges are discrete — 0 or 1. I use categorical diffusion operating directly in discrete space, avoiding quantization errors. And critically, I formulate it in continuous time via CTMCs, which enables flexible inference with any numerical solver."

**[WHITEBOARD: forward and reverse process]**
```
  FORWARD PROCESS (corruption):
  ┌─────────────────────────────────────────────────────┐
  │  X₀ (clean tour) ──────────────────→ X₁ (noise)    │
  │  t=0              progressive corruption    t=1     │
  │                                                     │
  │  Rate matrix: Q(t) = β(t) · (1/K · 11ᵀ - I)       │
  │  where K=2 (binary edges), β(t) = 0.1 + 1.4t       │
  │                                                     │
  │  Transition: P(Xₜ=j|X₀=i) = 1/2 + (δᵢⱼ - 1/2)    │
  │              · exp(-2 ∫₀ᵗ β(u)du)                  │
  │                                                     │
  │  Key: closed-form! Sample Xₜ from X₀ at ANY t      │
  │  without simulating intermediate states              │
  └─────────────────────────────────────────────────────┘

  REVERSE PROCESS (denoising):
  ┌─────────────────────────────────────────────────────┐
  │  X₁ (noise) ──────────────────→ X₀ (clean tour)    │
  │  t=1          learned denoising steps       t=0     │
  │                                                     │
  │  Neural network sθ(Xₜ, t, {cᵢ}) predicts X₀        │
  │  (x₀-prediction parameterization)                   │
  │                                                     │
  │  Posterior: q(Xₜ₋Δₜ|Xₜ,X₀) via Bayes' rule        │
  │                                                     │
  │  ADAPTIVE MIXING STRATEGY (my contribution):        │
  │  p_reverse = w(t)·p_diffusion + (1-w(t))·p_predict  │
  │  where w(t) = t (linear mixing)                     │
  │                                                     │
  │  High t → trust diffusion (predictions unreliable)  │
  │  Low t  → trust prediction (high confidence)        │
  │  t < 0.1 → deterministic argmax                     │
  └─────────────────────────────────────────────────────┘
```

> "The adaptive mixing is theoretically motivated: Proposition 3.1 shows the posterior variance Var[X₀|Xₜ] scales linearly with t for small t, so the linear weight w(t)=t naturally aligns with prediction uncertainty."

**Why continuous-time matters — [WHITEBOARD: discrete vs continuous]**
```
  Discrete-time (DIFUSCO, T2T):     Continuous-time (EDISCO):
  ┌─────────────────────┐           ┌─────────────────────────┐
  │ Fixed step schedule  │           │ Flexible step schedule   │
  │ T=120 or T=20 steps │           │ Choose ANY number of     │
  │ Retrain for new T   │           │ steps at inference!      │
  │ Euler-only solver    │           │ PNDM, DEIS, RK4, Heun...│
  │                     │           │ No retraining needed     │
  └─────────────────────┘           └─────────────────────────┘

  Benefit: 50-step PNDM for best quality (1.95% gap)
           5-step DEIS-2 for fast inference (2.78% gap, 9× faster)
           Same trained model for both!
```

---

#### Step 3: E(2)-Equivariant GNN Architecture (EGNN)

> "This is the core architectural contribution. The EGNN maintains three types of features — node features hᵢ, edge features eᵢⱼ, and coordinates xᵢ — and processes them through message passing layers that guarantee E(2)-equivariance."

**[WHITEBOARD: EGNN layer architecture]**
```
  ┌──────────────────────────────────────────────────────────────┐
  │                    EGNN Layer ℓ                               │
  │                                                              │
  │  INPUT: hᵢ⁽ℓ⁾ (node), eᵢⱼ⁽ℓ⁾ (edge), xᵢ⁽ℓ⁾ (coords)       │
  │                                                              │
  │  1. MESSAGE (invariant):                                     │
  │     mᵢⱼ = MLP_m([hᵢ, hⱼ, eᵢⱼ, ‖xᵢ-xⱼ‖₂])                 │
  │                    ↑                                         │
  │              distance is E(2)-INVARIANT                      │
  │                                                              │
  │  2. COORDINATE UPDATE (equivariant):                         │
  │     Δxᵢ = α · Σⱼ wᵢⱼ · (xⱼ-xᵢ)/‖xⱼ-xᵢ‖₂                 │
  │     where wᵢⱼ = tanh(MLP_c(mᵢⱼ)/τ)                         │
  │     α = 0.1 (step size), τ = 10 (temperature)               │
  │                                                              │
  │     WHY equivariant: normalized direction (xⱼ-xᵢ)/‖·‖      │
  │     transforms correctly under rotation!                     │
  │     Δ(Rx) = R·Δx  ✓                                         │
  │                                                              │
  │  3. EDGE UPDATE (invariant, time-conditioned):               │
  │     eᵢⱼ⁽ℓ⁺¹⁾ = LN(eᵢⱼ + MLP_e([eᵢⱼ, mᵢⱼ]) + MLP_t(t_emb))│
  │                                           ↑                  │
  │                                   sinusoidal encoding        │
  │                                                              │
  │  4. NODE UPDATE (invariant, gated attention):                │
  │     hᵢ⁽ℓ⁺¹⁾ = LN(hᵢ + MLP_h([hᵢ, Σⱼ σ(mᵢⱼ)⊙hⱼ]))         │
  │                                                              │
  │  OUTPUT: hᵢ⁽ℓ⁺¹⁾, eᵢⱼ⁽ℓ⁺¹⁾, xᵢ⁽ℓ⁺¹⁾                       │
  └──────────────────────────────────────────────────────────────┘

  Full architecture: 12 EGNN layers × (64 node, 64 edge, 256 hidden)
  Parameters: ~5.5M (comparable to baselines at 5.3M)
  Time embedding: 128-dim sinusoidal encoding
```

> "The key insight is that distances ‖xᵢ-xⱼ‖₂ are invariant under E(2), so the message function only sees invariant inputs. Coordinate updates use normalized directions which transform equivariantly. The output edge probabilities P(Xᵢⱼ=1) are therefore invariant — rotating inputs produces identical edge predictions."

**[WHITEBOARD: equivariance proof sketch]**
```
  For rotation R and translation t:

  1. Distance invariant:  ‖Rcᵢ+t - Rcⱼ-t‖ = ‖R(cᵢ-cⱼ)‖ = ‖cᵢ-cⱼ‖  ✓
  2. Messages invariant:  m depends only on h, e, distance → m(g) = m  ✓
  3. Coords equivariant:  Δ(Rx) = R·Σ wᵢⱼ·(xⱼ-xᵢ)/‖·‖ = R·Δx     ✓
  4. Edge/node invariant: depend only on invariant quantities           ✓

  Proposition 3.2: E(2)-invariant learning ≡ learning on (2n-3)-dim
  manifold instead of 2n-dim space. E(2) removes 3 DoF:
  2 (translation) + 1 (rotation). Reflections = discrete Z₂.

  Impact: reduced hypothesis class → better sample efficiency
```

---

#### Step 4: Training & Inference Pipeline

> "Training uses weighted cross-entropy with a time-dependent weight that emphasizes reconstruction accuracy near t=0. Inference leverages the continuous-time formulation to use advanced numerical solvers without retraining."

**[WHITEBOARD: training configuration]**
```
  Training Objective:
  L(θ) = E_{t~U(0,1), X₀~p_data, Xₜ~q(Xₜ|X₀)} [(1-√t)·CE(sθ(Xₜ,t), X₀)]
                                                     ↑
                                              weight: emphasize
                                              low-noise regime

  ┌──────────────────────────────────────────────────────────┐
  │  Scale    │ Train Data │ Batch │ Epochs │ GPU    │ Time  │
  │───────────│────────────│───────│────────│────────│───────│
  │  TSP-50   │ 500K       │  64   │  100   │ 1×A6000│ 19h   │
  │  TSP-100  │ 500K       │  32   │  100   │ 1×A6000│ 69h   │
  │  TSP-500  │  60K       │  16   │   50   │ 1×A6000│ 31h   │
  │  TSP-1000 │  30K       │   8   │   50   │ 1×A6000│ 61h   │
  │  TSP-10000│   3K       │   4   │   50   │ 1×A6000│ 18h   │
  └──────────────────────────────────────────────────────────┘

  Key efficiency: 33-50% LESS training data than baselines (500K vs 1.5M)
  Single GPU training across ALL scales (graph sparsification for n>100)

  Optimizer: AdamW, lr=2e-4, weight decay=1e-5, cosine annealing
  Gradient clipping at unit norm
  Curriculum learning: TSP-500/1000/10000 init from smaller checkpoints
```

**[WHITEBOARD: solver flexibility at inference]**
```
  Multi-step methods adapted for categorical CTMCs:

  1. Adams-Bashforth prediction smoothing (PNDM):
     p̃(X₀) = (55p̂⁽ᵗ¹⁾ - 59p̂⁽ᵗ²⁾ + 37p̂⁽ᵗ³⁾ - 9p̂⁽ᵗ⁴⁾) / 24

  2. Exact CTMC posterior sampling:
     q(Xs=1|Xt, p̃) via Bayes' rule with closed-form transitions

  Solver Comparison on TSP-500 (greedy decoding):
  ┌────────────────────────────────────────────┐
  │ Solver        │ Steps │ Gap%  │ Time (min) │
  │───────────────│───────│───────│────────────│
  │ PNDM          │  50   │ 1.95  │   2.19     │ ← best quality
  │ Heun/RK2      │  10   │ 1.99  │   0.83     │ ← best efficiency
  │ RK4           │   5   │ 1.97  │   0.82     │
  │ DEIS-2        │   5   │ 2.78  │   0.23     │ ← fastest
  │ Euler         │  10   │ 3.14  │   0.45     │
  │───────────────│───────│───────│────────────│
  │ DIFUSCO(disc) │ 120   │ 9.41  │   5.70     │ ← baseline
  │ T2T (disc)    │  20   │ 5.09  │   4.90     │
  └────────────────────────────────────────────┘
```

---

#### Step 5: Tour Decoding & Graph Sparsification

> "The diffusion model outputs a probability matrix P ∈ [0,1]^{n×n}. We decode this into a valid TSP tour using greedy construction with optional 2-opt local search refinement."

**[WHITEBOARD: tour construction pipeline]**
```
  Diffusion Output → Edge Scores → Greedy Construction → 2-opt (optional)

  Edge scoring: sᵢⱼ = (Pᵢⱼ + Pⱼᵢ) / dᵢⱼ
                        ↑ symmetrize    ↑ distance-weight

  Greedy decoder:
  1. Sort edges by score (descending)
  2. Insert edge (i,j) if:
     - Both vertices have degree < 2
     - No subtour created (union-find with path compression)
  3. Last edge completes the Hamiltonian cycle

  Optional: 2-opt local search post-processing
  → Iteratively swap edge pairs to reduce tour length

  Graph Sparsification for n > 100:
  ┌──────────────────────────────────────────────┐
  │  TSP-50/100:  Dense adjacency (complete graph)│
  │  TSP-500:     k=50 nearest neighbors          │
  │  TSP-1000:    k=100 nearest neighbors          │
  │  TSP-10000:   k=100 nearest neighbors          │
  │                                                │
  │  Complexity: O(n²) → O(kn)                     │
  │  E(2)-equivariance preserved:                  │
  │  k-NN selection is rotation-invariant!         │
  └──────────────────────────────────────────────┘
```

---

#### Step 6: Cross-Distribution Generalization (Key Differentiator)

> "This is the most impressive result and the one that matters most for Synopsys. We train ONLY on uniform-random cities and test on completely different distributions. EDISCO shows only 4% average degradation, versus 133% for DIFUSCO and 1961% for Fast-T2T."

**[WHITEBOARD: OOD robustness]**
```
  Train on UNIFORM → Evaluate on 4 distributions:

  Uniform         Cluster          Explosion        Implosion
  ·  · ·  ·       ··  ···          · ·   ·  ·       ·  · ·  ·
  · ·  · ·         ··  ··         ·         ·        ···  ···
  ·  ·  · ·       ···  ··        ·    · ·    ·        ·····
  · ·  ·  ·        ··  ···       ·         ·        ···  ···
                                  · ·   ·  ·       ·  · ·  ·

  Cross-Distribution Degradation: Det = (Gap_OOD/Gap_Uniform - 1)×100%

  ┌────────────────────────────────────────────────────────┐
  │ Method     │ Uniform │ Cluster │ Explosion │ Avg Det  │
  │────────────│─────────│─────────│───────────│──────────│
  │ EDISCO     │  0.04%  │  0.05%  │   0.03%   │    4%   │ ★
  │ GLOP       │  0.09%  │  0.17%  │   0.07%   │   15%   │
  │ DIFUSCO    │  1.01%  │  2.87%  │   1.38%   │  133%   │
  │ T2T        │  0.18%  │  1.50%  │   0.15%   │  687%   │
  │ Fast-T2T   │  0.06%  │  1.18%  │   0.03%   │ 1961%   │
  └────────────────────────────────────────────────────────┘

  WHY: Equivariance forces learning DISTANCE-BASED features
       (invariant to coordinate frame) rather than memorizing
       absolute position patterns specific to one distribution.
```

> "This is directly relevant to Synopsys: an EM simulator trained on one class of antenna geometries should generalize to rotated, scaled, or differently-shaped structures. Equivariance provides that robustness by construction rather than hoping data augmentation covers all cases."

---

#### Step 7: Results Summary

> "EDISCO achieves state-of-the-art across all TSP scales with significant speedups."

**[WHITEBOARD: main results table]**
```
  TSP Benchmarks (Sampling + 2-opt decoding):
  ┌──────────────────────────────────────────────────────────────┐
  │ Method          │ TSP-500       │ TSP-1000      │ TSP-10000 │
  │                 │ Gap%  │ Time  │ Gap%  │ Time  │ Gap%      │
  │─────────────────│───────│───────│───────│───────│───────────│
  │ DIFUSCO         │ 0.83% │ 19.1m │ 1.30% │ 59.5m │ 4.03%    │
  │ T2T             │ 0.37% │ 16.0m │ 0.78% │ 54.7m │ 2.84%    │
  │ Fast-T2T        │ 0.21% │  6.9m │ 0.42% │ 18.3m │   —      │
  │ CADO (SL+RL)    │ 0.12% │ 27.0m │ 0.30% │ 61.5m │ 2.68%    │
  │ EDISCO-PNDM     │ 0.08% │  8.0m │ 0.22% │ 23.5m │ 1.20%    │
  │ EDISCO-DEIS2    │ 0.12% │  1.0m │ 0.35% │  2.8m │ 1.92%    │
  └──────────────────────────────────────────────────────────────┘

  Key Numbers to Remember:
  • TSP-500: 0.08% gap (prev SOTA 0.12%), 3.4× faster than CADO
  • TSP-1000: 0.22% gap (prev SOTA 0.30%), 2.6× faster
  • TSP-10000: 1.20% gap (prev SOTA 2.68%), 55% improvement
  • TSPLIB real-world: 0.088% average gap (prev best 0.133%)
  • Training data: 33-50% less than all baselines
  • Single GPU training across all scales
```

**[WHITEBOARD: ablation — what matters most]**
```
  Ablation on TSP-500 (greedy decoding):
  ┌─────────────────────────────────────────────────────┐
  │ Variant              │ Gap%  │ Time  │ Conv. Epoch  │
  │──────────────────────│───────│───────│──────────────│
  │ EDISCO (Full)        │ 1.95  │ 2.19m │    35        │
  │ w/o Mix Strategy     │ 2.44  │ 2.18m │    38        │
  │ w/o Continuous-Time  │ 2.86  │ 4.43m │    42        │
  │ w/o EGNN             │ 5.71  │ 2.31m │    51        │ ← BIGGEST hit
  │ Vanilla DIFUSCO      │ 9.41  │ 5.70m │    61        │
  └─────────────────────────────────────────────────────┘

  EGNN is the core contribution:
  • Removing it: 1.95% → 5.71% (2.9× worse)
  • EGNN alone: 3.58% (2.6× better than vanilla DIFUSCO)
  • Continuous-time: 2× faster inference + 0.91% better gap
  • Adaptive mixing: 0.49% improvement + faster convergence

  Also extends to: ESTP (1.5% gap), CVRP (0.38% gap), MIS
```

**Connection to this role:**
> "The relevance is three-fold: (1) EM problems have the same geometric symmetries — rotating an antenna doesn't change its S-parameters — so equivariant architectures should help for HFSS surrogate models. (2) The continuous-time framework provides flexible speed-quality trade-offs, critical for a simulation tool where users need both quick estimates and high-accuracy results. (3) The cross-distribution robustness means a model trained on one family of structures can generalize to unseen geometries — exactly what you need for a practical design tool."

---

### The Connecting Thread — Have This Ready

> "All three papers share a common methodology: **identify the structure in the problem, build it into the architecture, and use that to enable efficient learning.** In JESTIE, the structure was temporal vs. memoryless physics → GRU vs. MLP. In DAC, the structure was iterative refinement with a computable energy → dense per-step feedback. In EDISCO, the structure was geometric symmetry → equivariant architecture. For EM simulation at Synopsys, the same principle applies — Maxwell's equations have rich structure (symmetry, locality, multi-scale) that should be baked into the ML architecture, not learned from scratch."

---

## 4. Paper Defense: Tough Questions & Best Answers

### Paper 1: IEEE JESTIE (ML for EMT Simulation) — YOUR STRONGEST CARD

**Q1 (HARDEST): "Your EMT work uses lumped-parameter circuit models at 50µs timesteps. HFSS solves 3D Maxwell's equations on meshes at GHz. The physics abstractions are fundamentally different. What transfers?"**

> "You're absolutely right — EMT uses lumped circuit equations (Kirchhoff's laws, machine ODEs, converter switching) while HFSS solves distributed-field Maxwell's PDEs on tetrahedral meshes. But the scale and methodology of my work are what transfer. I built ML surrogates for a *large-scale* system — 270MW with 180 turbines on an IEEE 118-bus network — by decomposing it into component-level surrogates: 30 GRU models (each covering 6 turbines, single hidden layer with just 30 neurons) for stateful DFIG dynamics, 20 MLPs (4 hidden layers, 64 neurons) for memoryless PV arrays, and separate GRUs (hidden=20) for battery storage. The key insight was that tiny, specialized models matched to each component's physics outperform monolithic ones and are deployable on FPGA. In HFSS, the analogous decomposition would be recurrent architectures for transient EM and feedforward surrogates for frequency-domain — or spatial decomposition via domain decomposition. The multi-time-step coupling I designed — ML-IBRs at 50µs EMT with the 118-bus network at 5ms TS timestep (100:1 ratio) — is directly analogous to multi-scale simulation coupling in EM. The FPGA deployment pipeline — quantization to fixed-point with LUT-based activations, latency profiling, error-bounded inference — transfers regardless of physics domain."

**Red flags:** Don't claim GRU directly works for Maxwell's equations. Don't say "deep learning is general purpose" without specifics. Don't let the interviewer frame your work as "small circuit" — it's a large-scale system with circuit-fidelity modeling.

**Q2: "How accurate are your models really? Walk me through the actual error numbers."**

> "Let me be precise — there are two distinct error metrics in the paper. The **model accuracy** (ML vs. ground-truth simulation) varies by component: DFIG GRU training loss converges to 0.02% MSELoss, battery GRU to 0.00078%, PV MLP to 0.2% under normal irradiance, and up to 4% under partial shading (the hardest scenario). The second metric is **quantization error** (float model vs. fixed-point FPGA model), which is below 0.01% — this confirms the deployment pipeline preserves accuracy. For out-of-distribution validation, the paper tests 80ms faults (20% shorter than training data), 200ms faults (100% longer), and wind speed step changes (8→13, 10→5 m/s). The models handle these well, with errors staying below 1% near transitions. However, Section V of the paper honestly acknowledges that models are optimized for inputs within rated range, and outlier data may cause inaccuracy — I'd suggest a conventional model monitor as a safety fallback."

**Pivot:** "For HFSS surrogates, silent wrong answers on new geometries would be catastrophic. I'd architect any ML surrogate with mandatory uncertainty bounds and solver fallback — the same monitoring philosophy I advocated in the JESTIE paper."

**Q3: "Why GRU over LSTM or Transformer?"**

> "Three concrete reasons driven by the FPGA deployment constraint. First, GRU has 2 gates (reset, update) vs. LSTM's 3 gates (input, forget, output) plus a cell state — that's fewer parameters and fewer matrix multiplications per timestep, which is critical when each GRU has only 30 hidden units and must run in 15µs on FPGA. Second, GRU only needs sigmoid and tanh activations, which I implemented as lookup tables (sigmoid range [-10,10], tanh range [-6,6], step 0.001) on FPGA — adding the extra cell state pathway of LSTM would increase LUT and DSP utilization beyond our 93.3% DSP budget. Third, Transformers require high-precision exponentials for softmax and O(n²) attention, which are impractical on fixed-point FPGA hardware. The small model size (hidden=30) means there's no benefit from the additional expressiveness of LSTM or Transformer — the bottleneck is data quality and physics structure, not model capacity."

**Q4: "What about error accumulation in autoregressive rollout?"**

> "This is a real challenge that the paper addresses honestly. The autoregressive mechanism works through recursive features — for DFIG, the model feeds back its own predictions of mechanical torque (Tm) and rotor speed (ωr) as inputs to the next timestep. For battery, it's SOC and internal current (Ii). These create temporal dependency chains where errors can compound. Three factors keep this manageable in practice. First, the physical systems have dissipative dynamics — energy losses naturally damp error growth, so the model learns stability attractors from training data. Second, the multi-time-step coupling (EMT at 50µs, TS at 5ms) means the slower TS solver periodically 'anchors' the system-level state. Third, the validation results on 10-second horizons show accumulated errors remain bounded. However — and I think intellectual honesty matters here — Section V of the paper explicitly acknowledges error accumulation as inherent to the approach and suggests pairing ML models with a conventional model monitor that can flag divergence. If I were extending this, I'd add uncertainty quantification (MC-dropout or ensemble disagreement) as a runtime trigger to switch to a physics solver."

**Q5: "Why not use PINNs instead of purely data-driven?"**

> "Three reasons for this specific application. First, power electronic converters have discontinuous switching functions (PWM switching, fault transients) — PINN losses with PDE residuals are poorly suited to discontinuities. Second, my goal was 15µs latency on FPGA; PINNs don't produce smaller or faster inference graphs than a direct surrogate. Third, generating training data was feasible — I used Monte Carlo sampling from MATLAB/Simulink for DFIG and battery models (wind speed drawn from N(10,3) m/s, with 5% fault scenarios), and C++ simulation for PV arrays. The data generation pipeline, while non-trivial, was automatable. However, for HFSS where a single 3D EM simulation can take hours, PINNs become much more attractive. I'd seriously explore embedding Maxwell's equations as soft constraints to reduce the number of expensive simulations needed for training."

---

### Paper 2: DAC (Chip Placement with Diffusion)

**Q1 (HARDEST): "4.4 minutes vs DREAMPlace at 0.48 minutes. 9x slower. Why?"**

> "Honest answer: for a single placement, a designer would choose DREAMPlace. Our advantage is in design space exploration — our generative model produces thousands of *diverse* placements by sampling, each structurally different. DREAMPlace converges to essentially one solution. On runtime: the 4.4 minutes is a research prototype with 1000 steps. Distillation to 10-50 steps, smaller networks, and batched GPU execution could bring this down 10-50x."

**Q2: "How does this scale to real circuits with millions of cells?"**

> "The ISPD2005 benchmarks range up to ~2.1M standard cells (bigblue4), but we preprocess them into 512 macro clusters using hMetis — so the model effectively handles 512 components. That's still 2-3 orders of magnitude smaller than cutting-edge designs with millions of macros. My scaling roadmap: (1) hierarchical decomposition — our clustering approach already does this at one level; extending to multi-level clustering would handle larger circuits, (2) graph sampling during training (GraphSAGE-style) to handle larger netlists, (3) mixed approach — our model for global macro/cluster placement, DREAMPlace for standard cell legalization, which is exactly what we already do for mixed-size results in Table 1."

**Q3: "Policy gradient training is notoriously unstable. How did you handle it?"**

> "Three specific problems and solutions. First, high variance in credit assignment across 1000 diffusion steps — solved with dense per-step energy feedback (our key innovation), computing the energy function at every step rather than only the final placement. This reduced gradient variance by 3.2× compared to sparse-reward baselines (Table 3 in the paper). Second, balancing wirelength vs. constraint satisfaction — solved by tuning the energy function weights (λ_overlap=2.0, λ_bound=1.0) through an empirical sweep, which we ablated in Figure 4. Third, maintaining exploration — solved with an entropy regularization term in the per-step reward (α_ent=0.001) and a noise penalty term that prevents the reverse process from deviating too far from the forward diffusion kernel. We used PPO with clipped objectives (ε=0.2) and GAE (λ=0.95, γ=0.99) for stable advantage estimation."

---

### Paper 3: EDISCO (Equivariant Diffusion for TSP)

**Q1 (HARDEST): "LKH-3 solves TSP-500 to 0.01% in seconds. Why should I care?"**

> "For standard TSP, you're right — LKH-3 is extraordinary. The value is threefold. First, amortized inference for millions of similar instances. Second, TSP is a testbed for a general methodology — the CTMC equivariant diffusion framework applies to problems where Concorde doesn't exist (routing with constraints, mesh optimization, topology optimization). Third, cross-distribution robustness: 4% degradation vs 133% for baselines means genuine geometric reasoning."

**Pivot:** "In EM simulation, mesh optimization is a combinatorial problem with geometric structure that lacks exact solvers. The equivariant diffusion framework could generate high-quality initial meshes."

**Q2: "How much comes from the neural network vs 2-opt post-processing?"**

> "On TSP-500: neural network alone gives ~1.2% gap. After 2-opt: 0.08%. Random tour + 2-opt gives ~3.5%. So the ML model provides a dramatically better starting point — 2-opt does local polishing, but the model determines which basin of attraction we land in. That 3.42 percentage point difference comes entirely from the neural network's global structure."

**Q3: "E(2) equivariance — is data augmentation sufficient?"**

> "I ran this ablation. On TSP-100 with abundant data, augmentation achieves comparable accuracy. But two critical differences at scale: (1) equivariance needs 33-50% less training data, and (2) cross-distribution transfer degrades only 4% vs 50%+ for augmented baselines. Equivariance forces the network to learn distance-based features rather than absolute-coordinate features."

---

### Cross-Cutting Questions

**"If you could pick ONE idea from your papers to apply to EM simulation?"**

> "The dense per-step feedback idea from my DAC paper — applied to making predictions on solver decisions during simulation. Concretely: train a model that, at each iteration of HFSS's adaptive mesh refinement, predicts where the mesh needs refinement and when to stop — dynamically refining its prediction as more information (error estimates, field gradients) becomes available from each solve. This directly matches the job description's focus, and my DAC work proves the approach works: computing a quality metric at every step of an iterative process, not just the end, gives dramatically better gradient signal for training. If there's room for a second idea, I'd explore E(3)-equivariant architectures that respect Maxwell's symmetries."

**"How long before you're productive on HFSS-related ML?"**

> "2-3 months, with two parallel tracks. Physics: running HFSS on canonical problems, studying solver behavior and data formats. Literature: ML for EM survey, Ansys's published work. The technical skills transfer quickly — building surrogates, generative design, equivariant architectures. The domain knowledge takes time, and I won't pretend otherwise."

---

## 5. Latest Ansys/Synopsys AI Developments (2025-2026)

### Show You're Current — Mention These

| Development | Date | Key Detail |
|---|---|---|
| **Ansys 2025 R1** | Early 2025 | Introduced **Electronics AI+** — predicts time/memory for HFSS, Maxwell, Icepak simulations. Trained on 1,500+ real projects. |
| **Ansys 2025 R2** | July 2025 | **Ansys Engineering Copilot** — AI assistant built into HFSS, AEDT, and other tools. Seven products now have AI+ built in. |
| **Synopsys acquisition** | July 17, 2025 | $35B. "Silicon to systems." First integrated capabilities expected H1 2026. |
| **NVIDIA partnership** | Dec 2025 | HFSS accelerated up to **50x on NVIDIA Blackwell**. Ansys Fluent: **500x speedup** with GPU + AI. |
| **Synopsys Converge** | March 11-12, 2026 | Will showcase **first integrated Synopsys-Ansys solutions**. |
| **India AI Summit** | Feb 2026 | Dr. Prith Banerjee (SVP Innovation) presented on "AI/ML in EDA and Engineering Simulation." |
| **SimAI 2025 R2** | July 2025 | "Predict as learned" post-processing, iterative model refinement ("build on top"), improved training diagnostics. |

### SimAI Technical Details Worth Knowing

- **Architecture:** Fusion of multiple deep learning neural networks (not a single architecture)
- **Input:** 3D geometry (the shape itself, unparameterized) + boundary conditions
- **Approach:** Implicit neural representations, physics-agnostic (learns from data)
- **Training:** Minimum ~30 simulations; training takes 48 hours to 5 days
- **Performance:** 10-100x speedup, comparable to full-fidelity simulation
- **Example:** Automotive aerodynamics: 50 hours on 500 CPUs → <1 hour on single GPU

### HFSS 2025 R2 Highlights
- **17x faster** radiation pattern calculations
- More precise phased array antenna beam steering
- Engineering Copilot integration

---

## 6. Your Value Proposition (Memorize This)

> "I bring three complementary capabilities that map directly to this role's focus on *making predictions on decisions during simulation* and *dynamically refining predictions*:
>
> 1. **ML predictions for simulation decisions** — My IEEE JESTIE work demonstrates I can analyze a complex physics system, decide which ML architecture fits each component's physics (GRUs for stateful dynamics, MLPs for memoryless mappings), and deploy tiny, specialized surrogates on FPGA achieving 3.33x faster-than-real-time with <0.01% quantization loss. I understand the full pipeline from training data generation to production hardware.
>
> 2. **Dynamic refinement as information arrives** — My diffusion work is literally about refining predictions iteratively. In my DAC paper, I compute an energy function at *every* diffusion step — not just the end — so the model dynamically improves its prediction as more information becomes available. This maps directly to simulation solvers that refine solutions iteratively, like adaptive meshing in HFSS.
>
> 3. **Unsupervised learning from physics-based metrics** — My DAC paper trains models using only a quality metric, without optimal examples. This is critical when labeled optimal simulation data is expensive — you can train from the physics residual or error metric directly."

---

## 7. Questions to Ask Xin Xu (Pick 3-4)

**About the role (echoes JD language — shows you read it carefully):**
1. "The job description mentions *making predictions on decisions during simulation*. Could you give me an example of what kind of decisions the team is targeting — mesh refinement, solver selection, convergence estimation?"
2. "What does the internship project scope typically look like — a self-contained research project, or contributing to a larger initiative?"

**About technical direction:**
3. "Ansys has both physics-informed and physics-agnostic approaches. Which direction is the team focusing on, and why?"
4. "What are the main challenges when applying ML to EM simulation specifically? I imagine high-frequency fields have different characteristics than CFD."

**About integration (shows you've done homework):**
5. "With Synopsys Converge coming up in March, are there new opportunities combining Ansys simulation with Synopsys EDA tools? I'm curious because my DAC paper is on chip placement."

**About growth:**
6. "What skills or knowledge do successful interns typically develop during the summer?"

---

## 8. Key Numbers — Quick Reference Card

### Your Work
| Metric | Value | Paper |
|--------|-------|-------|
| Wind farm FTRT | **3.33x** (50µs timestep / 15µs latency) | JESTIE |
| System equivalent timestep | **1.5ms** (100 × 15µs wind farm latency) | JESTIE |
| DFIG model accuracy | **0.02% MSELoss** (training loss) | JESTIE |
| Battery model accuracy | **0.00078% MSELoss** | JESTIE |
| PV model accuracy | **0.2%** normal, **4%** partial shading | JESTIE |
| Quantization error | **<0.01%** (float vs FPGA fixed-point) | JESTIE |
| Scalability | **8193x speedup over traditional** at 10,000 turbines | JESTIE |
| System scale | 180 turbines (270MW) + 320 PV + 50MW battery, IEEE 118-bus | JESTIE |
| FPGA utilization | 93.3% DSP, 24.42% FF, 61.3% LUT (Xilinx VCU118) | JESTIE |
| GRU architecture | hidden=30 (DFIG), hidden=20 (battery), single layer | JESTIE |
| Multi-time-step ratio | 100:1 (50µs EMT : 5ms TS) | JESTIE |
| HPWL improvement | **3.7-5.6%** over supervised | DAC |
| Gradient variance reduction | **3.2x** | DAC |
| TSP-500 gap | **0.08%** (SOTA) | EDISCO |
| Training data reduction | **33-50% less** | EDISCO |
| Cross-distribution degradation | **4%** (vs 133% DIFUSCO, 687% T2T) | EDISCO |

### Ansys/Synopsys
| Metric | Value |
|--------|-------|
| SimAI speedup | 10-100x |
| Acquisition price | $35 billion |
| Electronics AI+ training data | 1,500+ real projects |
| HFSS 2025 R2 radiation calc speedup | 17x |
| NVIDIA Blackwell HFSS acceleration | up to 50x |
| Synopsys Converge | March 11-12, 2026 |

---

## 9. Behavioral Questions — STAR Stories

> **Framework:** Every answer follows **S**ituation → **T**ask → **A**ction → **R**esult. Keep each story under 90 seconds. End with what you learned or how it connects to the role.

---

### Story 1: Overcoming a Major Technical Challenge

**Use for:** "Tell me about a time you solved a difficult problem" / "Describe a technical challenge you faced"

> **Situation:** In my IEEE JESTIE work, I had trained GRU models in PyTorch that achieved 0.02% MSELoss in floating-point. But the entire project's value depended on deploying these models on FPGA for real-time simulation — and FPGA only supports fixed-point arithmetic.
>
> **Task:** Convert the trained models to fixed-point without destroying accuracy, and fit everything within the FPGA's resource budget (DSP blocks, LUTs, flip-flops) while maintaining 15µs latency per timestep.
>
> **Action:** I tackled this in three stages. First, I designed LUT-based activation functions — precomputing sigmoid over [-10, 10] and tanh over [-6, 6] at step size 0.001, stored in on-chip memory. This eliminated the need for floating-point exponentials entirely. Second, I developed a systematic quantization pipeline: dynamic quantization for GRU weights (where range varies across gates) and static quantization for MLP weights. I profiled each layer's numerical range to choose optimal fixed-point formats. Third, when the initial deployment hit 97% DSP utilization — dangerously close to routing failure — I restructured the computation pipeline to share multipliers across sequential operations rather than instantiating them in parallel.
>
> **Result:** Final quantization error was below 0.01% — essentially lossless. DSP utilization came down to 93.3%, and the wind farm model achieved 15µs latency (3.33× faster-than-real-time). The FPGA pipeline became the core contribution of the paper.
>
> **Takeaway:** "I learned that deployment constraints should inform architecture decisions from the start, not be an afterthought. At Synopsys, I'd apply the same principle — designing ML models with the simulation tool's latency and integration requirements in mind from day one."

---

### Story 2: Pivoting After a Failed Approach

**Use for:** "Tell me about a time something didn't work" / "Describe a failure and what you learned"

> **Situation:** For my DAC chip placement paper, I initially followed the standard approach from the literature — training a diffusion model with supervised learning on optimal placements generated by DREAMPlace.
>
> **Task:** Build a generative model for chip placement that could produce diverse, high-quality placements and generalize across different circuit designs.
>
> **Action:** After two months of work, I had a supervised model that could reproduce placements it had seen but failed to generalize to new circuits. The fundamental issue was clear: optimal placements are expensive to generate (each takes minutes with DREAMPlace), and the model was memorizing layout-specific patterns rather than learning general placement principles. I stepped back and asked: what if we don't need optimal labels at all? The placement quality function (HPWL + overlap penalty) is differentiable and cheap to evaluate. So I redesigned the entire training framework around policy gradients — treating each denoising step as an RL action and computing the energy function at every step, not just the final placement. This was a significant pivot — I essentially threw away two months of supervised training code and rebuilt from scratch.
>
> **Result:** The unsupervised approach outperformed the supervised one by 3.7–5.6% in HPWL. More importantly, it achieved zero-shot transfer to ISPD2005 and ICCAD2004 benchmarks without any retraining. The dense per-step feedback reduced gradient variance by 3.2× compared to sparse-reward baselines.
>
> **Takeaway:** "The failed approach taught me that when labeled data is scarce or expensive, you should look for computable quality metrics instead. This is directly relevant to simulation — you rarely have 'optimal' simulation outputs, but you always have physics-based error metrics you can train against."

---

### Story 3: Managing Competing Priorities

**Use for:** "How do you manage your time?" / "Tell me about balancing multiple responsibilities"

> **Situation:** I'm simultaneously pursuing a PhD in ECE at the University of Alberta and a Master of Music in Viola Performance at Western University. During the Fall 2025 semester, I had a paper deadline for ICML (EDISCO), a DAC submission, coursework, and orchestra performances — all with overlapping timelines.
>
> **Task:** Deliver high-quality research outputs for both paper submissions while maintaining my academic and musical commitments.
>
> **Action:** I developed a structured approach. First, I identified the critical path for each paper — for EDISCO, the bottleneck was the large-scale TSP-10000 experiments (each training run took 3 days on a single GPU), so I started those first and worked on other tasks while they ran. For DAC, the bottleneck was the RL training stability, so I scheduled those experiments to run overnight with automated logging. I used a weekly planning system where I allocated dedicated blocks: mornings for research coding, afternoons for paper writing and analysis, evenings for music practice. I also learned to say no to non-essential commitments during crunch periods.
>
> **Result:** Both papers were submitted on time. EDISCO achieved state-of-the-art on all TSP benchmarks, and the DAC paper's unsupervised approach was a novel contribution. I also performed in three orchestra concerts that semester without missing any deadlines.
>
> **Takeaway:** "The dual-degree experience taught me to be ruthlessly systematic about prioritization and to identify bottlenecks early so I can parallelize work. In an internship, I'd apply the same approach — understanding the project's critical path and structuring my time to maximize throughput."

---

### Story 4: Learning a New Domain Quickly

**Use for:** "How do you approach learning something unfamiliar?" / "Tell me about a time you had to ramp up quickly"

> **Situation:** My first paper (JESTIE) was in power systems and FPGA deployment — no generative modeling at all. When I started the DAC project, I needed to learn diffusion models from scratch, including the mathematical framework (SDEs, score matching, denoising score matching), implementation in PyTorch, and the chip placement domain (EDA, HPWL, legalization).
>
> **Task:** Go from zero knowledge of diffusion models to a novel research contribution within one semester.
>
> **Action:** I followed a structured learning path. First, I read the foundational papers in order: DDPM (Ho et al.), Score-Based SDE (Song et al.), then application papers like DIFUSCO. I implemented DDPM from scratch on MNIST before touching the placement problem — this gave me hands-on intuition for the noise schedule, loss function, and sampling process. Then I studied the EDA domain: read DREAMPlace's code, understood HPWL computation, learned about placement legalization. Only after building both foundations did I start combining them. I kept a research log where I documented every experiment, failed idea, and insight — this prevented me from repeating mistakes and helped me identify the key insight (dense per-step feedback) faster.
>
> **Result:** Within four months, I had not just implemented an existing method but developed a novel training framework that outperformed prior work. The research log habit was crucial — I could trace back exactly which experiment led to the per-step energy idea.
>
> **Takeaway:** "I'd take the same structured approach to ramping up on HFSS and EM simulation — start with fundamentals (run HFSS on canonical problems, understand the solver pipeline), then study the team's existing ML work, then identify where my skills can contribute. I'd expect to be productive within 2-3 months."

---

### Story 5: Collaboration and Working with Others

**Use for:** "Tell me about a teamwork experience" / "How do you work with others?"

> **Situation:** The JESTIE paper had five co-authors, each responsible for different components of a large-scale simulation system — 180 wind turbines, 320 PV arrays, and battery storage on an IEEE 118-bus network. I was responsible for the ML modeling and FPGA deployment, while other team members handled the power system network solver, validation framework, and co-simulation interface.
>
> **Task:** Integrate my ML-based component models with the team's existing network solver so that the ML outputs (voltage, current predictions) could be consumed at each 50µs timestep without introducing numerical instability.
>
> **Action:** The main challenge was the interface between my FPGA-deployed ML models and the network solver running on CPU. We needed to agree on data formats, timing protocols, and error handling. I organized weekly sync meetings where each team member presented their component's I/O specification. When we discovered that my GRU models' recursive features (feeding back predicted torque and rotor speed) could cause instability if the network solver updated boundary conditions at a different rate, I proposed the multi-time-step coupling scheme: ML-IBRs at 50µs EMT with the network solver at 5ms TS (100:1 ratio). This required careful coordination — I wrote a detailed interface specification document and we iterated on it over three weeks until both sides agreed.
>
> **Result:** The integrated system ran stably for all test scenarios, including 80ms and 200ms fault events. The multi-time-step coupling became a key contribution of the paper. Two co-authors later told me the interface specification document I wrote saved them significant debugging time.
>
> **Takeaway:** "I learned that clear documentation and regular communication are as important as the technical work itself. In an internship at Synopsys, I'd prioritize understanding the team's existing codebase and interfaces before jumping into implementation."

---

### Story 6: Intellectual Honesty / Presenting Limitations

**Use for:** "Tell me about a time you had to give honest feedback" / "How do you handle uncertainty?"

> **Situation:** During the review of my JESTIE paper, I faced a decision about how to present the PV model's accuracy. Under normal irradiance, the MLP achieved 0.2% error — excellent. But under partial shading conditions (when some panels are shadowed), error jumped to 4% — significantly worse.
>
> **Task:** Decide whether to downplay the partial shading results or address them head-on.
>
> **Action:** I chose to address the limitation explicitly in the paper. In Section V, I added a dedicated discussion of the accuracy degradation under partial shading, explained the technical reason (the nonlinear bypass diode activation creates sharp discontinuities in the I-V curve that MLPs struggle with), and proposed mitigation strategies (piecewise models, physics-informed loss terms). I also added a general recommendation that ML surrogates should always be paired with a conventional model monitor as a safety fallback for out-of-distribution inputs.
>
> **Result:** The reviewers specifically praised the honest discussion of limitations. One reviewer wrote that the failure analysis and proposed mitigations "significantly strengthened the paper's contribution." The paper was accepted without major revision requests on the accuracy front.
>
> **Takeaway:** "In simulation tools that engineers depend on, silent wrong answers are dangerous. At Synopsys, I'd always architect ML components with uncertainty quantification and solver fallback — knowing when the model doesn't know is as important as the prediction itself."

---

### Quick-Reference: Mapping Stories to Common Questions

| Behavioral Question | Primary Story | Backup Story |
|---|---|---|
| "Biggest technical challenge?" | Story 1 (FPGA deployment) | Story 2 (DAC pivot) |
| "Tell me about a failure" | Story 2 (supervised → unsupervised pivot) | Story 6 (PV accuracy) |
| "How do you manage time?" | Story 3 (PhD + Music dual degree) | Story 4 (learning diffusion) |
| "How do you learn new things?" | Story 4 (zero to diffusion expert) | Story 1 (FPGA from scratch) |
| "Teamwork experience?" | Story 5 (JESTIE integration) | Story 3 (multi-project coordination) |
| "Dealing with ambiguity?" | Story 2 (pivoting approach) | Story 4 (new domain) |
| "Handling disagreements?" | Story 5 (interface negotiation) | Story 6 (honest limitations) |
| "What are you proud of?" | Story 1 (FPGA deployment) | Story 2 (novel unsupervised framework) |
| "How do you handle pressure?" | Story 3 (dual degree + deadlines) | Story 5 (integration deadline) |
| "Integrity / ethics?" | Story 6 (honest limitations) | Story 5 (clear documentation) |

---

## 10. Files in This Folder

| File | Contents | When to Read |
|------|----------|-------------|
| `synopsys_final_prep.md` | **THIS FILE** — Master document | Day 1-4 |
| `synopsys_technical_interview_prep.md` | Original technical prep (detailed Q&A) | Day 1 |
| `synopsys_ml_system_design_scenarios.md` | 4 system design scenarios with full worked answers | Day 2 |
| `pytorch_coding_review.py` | Python fundamentals + tensor ops + practical PyTorch patterns + 10 gotchas | Day 2 |
| `pytorch_coding_prep.md` | Study guide companion for the .py file — quick references, verbal answers, concepts | Day 2 |
| `synopsys_interview_prep.md` | Recruiter screen prep (completed) | Reference |
| `synopsys_conversation_summary.md` | Full conversation details | Reference |
| `Learning Mesh-Based Simulation with GNN.pdf` | MeshGraphNets paper (Pfaff et al., ICML 2021) — most relevant external paper | Day 1 |

---

## 11. Final Reminders

**Do:**
- Lead with IEEE JESTIE — it's your closest match to the role
- Be honest about limitations — a 15-year veteran will detect overselling instantly
- Show you've studied HFSS — mention adaptive meshing, S-parameters, Delta-S convergence
- Connect your diffusion work to iterative solvers — "diffusion denoising IS an iterative solver"
- Frame ML as an accelerator for physics, not a replacement

**Don't:**
- Claim your GRU directly works for Maxwell's equations
- Claim your method is better than classical solvers (LKH-3, Concorde, DREAMPlace)
- Bluff about EM knowledge you don't have
- Over-engineer answers — simple, direct, technically precise
- Forget to ask questions at the end

**The single most important insight to convey:**
> "The diffusion reverse process is a learned iterative solver. It starts from noise and progressively refines — exactly like HFSS starting from a coarse mesh and iteratively improving. My research on making this process efficient, physics-aware, and equivariant transfers directly to simulation."

---

*Good luck on Monday, Ruogu.*
