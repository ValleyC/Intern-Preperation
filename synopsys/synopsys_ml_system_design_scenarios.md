# ML System Design Scenarios for Synopsys/Ansys Technical Interview

**Prepared for:** Ruogu Chen
**Interview Date:** Monday, February 24, 2026
**Interviewer:** Xin Xu, Principal R&D Engineer (HFSS team)
**Focus:** "How would you design an ML system for X" questions in EM simulation

---

## How to Use This Document

Each scenario follows a consistent structure:
1. **The Question** -- exactly how the interviewer might phrase it
2. **Framework** -- a step-by-step mental scaffold (30 seconds of organizing before answering)
3. **Detailed Answer** -- what to actually say, in natural technical language
4. **Key Concepts** -- specific terms and ideas that signal domain expertise
5. **Follow-up Questions** -- likely probes and how to handle them

**General Strategy:** Always start by clarifying the problem, then walk through data, model, training, validation, and deployment. Tie back to your own research at natural junctures -- do not force it.

---
---

# SCENARIO 1: ML-Accelerated Adaptive Mesh Refinement for HFSS

---

## 1.1 The Question

> "HFSS uses adaptive mesh refinement -- we start with a coarse tetrahedral mesh, solve the full FEM system, compute error indicators, refine the mesh where errors are high, and repeat until S-parameters converge. This loop is the single biggest computational bottleneck. Each pass requires a full electromagnetic solve. How would you design an ML system to speed this up?"

---

## 1.2 Framework for Answering (Say This First)

> "Great question -- this is an iterative decision-making problem, which connects to my diffusion model research. Let me break this down into: (1) understanding the current workflow and where the bottleneck is, (2) what data we can collect, (3) what exactly to predict, (4) architecture choices, (5) training strategy, (6) validation, and (7) how it integrates back into HFSS."

---

## 1.3 Detailed Answer

### Step 1: Understand the Current Workflow

"First, let me make sure I understand the HFSS adaptive meshing loop:

1. Generate an initial tetrahedral mesh from the 3D geometry
2. Solve Maxwell's equations via FEM on that mesh -- this produces E-field and H-field solutions
3. Compute a posteriori error indicators per tetrahedron (comparing the field discontinuity across element faces)
4. Grade each tetrahedron -- the top ~30% highest-error elements are flagged for refinement
5. Refine those elements (split tetrahedra, smooth locally)
6. Re-solve the full FEM system on the new mesh
7. Check convergence: has Delta-S (the change in S-parameters between consecutive passes) dropped below the threshold?
8. If not, repeat from step 3

The bottleneck is clear: each iteration requires a full FEM solve, which means assembling and solving a large sparse linear system. Typically this takes 3-10 passes to converge, so we are paying for 3-10 full solves."

### Step 2: What Data to Collect

"I would instrument existing HFSS runs to capture the full history of each adaptive refinement:

**Per-pass data:**
- The tetrahedral mesh at each pass (node positions, element connectivity, material labels)
- The computed field solution (E-field, H-field values at nodes or integration points)
- Per-element error indicators (the a posteriori error estimate)
- Which elements were selected for refinement (the ground truth decisions)
- S-parameter values at each pass and the Delta-S convergence metric

**Per-problem metadata:**
- Geometry description (CAD features, material properties, port locations)
- Frequency of interest (or frequency sweep parameters)
- Boundary conditions (PEC, radiation, PML)
- Final converged mesh and S-parameters (our ground truth for validation)

**Why this matters:** We need pairs of (current mesh state, correct refinement action) to train a supervised model, plus the full trajectory to train a sequential decision model."

### Step 3: What to Predict -- Three Levels of Ambition

"There are three levels of what the ML model could predict, each with different risk/reward:

**Level 1 (Conservative -- Error Indicator Prediction):**
Predict the per-element error indicator directly from the mesh geometry and a coarse field estimate, without running a full FEM solve. This means instead of solving Ax = b (the expensive part) to get fields, then computing errors from fields, we skip the solve and predict errors directly.

- Input: mesh graph (tetrahedra, connectivity, geometry features) + boundary conditions + frequency
- Output: per-element scalar error indicator
- Benefit: skip the FEM solve in intermediate passes, only run the full solve on the final mesh for validation

**Level 2 (Medium -- Refinement Decision Prediction):**
Predict which elements to refine directly, bypassing both the solve and the error computation. This is a binary classification per element: refine or not.

- Input: same as Level 1
- Output: per-element binary label (refine / don't refine)
- Benefit: even faster since we skip error estimation entirely

**Level 3 (Ambitious -- Direct Mesh Prediction):**
Given the initial geometry, predict something close to the final converged mesh in one shot, or predict the mesh after k passes in one step (skipping intermediate passes).

- Input: geometry + simulation setup
- Output: target element sizes or mesh density field over the domain
- Benefit: could reduce 8 passes to 1-2 passes

I would start with Level 1 because it is the most straightforward to validate -- we can compare predicted error indicators to actual ones -- and if it works, Level 2 follows naturally."

### Step 4: Architecture Choice

"The mesh is inherently a graph: tetrahedra are nodes (or we use the dual graph where faces are edges), and adjacency defines connectivity. This screams Graph Neural Network.

**My recommended architecture: a GNN operating on the mesh graph.**

Specifically:
- **Node features:** For each tetrahedron -- volume, aspect ratio, material properties, distance to ports/boundaries, distance to geometric features (edges, curves), local geometry curvature, and if available, a coarse field estimate (even from a single-pass solve on the initial mesh)
- **Edge features:** Shared face area between adjacent tetrahedra, face normal direction, material interface indicator
- **Global features:** Frequency, problem type, number of ports, bounding box size (to provide scale context)

The GNN performs message passing: each tetrahedron aggregates information from its neighbors, building up a multi-scale understanding of the mesh. After L message-passing layers (I would start with L=8-12 based on my EDISCO experience), each node has a representation that captures both local geometry and broader spatial context.

The final per-node MLP head outputs the predicted error indicator (for Level 1) or refinement probability (for Level 2).

**Why GNN over CNN on voxelized fields?**
- Voxelization loses geometric fidelity at material boundaries and curved surfaces -- exactly where mesh refinement matters most
- GNNs naturally handle variable-resolution meshes (the mesh changes every pass)
- The graph structure is already there -- no information is lost in representation
- GNNs scale better: a voxelized volume at fine resolution is O(N^3), while the mesh graph is O(N_elements)

**Why not a neural operator (FNO)?**
- FNO works on regular grids, not unstructured tetrahedral meshes
- We are not trying to learn the field solution -- we are trying to learn the refinement decision
- However, a neural operator could complement this as a fast field predictor (I will come back to this)

**Hybrid option:** Use a cheap neural operator to predict an approximate field on the mesh, then feed those approximate field values as additional node features into the GNN for error prediction. This gives the GNN physics information without running the full FEM solve."

### Step 5: Training Strategy

"**Data generation:** Run HFSS on a diverse library of geometries -- antennas, waveguides, PCB structures, IC packages -- and log the full adaptive refinement trajectory. This gives us thousands of (mesh, error indicator) pairs per design, across multiple passes.

**Loss function for Level 1 (error prediction):**
```
L = MSE(predicted_error, true_error) + lambda * ranking_loss
```
The ranking loss is important: we do not need the absolute error values to be perfect -- we need the relative ranking to be correct, because HFSS selects the top 30% highest-error elements. So I would add a pairwise ranking loss (like the hinge loss used in learning-to-rank) that penalizes misorderings. This is analogous to how in my DAC paper, we care about relative energy improvement at each step, not absolute values.

**Loss function for Level 2 (refinement decision):**
```
L = BCE(predicted_refine_prob, true_refine_label) + lambda * focal_loss_term
```
With focal loss to handle class imbalance (only ~30% of elements are refined in each pass).

**Training curriculum:**
1. Train first on late passes (passes 5-8) where the mesh is already reasonable and the refinement decisions are more fine-grained
2. Then fine-tune on early passes where the mesh is coarser and refinement patterns are more dramatic
3. This curriculum avoids the model learning only 'refine everything' from early-pass data

**Augmentation:**
- Random rotations/reflections of the geometry (the physics is rotation-invariant -- connecting to my EDISCO work on E(2) equivariance)
- Frequency perturbation (train at nearby frequencies to improve generalization)
- Scale normalization (normalize mesh coordinates to unit bounding box)"

### Step 6: Validation Strategy

"This is critical because we cannot deploy an ML model that produces an incorrect mesh and silently gives wrong S-parameters.

**Level 1 validation (error indicator quality):**
- Compare predicted error indicators to actual ones: correlation, ranking accuracy (Kendall's tau)
- Check that the predicted top-30% overlaps significantly with the actual top-30% (precision/recall)

**End-to-end validation (what matters):**
- Run the ML-guided adaptive refinement loop and compare final S-parameters to the fully converged HFSS result
- Metric: |S_ML - S_HFSS| across the frequency band -- must be within acceptable tolerance (e.g., 0.01 dB for |S11|)
- Also compare: number of passes needed, total element count of final mesh, total wall-clock time

**Safety mechanism:**
- Always run one full FEM solve on the final mesh as a verification step
- If the ML model's predicted errors disagree significantly with the actual errors on that final solve, fall back to standard adaptive refinement
- This means worst case, we add one extra solve; best case, we save 5-7 solves"

### Step 7: Integration with HFSS

"The deployment path I envision:

**Phase 1 (Drop-in replacement for error estimation):**
- After the first full FEM solve (which we always run), feed the mesh + field solution into the GNN
- GNN predicts error indicators for subsequent passes
- Run 2-3 refinement passes guided by GNN predictions without full solves
- Run a final FEM solve to validate

**Phase 2 (Warm-start mesh prediction):**
- Before any FEM solve, use the GNN to predict a good initial mesh from the geometry alone
- This replaces the default initial meshing heuristics
- Then run 1-2 adaptive passes with full solves for convergence verification

**Expected speedup:**
If the standard workflow is 6 passes and each pass takes T minutes:
- Phase 1 saves ~4 full solves, reducing 6T to ~2T plus GNN inference (negligible)
- Phase 2 could further reduce to 1-2 passes total

This is conservative but trustworthy. The key principle: ML accelerates, physics validates."

---

## 1.4 Key Concepts to Mention

| Concept | Why It Impresses an HFSS Expert |
|---------|-------------------------------|
| **A posteriori error estimation** | Shows you understand how HFSS decides where to refine |
| **Delta-S convergence criterion** | Shows you know the stopping condition (not just generic "convergence") |
| **Tetrahedra grading** | Shows familiarity with the actual HFSS workflow |
| **Graph dual representation** | Shows you have thought about how to represent a mesh as a GNN input |
| **Ranking loss** | Shows you understand that ordinal correctness matters more than absolute accuracy for refinement decisions |
| **Safety mechanism / physics validation** | Shows respect for simulation accuracy -- you are not replacing the solver, you are guiding it |
| **E(2) equivariance for meshes** | Connect to your EDISCO work -- mesh refinement should not depend on orientation |
| **Curriculum training on pass number** | Shows understanding that early vs. late passes have different characteristics |

---

## 1.5 Potential Follow-up Questions

### "How would you handle broadband adaptive meshing?"

> "In broadband mode, HFSS adapts the mesh at multiple frequencies simultaneously, because a mesh that is good at one frequency may be too coarse at another. I would extend the GNN model to accept frequency as a conditioning input -- not just a scalar feature, but potentially through Fourier feature encoding (mapping frequency to a high-dimensional sinusoidal representation). The model would then predict error indicators conditioned on frequency, and the refinement aggregates across frequencies. This is similar to how neural operators handle parameterized PDEs -- the frequency is essentially a parameter of the underlying Maxwell's equations."

### "The mesh changes every pass -- how do you handle that in the GNN?"

> "This is actually a strength of the GNN approach. Unlike a CNN that requires a fixed grid, a GNN operates on whatever graph it receives. Each pass produces a new mesh graph -- the GNN simply takes the new graph as input. For temporal awareness across passes, I have two options: (1) process each pass independently (stateless), or (2) use a recurrent approach where a GRU aggregates information across passes -- similar to my IEEE JESTIE work where I used GRUs to capture temporal dynamics. The recurrent approach could learn patterns like 'this region was refined last pass but still has high error, so it needs more aggressive refinement.'"

### "What about physics-informed approaches? Could you embed Maxwell's equations?"

> "Yes, I could add a physics-informed regularization term. After predicting the error indicators, I could compute a cheap approximation of the field residual -- how well the predicted error field satisfies Maxwell's equations in a weak sense. This would be:
> ```
> L_total = L_data + alpha * L_physics
> ```
> However, I would be cautious. The full FEM solve already enforces Maxwell's equations exactly. The ML model's job is to predict where the discretization error is highest, not to solve Maxwell's equations itself. So I would start purely data-driven and add physics regularization only if generalization to new geometries is poor."

### "How much training data would you need?"

> "Based on my experience with GNNs in EDISCO (5.5M parameters, trained on ~1M TSP instances), I would estimate needing on the order of 500-1000 complete HFSS adaptive refinement trajectories across diverse geometries. Each trajectory has 5-8 passes, giving us 2500-8000 (mesh, error) pairs. The key is diversity: we need antennas, waveguides, connectors, PCB traces, IC packages -- the model must generalize across problem types. I would start with a smaller pilot (100 trajectories) to verify the approach works, then scale up."

### "What if the GNN makes a bad prediction and the mesh is wrong?"

> "This is why the safety mechanism is essential. The worst outcome is not a slow simulation -- it is a wrong answer that the user trusts. My design always includes a final FEM verification solve. If the ML-predicted errors and the actual errors from that final solve disagree by more than a threshold, the system falls back to standard adaptive refinement and flags the case for retraining. Over time, these failure cases become the most valuable training data. This is essentially active learning -- similar in spirit to how my DAC paper uses per-step energy feedback to identify where the diffusion model's predictions need improvement."

---
---

# SCENARIO 2: Surrogate Model for Antenna Design Exploration

---

## 2.1 The Question

> "Antenna engineers use HFSS to explore design spaces -- they vary patch width, feed position, substrate thickness, ground plane size, and so on, running hundreds of simulations to understand how S-parameters and radiation patterns change. Each simulation takes 10-60 minutes. How would you build a surrogate model that lets them explore the design space interactively?"

---

## 2.2 Framework for Answering

> "This is a parametric surrogate modeling problem -- very similar to what I did in my IEEE JESTIE paper, but for a different physics domain. Let me walk through: (1) the data pipeline, (2) input/output representations, (3) architecture, (4) training, (5) uncertainty quantification, and (6) active learning to improve the surrogate over time."

---

## 2.3 Detailed Answer

### Step 1: Data Pipeline

"The foundation is a well-designed parametric sweep in HFSS.

**What to vary (input parameters):**
For a patch antenna as a concrete example:
- Patch width W and length L (2 continuous parameters)
- Feed position (x_feed, y_feed) (2 continuous)
- Substrate thickness h and dielectric constant epsilon_r (2 continuous)
- Ground plane dimensions (2 continuous)
- Possibly discrete parameters: number of stacked patches, feed type (probe vs. microstrip)

That gives us ~8-10 design parameters. The design space is a hypercube in R^8-10.

**What to store (outputs) -- multiple levels:**
- **Level 1 (minimal):** S-parameters (S11, S21, ...) as complex-valued functions of frequency -- typically 200-1000 frequency points spanning the band of interest. This is what engineers look at first.
- **Level 2 (richer):** Radiation patterns -- far-field gain as a function of (theta, phi) at key frequencies. This is a 2D function on the sphere per frequency.
- **Level 3 (complete):** Full 3D field distributions (E, H) on the computational domain. This is the most expensive to store but most informative.

I would definitely store Level 1 and Level 2. Level 3 is optional -- it enables the surrogate to predict fields, but storage and training cost are higher.

**Data generation strategy:**
- Use Latin Hypercube Sampling (LHS) to get good space-filling coverage of the parameter space
- Start with ~200-500 designs for an 8-parameter space
- Run HFSS on each design (parallelizable across a compute cluster)
- Total cost: 200 designs x 30 min average = ~100 compute-hours = achievable
- Store all results in a structured format: parameter vector -> (S-parameters, radiation pattern, mesh info)"

### Step 2: Input Representation

"For parametric surrogates, the input is simply the design parameter vector x in R^d. But there are important design choices:

**Option A: Raw parameter vector**
- Input: [W, L, x_feed, y_feed, h, epsilon_r, W_gnd, L_gnd]
- Simple, direct, works well when the parameterization is known
- This is what I would start with

**Option B: Geometry encoding**
- Convert parameters to a 3D geometry representation (voxels, point cloud, or mesh)
- Feed through a 3D encoder (PointNet, 3D CNN, or GNN)
- More general -- works for design spaces that cannot be easily parameterized
- This is closer to what SimAI does

**I would start with Option A** because antenna engineers already think in terms of design parameters, and it is simpler. Option B becomes important when the geometry is too complex for simple parameterization (e.g., free-form optimization).

**Frequency as input:** Critically, frequency must also be an input. We want the model to predict S11(f; x) -- S-parameters as a function of both frequency and design parameters. This means frequency is concatenated with the design vector:
```
input = [W, L, x_feed, y_feed, h, epsilon_r, W_gnd, L_gnd, f]
```
This enables the model to predict S-parameters at any frequency, not just the training frequencies."

### Step 3: Output Representation

"**For S-parameters:**
S-parameters are complex-valued. I would represent them as (magnitude_dB, phase_degrees) or (real, imaginary) at each frequency. The model predicts both components.

Two approaches:
- **Point-wise:** Input includes frequency f, output is S11(f) -- a single complex number per query. The model is queried at many frequencies to build the curve. This is simple and flexible.
- **Curve-wise:** Input is just the design parameters, output is the entire S-parameter curve discretized at N frequency points -- a vector in R^(2N). This captures the shape of the curve holistically.

I would use the curve-wise approach because the shape of the S-parameter curve matters (location of resonance dips, bandwidth, out-of-band behavior), and an MLP predicting the full curve can learn these structural features.

**For radiation patterns:**
The radiation pattern is a function on the sphere. I would discretize it on a regular (theta, phi) grid -- say 37 x 73 points (5-degree resolution) -- giving a 2D image of size 37 x 73 per frequency. This can be treated as an image prediction problem.

**Multi-task output:**
The model predicts both S-parameters and radiation patterns simultaneously, sharing a common encoder. This is multi-task learning -- the shared representation learns geometry features useful for both tasks."

### Step 4: Architecture

"**My recommended architecture: Encoder-Decoder with skip connections.**

```
Design Parameters [R^d]
        |
  [MLP Encoder: d -> 256 -> 512 -> 512]
        |
  Latent Representation z [R^512]
        |
   +---------+---------+
   |                   |
[S-param Decoder]   [Pattern Decoder]
   |                   |
[MLP: 512->256->2N]  [MLP: 512->256->...]
   |                   |
S11(f) curve        Gain(theta,phi)
```

**Why an MLP and not something fancier?**
For parametric surrogates with ~8-10 input dimensions and smooth parameter-to-output mappings, MLPs work remarkably well. My IEEE JESTIE paper showed this: MLPs for PV arrays (4 layers, 64 hidden) achieved 0.2% error under normal conditions. The mapping from antenna parameters to S-parameters is smooth and continuous (small geometry changes cause small S-parameter changes), so an MLP can approximate it efficiently.

**When to go fancier:**
- **Neural operator (DeepONet or FNO):** If we want to predict the full 3D field distribution, not just S-parameters. DeepONet learns a mapping from input functions (geometry + boundary conditions) to output functions (field). This is overkill for S-parameter prediction but powerful for field prediction.
- **Implicit Neural Representation (INR):** If we want to predict fields at arbitrary spatial locations. The INR takes (x, y, z, design_params) and outputs E(x,y,z). This is what SimAI uses. It naturally handles different mesh resolutions.
- **Autoencoder + MLP:** If the output is high-dimensional (full 3D field), first train an autoencoder to compress HFSS field solutions to a low-dimensional latent code, then train an MLP from design parameters to latent code, and decode. This two-stage approach handles high-dimensional outputs efficiently.

For the antenna design exploration use case, I would use the MLP for S-parameters and the Autoencoder+MLP for radiation patterns."

### Step 5: Training

"**Loss function:**
```
L = w1 * L_Sparam + w2 * L_pattern + w3 * L_physics
```

Where:
- `L_Sparam = MSE(predicted_S11_curve, true_S11_curve)` in dB scale, with extra weight near resonance frequencies (the dips in S11 matter most)
- `L_pattern = MSE(predicted_gain, true_gain)` on the radiation pattern grid
- `L_physics = |predicted_S11_at_resonance - min(predicted_S11)|` -- a soft constraint encouraging the model to correctly identify the resonance frequency

**Weighted loss near resonance:**
Antenna engineers care most about the resonance frequency (where S11 dips below -10 dB) and the bandwidth. I would add higher weight to frequency points near and below -10 dB. This focuses the model's capacity on the operating region.

**Training details:**
- 80/10/10 train/validation/test split, stratified to ensure coverage of the parameter space
- Adam optimizer, learning rate 1e-3 with cosine decay
- Batch size 64-128
- Train for ~500-1000 epochs (small dataset, so training is fast)
- Early stopping on validation loss

**This is very similar to my JESTIE workflow:** I trained GRUs and MLPs on simulation data with MSE loss and validated on held-out scenarios. The key lesson from that work: the training data must cover the operating range, including edge cases."

### Step 6: Uncertainty Quantification

"This is arguably the most important part for a production surrogate. Engineers must know when to trust the model and when to fall back to HFSS.

**Approach 1: Ensemble**
Train 5-10 models with different random seeds. For a new design, run all models and compute:
- Mean prediction (the surrogate's best estimate)
- Standard deviation (the uncertainty)

If std > threshold, flag the design as 'low confidence -- recommend full HFSS simulation.'

**Approach 2: MC Dropout**
Add dropout layers in the MLP, keep dropout active at inference. Run N forward passes, compute mean and variance. Cheaper than an ensemble.

**Approach 3: Distance-aware uncertainty**
Compute the distance from the query point to the nearest training points in parameter space. Designs far from any training data naturally have higher uncertainty. This is simple but effective as a first-pass filter.

**I would use Approach 1 (ensemble)** because it is the most reliable, and the inference cost is negligible (an MLP forward pass takes microseconds).

**In the UI:** Show the predicted S-parameter curve with a shaded confidence band. If the band is wide, suggest the user run HFSS. If narrow, the surrogate is trustworthy."

### Step 7: Active Learning

"The initial training set from LHS is uniform -- it wastes samples in regions where the mapping is smooth and under-samples regions where it is complex (near resonance transitions, mode coupling).

**Active learning loop:**
1. Train the surrogate on current data
2. Generate candidate designs (random or grid in parameter space)
3. For each candidate, compute the ensemble disagreement (uncertainty)
4. Select the top-K most uncertain designs
5. Run HFSS on those K designs
6. Add results to training set
7. Retrain the surrogate
8. Repeat

**Acquisition function:** I would use the Expected Improvement over the current model's uncertainty, or simply the maximum variance from the ensemble.

**Budget allocation:**
- Start with 200 LHS designs (initial batch)
- 5 active learning rounds of 50 designs each
- Total: 450 HFSS runs -- still manageable
- The active learning rounds target the hard regions, so the model improves where it matters most

**Connection to my DAC paper:** This is conceptually similar to the dense per-step feedback idea. In my DAC work, I evaluate the energy at every diffusion step to guide training. Here, I evaluate the model's uncertainty at every design point to guide data collection. Both are about allocating computational budget where it provides the most information."

---

## 2.4 Key Concepts to Mention

| Concept | Why It Impresses |
|---------|-----------------|
| **S-parameter curves, not just scalars** | Shows you understand the output is frequency-dependent |
| **Resonance frequency weighting** | Shows you know what antenna engineers actually care about |
| **Latin Hypercube Sampling** | Standard DOE technique -- shows you know experimental design |
| **Ensemble uncertainty** | Critical for trustworthy surrogates -- this is what makes it production-ready |
| **Active learning / adaptive sampling** | Goes beyond "just train on data" -- shows ML maturity |
| **Multi-task (S-params + patterns)** | Shows you think holistically about the antenna design problem |
| **SimAI connection (INR)** | Shows you know the company's existing approach |
| **MLP effectiveness for smooth mappings** | Grounded in your own published results (JESTIE) |

---

## 2.5 Potential Follow-up Questions

### "How do you handle discrete parameters, like feed type?"

> "For discrete parameters (e.g., probe feed vs. microstrip feed), I would use a categorical embedding -- a small learned vector for each discrete option -- concatenated with the continuous parameters before the encoder. This is standard in tabular deep learning. If the discrete parameter fundamentally changes the physics (like switching from probe to slot feed), I would train separate models or add a mixture-of-experts head."

### "What about geometric changes that are not easily parameterized?"

> "For non-parametric geometry changes -- like free-form shape optimization or topology changes -- I would switch to a geometry encoder. Options include: (1) PointNet on a point-cloud representation of the surface, (2) a 3D CNN on a voxelized occupancy grid, or (3) a GNN on the surface mesh. The geometry encoder replaces the parameter vector as input to the surrogate. This is closer to SimAI's approach. My EDISCO work gives me experience with GNNs on geometric data, so I am comfortable with this direction."

### "How do you compare this to a Gaussian Process surrogate?"

> "GPs are the classical choice for surrogate modeling and they give excellent uncertainty estimates by construction. They work well for low-dimensional parameter spaces (d < 10-15) with small datasets (< 1000 points). Their main limitation is scalability: standard GP inference is O(N^3) in the number of training points, and they struggle with high-dimensional outputs (like full S-parameter curves with 500 frequency points). For our problem -- moderate-dimensional input, high-dimensional output, potentially thousands of training points -- neural networks are more scalable. But I would actually benchmark a GP as a baseline, because if it works, it provides natural uncertainty quantification without the ensemble overhead."

### "How would you deploy this in the Ansys product?"

> "I envision two deployment modes. First, an 'explore' mode: the surrogate runs in real-time as the engineer drags sliders for design parameters, and the S-parameter plot updates instantly. This enables interactive design exploration -- something HFSS cannot do at 30 minutes per simulation. Second, an 'optimize' mode: run gradient-based optimization through the surrogate (since it is a neural network, it is differentiable) to find the optimal design parameters, then validate the optimum with a single HFSS simulation. The surrogate lives as a lightweight model file alongside the HFSS project, and is retrained whenever the user adds new HFSS results."

### "What about interpolation vs. extrapolation?"

> "This is the core risk. Surrogates interpolate well but extrapolate poorly. I would address this in three ways: (1) the uncertainty quantification flags extrapolation -- ensemble disagreement is high outside the training range, (2) the active learning specifically targets the boundary of the explored region, and (3) I would clearly communicate the valid parameter ranges in the UI. An antenna engineer who tries a substrate thickness of 10mm when all training data used 0.5-3mm should see a clear warning."

---
---

# SCENARIO 3: Predicting Resource Requirements for HFSS Simulations

---

## 3.1 The Question

> "Before running an HFSS simulation, engineers want to know: how long will this take, and how much memory will it need? We announced AI+ capability in 2025 R1 that predicts time and memory for SIwave AC SYZ simulations, trained on 1,500+ real projects. How would you design such a system for HFSS?"

---

## 3.2 Framework for Answering

> "This is fundamentally a regression problem with structured tabular input. The challenge is not the model -- it is the feature engineering and the diversity of problem types. Let me walk through: (1) what features predict runtime, (2) model choice, (3) training data, (4) handling heterogeneity, (5) calibration, and (6) product integration."

---

## 3.3 Detailed Answer

### Step 1: Feature Engineering -- What Predicts Runtime?

"This is the most critical step. The runtime and memory of an HFSS simulation depend on:

**Mesh complexity features:**
- Number of tetrahedra in the initial mesh (strongest predictor)
- Estimated number of tetrahedra after adaptive refinement (can be estimated from geometry complexity)
- Number of mesh elements per wavelength (a function of frequency and geometry size)
- Total model volume relative to wavelength cubed (electrical size)
- Number of material regions and material interfaces

**Geometry features:**
- Number of 3D objects in the model
- Number of faces, edges, vertices in the CAD model (geometric complexity)
- Presence of thin layers or small features (these force fine meshes)
- Aspect ratio statistics (max/min feature size ratio -- high aspect ratios are expensive)
- Bounding box dimensions relative to wavelength

**Simulation setup features:**
- Solution type: Driven Modal, Driven Terminal, Eigenmode
- Frequency: single point, sweep (interpolating vs. discrete), broadband
- Number of frequency points
- Number of ports
- Number of modes per port
- Adaptive mesh settings: maximum number of passes, convergence threshold (Delta-S)
- Maximum mesh refinement percentage per pass

**Boundary condition features:**
- Type: PEC, radiation (absorbing), PML, periodic
- Number of radiation boundaries (open problems are more expensive)
- Whether the problem has symmetry planes (reduces effective problem size)

**Hardware features:**
- Number of CPU cores available
- Available RAM
- GPU availability (for GPU-accelerated solvers)
- Solver type: direct vs. iterative

I would compute ~30-50 features from these categories. Some features are directly available from the HFSS project file (before running); others require a quick initial mesh generation (which is fast -- seconds, not minutes)."

### Step 2: Model Choice

"For tabular data with ~50 features and ~1500+ training samples, I would strongly consider:

**Primary recommendation: Gradient Boosted Trees (XGBoost or LightGBM)**

Why:
- Gradient boosted trees are consistently the best-performing model for tabular regression with moderate-sized datasets
- They handle heterogeneous features naturally (continuous, discrete, categorical)
- They are robust to feature scaling and outliers
- They provide feature importance rankings (interpretability -- engineers want to know WHY a simulation is slow)
- They train in seconds, inference in microseconds
- They handle non-linear interactions without manual feature engineering

**Alternative: Neural network (MLP)**
- Advantage: can learn more complex interactions, especially if features are correlated
- Disadvantage: needs more data, less interpretable, requires more tuning
- I would use this as a secondary model in an ensemble

**My recommendation: Ensemble of XGBoost + LightGBM + MLP**
- Average the predictions (or learn a stacking weight)
- Use the disagreement between models as an uncertainty indicator
- This is the approach that wins Kaggle competitions on tabular data, and for good reason

**Why not a deep neural network?**
With only 1,500 training samples, a deep network would overfit. Tree-based methods have a strong inductive bias for tabular data that neural networks lack. The 2022 paper 'Why do tree-based models still outperform deep learning on tabular data?' (Grinsztajn et al.) supports this. However, as the dataset grows to 10,000+ samples, neural networks become competitive.

**Log-transform the target:**
Simulation time and memory span orders of magnitude (30 seconds to 30 hours). I would predict log(time) and log(memory) rather than raw values. This makes the regression problem more well-conditioned and ensures the model is not dominated by a few extremely large jobs."

### Step 3: Training Data Collection

"**Data sources:**
1. **Telemetry from existing HFSS installations:** With user consent, collect simulation metadata (features) and actual runtime/memory from completed simulations. The 2025 R1 announcement mentions 1,500+ projects -- this is likely how they collected it.
2. **Internal benchmark suite:** Ansys maintains standard test cases for regression testing. These provide controlled data points.
3. **Synthetic data augmentation:** For under-represented problem types, create parameterized variants (scale geometry, change frequency, modify mesh settings) and run them.

**Data cleaning is critical:**
- Remove simulations that failed or were cancelled
- Remove simulations run on unusual hardware (one-off workstations)
- Normalize runtime to a reference hardware configuration (e.g., predict time on a 'standard' machine, then scale by hardware specs)
- Handle simulations that hit memory limits (these are censored data -- the true runtime is unknown)

**Handling censored data:**
When a simulation runs out of memory, we know the memory requirement exceeds the available RAM, but we do not know the actual requirement. This is analogous to survival analysis. I would use a Tobit model or modify the loss function:
```
L = MSE(pred, actual)  for completed simulations
  + max(0, available_RAM - pred)^2  for OOM simulations  (penalize under-predictions)
```

**Feature extraction pipeline:**
Build a lightweight HFSS plugin that, given a project file:
1. Parses the geometry and simulation setup (fast, no simulation needed)
2. Generates the initial mesh (fast, ~seconds)
3. Computes the ~50 features described above
4. Feeds them to the prediction model"

### Step 4: Handling Diversity of Problem Types

"HFSS is used for wildly different problems: 5G antenna arrays, automotive radar, IC packages, EMI shielding, waveguide filters. The physics and computational profiles are very different.

**Approach 1: Problem-type conditioning**
Add a categorical feature indicating the problem type (antenna, IC package, PCB, waveguide, etc.). The tree model will naturally learn different rules for different types.

**Approach 2: Mixture of experts**
Train separate models for each problem type. At inference, first classify the problem type, then use the appropriate expert model. This is more accurate but requires enough data per type.

**Approach 3: Hierarchical model**
Train a global model on all data, then fine-tune on each problem type. The global model captures general patterns (more elements = more time), and the fine-tuned models capture type-specific patterns.

I would start with Approach 1 (simplest) and move to Approach 2 if prediction accuracy differs significantly across problem types.

**Handling hardware diversity:**
The same design takes 10 minutes on a 128-core workstation and 2 hours on a laptop. I would:
- Include hardware features (cores, RAM, GPU) as input features
- Normalize to a reference configuration for training
- At inference, adjust the prediction based on the user's hardware
- The simplest scaling: `time_user = time_reference * (cores_reference / cores_user)^alpha`, where alpha < 1 accounts for imperfect parallelization"

### Step 5: Calibration and Confidence Intervals

"Engineers do not just want a point estimate -- they want to know 'will this finish by 5 PM?'

**Quantile regression:**
Instead of predicting the mean runtime, predict the 10th, 50th, and 90th percentiles:
```
'Expected time: 45 minutes (likely between 25 minutes and 1.5 hours)'
```
This is done by training with pinball loss at each quantile level.

**For XGBoost:** Use quantile regression objective or the NGBoost framework for probabilistic predictions.

**Calibration check:**
On the test set, verify that:
- The 50th percentile prediction is close to the median actual value
- 80% of actual values fall between the 10th and 90th percentile predictions
If calibration is off, apply isotonic regression or Platt scaling as a post-hoc correction.

**Asymmetric cost:**
Under-predicting runtime is worse than over-predicting (the engineer blocks out too little time vs. too much). I would use an asymmetric loss function:
```
L = alpha * max(0, actual - pred)^2 + beta * max(0, pred - actual)^2
```
where alpha > beta, penalizing under-predictions more heavily."

### Step 6: Product Integration

"**In the HFSS UI:**
Before hitting 'Analyze', the user sees:
```
Estimated time: ~45 minutes (30 min - 1.2 hours)
Estimated peak memory: ~12 GB (10 - 16 GB)
Recommended: 16 GB RAM, 8 CPU cores
```

If the predicted memory exceeds available RAM:
```
Warning: This simulation may require more memory than available.
Suggestion: Enable HPC distributed solve, or simplify the model.
```

**Progressive refinement of predictions:**
After each adaptive mesh pass (which completes in minutes), update the prediction:
- Pass 1 complete: actual mesh has 150K elements (initial estimate was 120K)
- Revised estimate: 55 minutes (increased from 45)

This connects to the JD's 'dynamically refine predictions as more information becomes available.' The model starts with pre-simulation features, then incorporates actual mesh size after pass 1, actual solve time for pass 1, and so on to refine the estimate.

**Feedback loop:**
When the simulation completes, record the actual time and memory. Compare to prediction. If the error is > 30%, flag for review. Over time, the model is retrained on accumulated data, continuously improving.

**Connection to my research:** This progressive refinement is conceptually similar to my DAC diffusion work -- each step provides new information that updates the prediction. The initial estimate is like the noisy starting point, and each completed pass is a denoising step that brings us closer to the true runtime."

---

## 3.4 Key Concepts to Mention

| Concept | Why It Impresses |
|---------|-----------------|
| **Electrical size (model size / wavelength)** | The single most important predictor -- shows EM domain knowledge |
| **Log-transform of targets** | Shows you understand the distribution of simulation times |
| **Censored data (OOM simulations)** | Shows you think about real-world data issues |
| **Progressive refinement after each pass** | Directly maps to the JD requirement |
| **Quantile regression** | Shows you care about uncertainty, not just point estimates |
| **Asymmetric loss** | Shows you understand the cost of wrong predictions from the user's perspective |
| **Tree-based models for tabular data** | Demonstrates awareness of when NOT to use deep learning |
| **Feature importance / interpretability** | Engineers want to understand predictions, not just trust them |

---

## 3.5 Potential Follow-up Questions

### "How would you estimate mesh size before running the simulation?"

> "The initial mesh size can be estimated from geometry complexity: number of CAD faces, volume, feature size relative to wavelength. For the final mesh size (after adaptive refinement), I would build a separate sub-model: predict the mesh growth factor (final mesh size / initial mesh size) based on features like the convergence threshold, maximum refinement percentage, frequency, and geometry complexity. In my experience, this growth factor is relatively predictable -- typically 2-5x for well-designed models, but can be 10x+ for models with thin layers or resonant structures."

### "How do you handle new problem types not in the training data?"

> "This is the extrapolation problem. My mitigation strategies: (1) the uncertainty estimate (ensemble disagreement) will be high for new problem types, so the system flags the prediction as 'low confidence', (2) after the simulation completes, the actual values are recorded and used for online model updates, (3) I would build a 'novelty detector' -- a simple model that checks whether the input features are within the convex hull of the training data. If not, show a cautionary message: 'Limited data for this problem type -- estimate may be less reliable.'"

### "Is 1,500 training samples enough?"

> "For tree-based models with ~50 features, 1,500 samples is a reasonable starting point -- enough to capture the main trends. However, accuracy will improve with more data. I would establish baseline performance metrics at 1,500 (e.g., mean absolute percentage error of predictions) and set targets for improvement. With telemetry collection from deployed users, the dataset will grow naturally over time. The 2025 R1 announcement starting at 1,500 is a smart MVP approach -- ship early, improve with user data."

### "What about multi-physics simulations?"

> "If the HFSS simulation is coupled with thermal or structural analysis (which is common for high-power applications), the resource prediction becomes harder because there are cross-physics dependencies. I would add features describing the coupling: type of coupling (one-way vs. two-way), number of coupled physics, convergence criteria for the coupling loop. This is an area where a neural network might outperform tree models, because the interactions between physics are more complex. But I would start by predicting resources for the EM portion only, which is the dominant cost."

---
---

# SCENARIO 4: Applying Diffusion Model Expertise to EM Field Prediction

---

## 4.1 The Question

> "I see from your papers that you have deep experience with diffusion models -- both for chip placement and combinatorial optimization. How would you apply that expertise to electromagnetic field prediction? If I gave you a 3D geometry and boundary conditions, could a diffusion model predict the EM field distribution?"

---

## 4.2 Framework for Answering

> "This is a great question and it directly connects my research to this domain. Let me think about: (1) what the diffusion process looks like for EM fields, (2) how to condition on geometry and physics, (3) architecture options, (4) why diffusion might be better than direct prediction, (5) the connection to iterative solvers, and (6) how this relates to my published work."

---

## 4.3 Detailed Answer

### Step 1: What Does the Diffusion Process Look Like for EM Fields?

"In standard diffusion models (like DDPM), we add Gaussian noise to clean data over T steps, then learn to reverse the process -- starting from pure noise and gradually denoising to a clean sample.

For EM fields, the 'clean data' is the converged field solution E(x,y,z) and H(x,y,z) from HFSS. The diffusion process would be:

**Forward process:** Gradually add noise to the true field solution
```
E_t = sqrt(alpha_t) * E_0 + sqrt(1 - alpha_t) * noise
```
where E_0 is the true EM field and E_t is the noisy version at timestep t.

**Reverse process:** Starting from E_T ~ N(0, I) (pure noise), learn to denoise step by step:
```
E_{t-1} = f_theta(E_t, t, geometry, frequency, BCs)
```

The model f_theta takes the current noisy field, the timestep, and the problem specification (geometry, frequency, boundary conditions) and predicts the denoised field at the previous timestep.

**The key insight is that the reverse diffusion process is an iterative refinement** -- each step takes a noisy/approximate field and makes it slightly more accurate. This maps beautifully to iterative numerical solvers: a linear solver like GMRES also starts from an initial guess and iteratively refines it toward the solution."

### Step 2: Conditioning on Geometry and Physics

"The diffusion model must know what geometry it is solving for. Conditioning mechanisms:

**Geometry conditioning:**
- Encode the 3D geometry as a signed distance function (SDF) on a regular grid, or as a binary occupancy grid. The SDF naturally represents boundaries, material interfaces, and curved surfaces.
- For material properties, create additional channels: epsilon_r(x,y,z), mu_r(x,y,z), sigma(x,y,z) as 3D fields co-located with the noise/field grid.
- Concatenate the geometry channels with the noisy field at each denoising step.

**Frequency conditioning:**
- Use Fourier feature encoding of the frequency: map f to [sin(2*pi*f*b_1), cos(2*pi*f*b_1), ..., sin(2*pi*f*b_K), cos(2*pi*f*b_K)] with logarithmically spaced b_k
- Inject this into the model via FiLM conditioning (Feature-wise Linear Modulation) -- modulating intermediate feature maps
- This allows a single model to handle a range of frequencies

**Boundary condition conditioning:**
- Encode boundary conditions (PEC, PML, radiation, port excitation) as additional 3D masks/channels
- Port excitation can be represented as the incident field pattern at the port boundary

**Physics conditioning (advanced):**
- At each denoising step, compute the Maxwell's equation residual of the current field estimate
- Feed this residual as an additional input channel
- This gives the model physics-aware feedback at every step -- directly analogous to the per-step energy feedback in my DAC paper"

### Step 3: Architecture

"**Option A: 3D U-Net on voxelized field (My primary recommendation)**

The standard diffusion architecture for spatial data. The U-Net processes the 3D field on a regular grid:

- Input: [E_t (3 channels for Ex, Ey, Ez), H_t (3 channels), geometry (SDF + material channels), timestep embedding]
- Architecture: 3D U-Net with residual blocks, attention at lower resolutions
- Output: predicted noise epsilon or predicted clean field E_0
- Resolution: start with 64^3 or 128^3 voxels

Advantages:
- Well-studied architecture with known training recipes
- 3D convolutions naturally capture spatial correlations
- Multi-scale U-Net captures both near-field detail and far-field structure

Disadvantages:
- Fixed resolution (voxel grid does not adapt like a mesh)
- Computational cost scales as O(N^3) with resolution
- May waste computation in empty regions

**Option B: GNN on the mesh**

If we want to operate directly on the tetrahedral mesh (avoiding voxelization):
- Nodes = tetrahedra or mesh vertices
- Message-passing GNN as the denoising network
- Naturally handles adaptive resolution (fine mesh near boundaries, coarse elsewhere)

Advantages:
- Respects the mesh structure
- No resolution loss from voxelization
- More memory efficient for problems with localized features

Disadvantages:
- More complex to implement
- Diffusion on graphs is less well-studied than on grids
- Variable-size graphs complicate batching

**Option C: Neural operator backbone**

Use a Fourier Neural Operator (FNO) or DeepONet as the denoising backbone:
- The spectral nature of FNO aligns well with EM fields (which have natural spectral representations)
- FNO layers learn in frequency space -- this is physically motivated for Maxwell's equations

**My recommendation: 3D U-Net for the initial prototype, with physics-informed residual feedback.** The U-Net is battle-tested for diffusion, and the voxelized representation is straightforward. Once proven, explore GNN on mesh for production deployment."

### Step 4: Training Strategy

"**Training data:** Existing HFSS solutions. Each training sample is:
- Input: (geometry, frequency, BCs) -- the problem specification
- Target: (E(x,y,z), H(x,y,z)) -- the converged field solution, voxelized onto the regular grid

**Data augmentation:**
- Random rotations and reflections (EM fields transform equivariantly under rotations -- E' = R*E for rotation R). This is directly connected to my EDISCO work on E(2) equivariance. In fact, I would explore making the U-Net equivariant using steerable convolutions. EM fields are vector fields, and their transformation under rotation is well-defined.
- Frequency shifting (within the training range)
- Scaling (normalize geometry to unit cube)

**Loss function options:**

**Option 1: Standard diffusion loss (denoising score matching)**
```
L = E[||epsilon - epsilon_theta(E_t, t, cond)||^2]
```
This trains the model to predict the noise that was added. Simple and effective.

**Option 2: Hybrid diffusion + physics loss**
```
L = L_diffusion + lambda * L_Maxwell
```
where L_Maxwell is the Maxwell's equation residual evaluated on the predicted clean field:
```
L_Maxwell = ||curl(E_predicted) + j*omega*mu*H_predicted||^2
          + ||curl(H_predicted) - j*omega*epsilon*E_predicted||^2
```
This is a PINN-style loss applied at each diffusion step.

I would start with Option 1 and add the physics loss as a regularizer if the model struggles to capture fine electromagnetic features (e.g., evanescent fields near conductors).

**Connection to DAC paper:**
In my DAC paper, I compute the energy E(x_t) = E_wire + lambda*E_overlap at every diffusion step t and use it as a per-step reward for policy gradient training. The Maxwell residual here plays the same role -- it is a physics-based quality metric evaluated at each denoising step. I could even train the diffusion model using my policy gradient framework:
- Treat each denoising step as an action
- Reward = negative Maxwell residual at that step
- Use PPO with GAE, exactly as in my DAC paper
This would enable unsupervised training (no ground-truth field solutions needed), using only the Maxwell equations as supervision. This is a novel contribution that directly transfers my methodology."

### Step 5: Why Diffusion Instead of Direct Prediction?

"An important question -- why not just train a feedforward model (like SimAI's approach) to directly predict E(x,y,z) from (geometry, frequency, BCs)?

**Advantages of diffusion over direct prediction:**

1. **Better quality for complex fields.** EM field distributions can have sharp features: standing waves, resonant modes, evanescent fields near conductors, surface currents. A feedforward model tends to produce blurry predictions (it minimizes MSE, which favors the mean). Diffusion models are known to produce sharper, more detailed outputs -- this is why they dominate image generation.

2. **Built-in refinement.** If you have a coarse initial estimate (e.g., from a cheap 1-pass FEM solve), you can start the diffusion process from that estimate instead of pure noise. This is called guided diffusion or warm-starting. You skip the early denoising steps and only run the fine refinement steps. This could accelerate inference by 5-10x.

3. **Variable compute budget.** With diffusion, you choose how many steps to run. For a quick exploration, run 10 steps for a rough field estimate. For a final answer, run 100 steps for maximum accuracy. Feedforward models always cost the same.

4. **Uncertainty through multiple samples.** Run the diffusion process multiple times with different noise seeds. The variance across samples indicates field regions where the model is uncertain -- natural uncertainty quantification without ensembles.

5. **Physics feedback at each step.** As I described, I can compute Maxwell residuals at each diffusion step and use them to guide the denoising. This is not possible with a feedforward model.

**When direct prediction is better:**
- If speed is the overriding concern (single forward pass vs. 50-100 steps)
- If the fields are smooth and a blurry estimate is acceptable
- If you need to differentiate through the model (for design optimization)

**My recommendation:** Use direct prediction (MLP/FNO) for S-parameter surrogate modeling (Scenario 2) where speed matters and outputs are lower-dimensional. Use diffusion for full 3D field prediction where quality and detail matter."

### Step 6: Connection to Iterative Solvers

"Here is the deepest insight, and it directly connects to the job description's 'dynamically refine predictions as more information becomes available':

**HFSS's iterative solver and diffusion are doing the same thing conceptually.**

In HFSS:
```
x_0 = initial guess
x_{k+1} = x_k + M^{-1} * (b - A*x_k)  [iterative solver step]
converge when ||b - A*x_k|| < epsilon
```

In diffusion:
```
E_T = noise
E_{t-1} = E_t + f_theta(E_t, t, cond)  [denoising step]
converge after T steps (or early stopping)
```

Both start from a poor initial state and iteratively improve it. The difference is that the solver uses the exact physics (matrix A) while diffusion uses a learned denoising function (f_theta).

**Hybrid approach -- the most promising direction:**
1. Run 1-2 passes of the actual FEM solver to get a coarse field solution
2. Use that coarse solution as the starting point for diffusion (not pure noise)
3. Run 10-20 diffusion denoising steps to refine the field
4. Optionally, run 1 more FEM pass to validate

This combines physics-based accuracy (the FEM passes ensure Maxwell's equations are approximately satisfied) with learned refinement (the diffusion model captures patterns the solver would find in later iterations).

**Connection to my DAC paper:** In DAC, my diffusion model refines chip placement by computing physical energy at each step. Here, the diffusion model refines the EM field by (implicitly or explicitly) reducing the Maxwell equation residual at each step. The per-step physics feedback is the common thread."

### Step 7: Connection to Candidate's Specific Research

"Let me explicitly map each element to my published work:

| Diffusion for EM | My Research | Connection |
|-----------------|-------------|------------|
| Per-step Maxwell residual feedback | DAC: per-step energy feedback | Same framework -- physics evaluation at every diffusion step |
| Policy gradient training on Maxwell residual | DAC: PPO with per-step rewards | Direct transfer of training methodology |
| E(2) equivariant denoising network | EDISCO: E(2)-equivariant GNN | EM fields transform predictably under rotation |
| Continuous-time diffusion formulation | EDISCO: CTMC-based diffusion | Better ODE solvers, adaptive step sizes |
| Warm-starting from coarse solution | DAC: noise schedule conditioning | Starting from informative prior instead of pure noise |
| GRU/MLP for fast initial estimate | JESTIE: GRU/MLP surrogates | Use trained surrogate to provide warm start for diffusion |

The pipeline I envision:
1. Use a fast GRU/MLP model (a la JESTIE) to produce a coarse field estimate in milliseconds
2. Use the diffusion model (a la DAC/EDISCO) to iteratively refine that estimate
3. Each diffusion step is guided by Maxwell residual feedback (a la DAC energy feedback)
4. The diffusion model uses an equivariant architecture (a la EDISCO)
5. Final output is a high-fidelity field prediction in seconds instead of minutes"

---

## 4.4 Key Concepts to Mention

| Concept | Why It Impresses |
|---------|-----------------|
| **Maxwell residual as per-step feedback** | Novel transfer of your DAC methodology to physics simulation |
| **Warm-starting from coarse FEM solution** | Shows you understand how to combine ML with existing solvers |
| **Equivariant architecture for vector fields** | EM fields are vector fields that transform under rotation -- directly from EDISCO |
| **Diffusion = learned iterative solver** | The deepest insight -- shows you see the conceptual unity |
| **Continuous-time formulation** | Adaptive step sizes for variable-accuracy predictions |
| **Sharp vs. blurry predictions** | Addresses the mode-collapse weakness of feedforward surrogates |
| **Variable compute budget** | Practical advantage for different use cases |
| **Policy gradient on Maxwell residual** | Unsupervised training without ground-truth fields |

---

## 4.5 Potential Follow-up Questions

### "How would you handle the massive memory cost of 3D diffusion?"

> "This is a real concern. A 128^3 voxel grid with 6 field components (E and H) is 12 million values per sample. The U-Net with multi-scale processing is memory-intensive. My mitigation strategies: (1) Start at 64^3 resolution and use super-resolution as a second stage, (2) use latent diffusion -- train an autoencoder to compress the 3D field to a low-dimensional latent code, then run diffusion in latent space. This is exactly what Stable Diffusion does for images and what I could adapt for 3D fields, (3) use patch-based diffusion -- process overlapping 3D patches independently and stitch them together, (4) for the GNN-on-mesh approach, memory scales linearly with mesh elements rather than cubically with resolution."

### "How would you ensure the predicted fields satisfy Maxwell's equations?"

> "Three levels of enforcement: (1) Soft enforcement -- add the Maxwell residual to the loss function as a regularizer (PINN-style). This encourages but does not guarantee compliance. (2) Hard enforcement -- project the predicted field onto the space of divergence-free fields after each denoising step. For example, enforce div(E) = rho/epsilon by computing the divergence, solving a Poisson equation for the correction, and subtracting the correction. This is cheap and guarantees Gauss's law. (3) Post-processing -- run a single FEM relaxation step on the predicted field, using it as the initial guess. The solver will correct any physics violations quickly because the initial guess is already close."

### "What about the physics-aligned field reconstruction with diffusion bridge from ICLR 2025?"

> "That paper addresses a related but different problem -- reconstructing fields from sparse measurements using a diffusion bridge between the measurement-consistent subspace and the full field. For our application (forward prediction from geometry), the setup is different: we have complete geometry information but want to predict the field without solving. However, the diffusion bridge idea is relevant for a specific use case: if we have a partial field solution (e.g., from a coarse mesh solve) and want to 'fill in' the fine-scale details. The bridge would connect the coarse solution distribution to the fine solution distribution, which is exactly the warm-starting idea I described. I would study that paper carefully for the mathematical framework."

### "How long would inference take compared to HFSS?"

> "For a 3D U-Net on a 128^3 grid with 50 diffusion steps: each step involves a single U-Net forward pass, which on a GPU takes roughly 50-100 ms. So 50 steps would take 2.5-5 seconds. Compare this to a typical HFSS solve of 10-60 minutes. Even with 100 steps, we are talking 10 seconds vs. 30 minutes -- a 200x speedup. With warm-starting from a coarse solution, we might need only 10-20 steps for the refinement, bringing it under 2 seconds. The latent diffusion approach would be even faster because the U-Net operates on a smaller representation."

### "Could this replace HFSS entirely?"

> "No, and it should not try to. The value proposition is for design exploration: quickly predict fields for hundreds of design variants, identify the most promising ones, then validate the top candidates with full HFSS. Think of it as a very sophisticated initial filter. For certification-grade results (meeting regulatory standards, final sign-off), you still need the full physics solve. The diffusion model sits between back-of-envelope calculations and full simulation -- it fills the gap where engineers currently have no good option."

---
---

# General Interview Tips for System Design Questions

## The META-Framework (Use for Any "How Would You Design..." Question)

1. **Clarify** (15 seconds): Restate the problem, ask one clarifying question
2. **Scope** (15 seconds): State what level of fidelity you are targeting (research prototype vs. production)
3. **Data** (2 minutes): What data exists, what would you collect, how much do you need
4. **Model** (3 minutes): Architecture choice with justification, alternatives considered
5. **Training** (2 minutes): Loss function, strategy, curriculum
6. **Validation** (1 minute): How do you know it works, comparison to baseline
7. **Integration** (1 minute): How it fits into the existing workflow
8. **Connection to your work** (1 minute): Natural tie-back to your papers

## Phrases That Show Domain Respect

- "I would not want to replace the solver -- I want to guide it"
- "The physics is the ground truth; the ML model is an accelerator"
- "We need a safety mechanism -- silent errors are worse than slow simulations"
- "I would validate against full HFSS, not against another ML model"
- "The engineer should always have the option to fall back to the full simulation"

## Phrases That Show ML Maturity

- "The model architecture matters less than the data pipeline and validation"
- "I would start with the simplest model that could work (MLP/XGBoost) and add complexity only if needed"
- "Uncertainty quantification is not optional for this application"
- "Active learning lets us be smart about which simulations to run"
- "Log-transform for targets that span orders of magnitude"

## Connecting to Your Research (Natural Tie-backs)

| When They Ask About... | Connect To... |
|------------------------|---------------|
| Iterative refinement | DAC: dense per-step energy feedback in diffusion |
| Surrogate modeling | JESTIE: GRU/MLP achieving 30x speedup, 0.02% NRMSE (DFIG), <0.01% quantization error |
| Geometry-aware learning | EDISCO: E(2)-equivariant GNN |
| Training without labels | DAC: unsupervised policy gradient on energy function |
| Continuous-time processes | EDISCO: CTMC formulation, compatible with ODE solvers |
| Real-time prediction | JESTIE: FPGA deployment at 15 microsecond latency |
| Handling variable-size inputs | EDISCO/DAC: GNN handling different graph sizes |

---

*Prepared February 20, 2026. Good luck with the interview on Monday!*
