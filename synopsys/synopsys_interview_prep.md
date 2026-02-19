# Synopsys ML Internship Recruiter Screen Preparation

**Position:** Machine Learning Internship (Summer 2026)  
**Date:** Thursday, February 19, 2026, 4 PM MST  
**Location:** Canada (Hybrid/Remote)  
**Salary:** $38–43 CAD/hour

---

## Table of Contents

1. [Role Overview](#1-role-overview)
2. [Your Research Summary](#2-your-research-summary)
   - [Paper 1: IEEE JESTIE — Hardware-in-the-Loop ML Simulation](#paper-1-ieee-jestie--hardware-in-the-loop-ml-simulation)
   - [Paper 2: DAC 2026 — Chip Placement with Policy Gradient Diffusion](#paper-2-dac-2026--chip-placement-with-policy-gradient-diffusion)
   - [Paper 3: ICML 2026 — EDISCO: Equivariant Diffusion for TSP](#paper-3-icml-2026--edisco-equivariant-diffusion-for-tsp)
3. [Mapping Your Experience to the Job](#3-mapping-your-experience-to-the-job)
4. [Elevator Pitch](#4-elevator-pitch)
5. [Likely Questions & Answers](#5-likely-questions--answers)
   - [Background Questions](#background-questions)
   - [Technical Questions](#technical-questions)
   - [Motivation Questions](#motivation-questions)
6. [Technical Concepts to Review](#6-technical-concepts-to-review)
7. [Questions to Ask Them](#7-questions-to-ask-them)
8. [Logistics & Final Checklist](#8-logistics--final-checklist)

---

## 1. Role Overview

### What the Team Does
This is a **simulation team** focused on electromagnetic/Multiphysics simulation — NOT chip placement/EDA optimization. The role involves using ML to accelerate physics-based simulations.

### Key Responsibilities (from JD)
| Responsibility | Your Relevant Experience |
|----------------|-------------------------|
| Study and summarize existing ML algorithms and frameworks | Extensive literature reviews in all three papers |
| Apply ML techniques to make predictions on decisions during simulation | IEEE JESTIE: Neural networks predicting EMT states |
| Dynamically refine predictions as more information becomes available | EDISCO & DAC: Iterative diffusion refinement |
| Summarize findings and present research achievements | Conference papers, publications |

### Requirements
- ✅ Full-time degree at Canadian university (University of Alberta)
- ✅ PhD in ECE (strongly preferred)
- ✅ Familiarity with ML and prior ML projects (3 papers)
- ✅ Python proficiency (PyTorch, all papers)
- ✅ Understanding of physics of electromagnetics (IEEE JESTIE)
- ⚠️ C++ (FPGA implementation experience)

---

## 2. Your Research Summary

### Paper 1: IEEE JESTIE — Hardware-in-the-Loop ML Simulation

**Title:** Hardware-in-the-Loop Real-Time Transient Emulation of Large-Scale Renewable Energy Installations Based on Hybrid Machine Learning Modeling

**Venue:** IEEE Journal of Emerging and Selected Topics in Industrial Electronics (Published 2024)

**Authors:** Ruogu Chen, Tianshi Cheng, Ning Lin, Tian Liang, Venkata Dinavahi

#### Problem
Traditional electromagnetic transient (EMT) simulation of large-scale renewable energy systems (wind farms, PV arrays, batteries) is computationally expensive due to:
- Complex nonlinear differential equations
- Sub-millisecond timesteps (50 µs) required
- Large-scale systems (IEEE 118-bus + microgrids)

#### Your Solution
Hybrid neural network approach:
- **GRUs** for time-dependent components (wind turbines, batteries) — captures temporal dynamics
- **MLPs** for stateless components (PV panels) — simpler I/O mapping
- Trained on validated EMT simulation data

#### Key Technical Details
| Component | Neural Network | Architecture | Input Features | Output |
|-----------|---------------|--------------|----------------|--------|
| DFIG Wind Turbine | GRU | 2 layers, hidden=128 | Wind speed, grid voltage, rotor speed | Stator/rotor currents, torque |
| Li-ion Battery | GRU | 2 layers, hidden=64 | SOC, load current, temperature | Terminal voltage, internal states |
| PV Array (4×4) | MLP | 4 layers, hidden=64 | Irradiance, terminal voltage | Output current |

#### FPGA Implementation
- **Platform:** Xilinx VCU118 (UltraScale+ XCVU9P)
- **Quantization:** Dynamic (GRU) and Static (MLP) — int8 precision
- **Optimizations:** LUT-based activation functions, pipelined loops

#### Results
| Model | Traditional Latency | ML Latency | Speedup (FTRT Ratio) |
|-------|--------------------:|------------|---------------------|
| Wind Farm (180 turbines) | 462.5 µs | 15 µs | 3.33× |
| Battery (50 MW) | — | 10 µs | 5× |
| PV Array (320 panels) | — | 6 µs | 8.33× |
| Accuracy | — | <0.01% relative error | — |

#### Why This Matters for Synopsys
This paper demonstrates exactly what the job asks for:
- **ML for physics simulation** — replacing expensive solvers with learned models
- **Real-time prediction** — neural networks predicting system states during simulation
- **Hardware deployment** — practical implementation on FPGA
- **Electromagnetics domain** — directly relevant physics background

---

### Paper 2: DAC 2026 — Chip Placement with Policy Gradient Diffusion

**Title:** Energy-Guided Continuous Diffusion for Unsupervised Chip Placement

**Venue:** Design Automation Conference 2026 (Under Review)

#### Problem
Existing chip placement methods have limitations:
- **Analytical methods (DREAMPlace):** Converge to local optima, require manual tuning
- **RL methods (CircuitTraining):** Poor generalization, expensive per-circuit retraining
- **Supervised diffusion:** Require optimal placements as training data (expensive)

#### Your Solution
Policy gradient-trained diffusion model with:
1. **Unsupervised learning** — no optimal placement examples needed
2. **Dense per-step energy feedback** — compute wirelength/overlap at every diffusion step
3. **Zero-shot generalization** — train on synthetic data, deploy on real benchmarks

#### Key Technical Details

**Energy Function:**
```
E(x) = E_wire(x) + λ_overlap·E_overlap(x) + λ_bound·E_boundary(x)
```
- `E_wire`: Half-perimeter wirelength (HPWL)
- `E_overlap`: Pairwise component overlap penalty
- `E_boundary`: Canvas boundary violation penalty

**Training Method:**
- PPO (Proximal Policy Optimization) with GAE
- Per-step reward includes energy at current state (not just terminal)
- Reduces gradient variance by 3.2× compared to sparse-reward methods

**Architecture:**
- GNN-based encoder-process-decoder
- 16 GNN message-passing layers, 256 hidden dimensions
- Conditions on netlist graph G

#### Results
| Method | HPWL (×10⁷) | Legality | Inference Time |
|--------|------------:|----------|----------------|
| DREAMPlace | 3.724 | 1.0 | 0.48 min |
| ChipDiffusion (supervised) | 2.976 | 0.998 | 4.2 min |
| **Ours (unsupervised)** | **2.84** | **0.998** | 4.4 min |

- **3.7–5.6% HPWL improvement** over supervised diffusion baselines
- **Zero-shot transfer** from synthetic data to ISPD 2005, ICCAD 2004 benchmarks

#### Why This Matters for Synopsys
- **EDA domain expertise** — directly relevant to Synopsys products
- **Iterative refinement** — diffusion progressively improves solutions
- **Dense feedback during generation** — exactly what JD asks ("dynamically refine predictions")

---

### Paper 3: ICML 2026 — EDISCO: Equivariant Diffusion for TSP

**Title:** EDISCO: Equivariant Continuous-Time Categorical Diffusion for Geometric Combinatorial Optimization

**Venue:** International Conference on Machine Learning 2026 (Under Review)

#### Problem
Existing neural TSP solvers don't exploit geometric symmetries:
- TSP solutions are invariant under rotation, translation, reflection (E(2) group)
- Standard GNNs and diffusion models don't preserve these symmetries
- Results in poor sample efficiency and generalization

#### Your Solution
First diffusion model combining:
1. **E(2)-equivariant GNN (EGNN)** — respects geometric transformations
2. **Continuous-time categorical diffusion** — analytical forward/reverse processes
3. **Adaptive mixing strategy** — balances exploration and exploitation

#### Key Technical Details

**E(2)-Equivariance:**
- Node positions transform equivariantly: `x' = Rx + t`
- Edge features (adjacency) transform invariantly
- 12-layer EGNN with 256 hidden dimensions (5.5M parameters)

**Continuous-Time Formulation:**
- Based on Continuous-Time Markov Chains (CTMCs)
- Enables analytical transition probabilities
- Compatible with advanced solvers (PNDM, DEIS)
- 2-3× faster inference vs. discrete-time methods

**Training:**
- Supervised learning on optimal TSP solutions
- Only requires 33-50% of training data compared to baselines

#### Results
| Method | TSP-50 | TSP-100 | TSP-500 | TSP-1000 | TSP-10000 |
|--------|-------:|--------:|--------:|---------:|----------:|
| DIFUSCO | 0.48% | 1.01% | 9.41% | 11.24% | 8.95% |
| T2T | 0.04% | 0.18% | 5.09% | 8.87% | 2.92% |
| CADO | 0.01% | 0.08% | 2.30% | 3.33% | — |
| **EDISCO (ours)** | **0.01%** | **0.04%** | **1.95%** | **2.85%** | **1.20%** |

With sampling + 2-opt refinement:
- **TSP-500:** 0.08% gap (SOTA, down from 0.12%)
- **TSP-1000:** 0.22% gap (SOTA, down from 0.30%)
- **TSP-10000:** 1.20% gap (SOTA, down from 2.68%)

#### Cross-Distribution Generalization
| Distribution | EDISCO | DIFUSCO | T2T |
|--------------|-------:|--------:|----:|
| Uniform | 0.04% | 1.01% | 0.18% |
| Cluster | 0.05% | 2.87% | 1.50% |
| Explosion | 0.03% | 1.38% | 0.15% |
| Implosion | 0.05% | 2.80% | 2.60% |
| **Avg Deterioration** | **4%** | **133%** | **687%** |

#### Why This Matters for Synopsys
- **Combinatorial optimization** — routing, scheduling in EDA are CO problems
- **Iterative refinement** — diffusion denoising as progressive solution improvement
- **Geometric awareness** — exploiting problem structure for efficiency
- **Continuous-time formulation** — connects to ODE/SDE solvers used in physics simulation

---

## 3. Mapping Your Experience to the Job

| JD Requirement | Your Experience | Paper |
|----------------|-----------------|-------|
| "Apply ML techniques to make predictions on decisions during simulation" | GRU/MLP predicting EMT states at each timestep | IEEE JESTIE |
| "Dynamically refine predictions as more information becomes available" | Dense per-step energy feedback in diffusion; iterative denoising | DAC, EDISCO |
| "Understanding of physics of electromagnetics" | EMT simulation, DFIG wind turbines, power systems | IEEE JESTIE |
| "Python proficiency" | PyTorch for all three papers | All |
| "C++ proficiency" | FPGA implementation, Vitis HLS | IEEE JESTIE |
| "PhD strongly preferred" | PhD candidate in ECE | — |

### Key Narrative
Your research has two complementary threads that perfectly match this role:

1. **ML for Physics Simulation (IEEE JESTIE)**
   - Learned to replace expensive physics solvers with neural networks
   - Real-time deployment on hardware (FPGA)
   - Electromagnetics domain knowledge

2. **Iterative Refinement with Generative Models (DAC, EDISCO)**
   - Diffusion models that progressively improve solutions
   - Dense feedback at each step (not just terminal)
   - Policy gradient training without supervised labels

**The Synopsys role asks for exactly the combination of these two threads.**

---

## 4. Elevator Pitch

### 30-Second Version
> "I'm a PhD candidate in ECE at the University of Alberta, working on machine learning for simulation and optimization. My most relevant work is a published IEEE paper where I used neural networks to accelerate electromagnetic transient simulation — achieving 3.3x faster-than-real-time on FPGA. I also have two papers under review on diffusion models: one for chip placement at DAC, and one for combinatorial optimization at ICML. I'm excited about this role because it combines exactly what I've been researching — using ML to make predictions that accelerate physics-based simulation."

### 60-Second Version
> "I'm a PhD candidate in ECE at the University of Alberta. My research focuses on using machine learning to accelerate complex simulations and optimization.
>
> Most relevant to this role, I published a paper in IEEE JESTIE where I developed hybrid neural networks — GRUs and MLPs — to replace expensive electromagnetic transient simulations for power systems. I trained these models on validated EMT data, then deployed them on Xilinx FPGAs. The wind farm model went from 462 microseconds per step to 15 microseconds — a 30x speedup with less than 0.01% error.
>
> I also have two papers under review on diffusion models. My DAC submission uses policy-gradient-trained diffusion for chip placement, where I compute energy at every diffusion step — not just at the end — to provide dense feedback during generation. My ICML submission introduces E(2)-equivariant diffusion for TSP, achieving state-of-the-art results.
>
> This role is exciting because it combines both threads of my research: ML for physics simulation and iterative refinement during generation."

---

## 5. Likely Questions & Answers

### Background Questions

#### "Tell me about yourself / Walk me through your background"

> "I'm a fourth-year PhD candidate in Electrical and Computer Engineering at the University of Alberta, working with Professor Jie Han. My research sits at the intersection of machine learning and hardware systems.
>
> I have three main research projects. First, I published a paper in IEEE JESTIE on using neural networks for real-time electromagnetic transient simulation — I deployed hybrid MLP/GRU models on FPGAs to achieve faster-than-real-time emulation of renewable energy systems.
>
> Second, I have a paper under review at DAC on using policy-gradient-trained diffusion models for chip placement. This is unsupervised — no optimal placements needed — and achieves zero-shot generalization to real benchmarks.
>
> Third, I have a paper under review at ICML on equivariant diffusion models for the Traveling Salesman Problem, achieving state-of-the-art results by exploiting geometric symmetries.
>
> I'm excited about Synopsys because my research directly aligns with using ML to accelerate physics simulation."

#### "What's your programming experience?"

> "Python is my primary language — I use PyTorch for all my ML research. For my IEEE JESTIE paper, I also worked extensively with C++ for FPGA implementation using Vitis HLS. I'm comfortable with the full ML pipeline: data preprocessing, model design, training, and deployment."

---

### Technical Questions

#### "Tell me about your most relevant research project"

> "My IEEE JESTIE paper is most directly relevant. The problem was that electromagnetic transient simulation of large-scale renewable energy systems — like wind farms with hundreds of turbines — is computationally expensive because you need to solve complex nonlinear differential equations at 50-microsecond timesteps.
>
> My solution was to train neural networks to learn the input-output mapping directly. I used GRUs for time-dependent components like wind turbines, because the current state depends on history. For stateless components like PV panels, I used simpler MLPs.
>
> I trained these on data from validated EMT simulations, then deployed on a Xilinx FPGA. The wind farm model achieved 15 microseconds latency compared to 462 microseconds for the traditional solver — a 30x speedup — while maintaining less than 0.01% relative error.
>
> This is directly applicable to EM simulation at Synopsys: using ML to predict simulation states rather than solving the full physics at each step."

#### "How would you approach using ML to accelerate simulation?"

> "Based on my experience, I'd consider several approaches:
>
> **1. Surrogate modeling** — Train neural networks to approximate expensive physics computations. This is what I did for power systems in my JESTIE paper. For EM simulation, you could learn field distributions given geometry and excitation.
>
> **2. Iterative refinement** — My diffusion work shows that you can start from a coarse or noisy state and progressively refine it. Each step conditions on current state to predict improvements. This could accelerate iterative solvers.
>
> **3. Adaptive decision-making** — The job description mentions 'predictions on decisions during simulation.' In my DAC paper, I compute placement quality at every step, not just at the end. Similarly, ML could predict which regions need finer mesh, or when to switch solver strategies.
>
> **4. Hybrid approaches** — Combine ML predictions with traditional solvers. Use ML for fast approximate predictions, then refine with physics-based methods where needed."

#### "Explain your diffusion model work"

> "I have two diffusion papers. Let me explain the DAC paper since it's most relevant to simulation.
>
> The core idea is that diffusion models iteratively refine solutions — starting from random noise and progressively denoising to a clean output. For chip placement, I start with random component positions and iteratively improve them.
>
> The key innovation is how I train it. Instead of supervised learning from optimal placements, I use policy gradients — treating each denoising step as a decision in an RL framework. Critically, I compute the energy (wirelength + overlap penalties) at every diffusion step, not just at the end. This provides dense feedback: at step t, the model sees how good the current placement is and learns how to improve it.
>
> This reduced gradient variance by 3.2x compared to sparse-reward methods and enabled training purely from synthetic data with zero-shot transfer to real benchmarks.
>
> For simulation, the same principle applies: you could use diffusion to iteratively refine an initial coarse solution, with ML predicting how to improve at each step."

#### "What's your experience with electromagnetics?"

> "My background is in electrical engineering, so I have foundational knowledge from coursework — Maxwell's equations, wave propagation, transmission lines, EM fields.
>
> More importantly, my IEEE JESTIE paper involved electromagnetic transient simulation of power systems. I worked with EMT models at 50-microsecond timesteps to capture fast transient phenomena like fault currents and inverter switching. I modeled doubly-fed induction generators, which involve complex electromagnetic interactions between stator and rotor circuits.
>
> While I haven't worked specifically on EM simulation software like what Synopsys develops, I understand the physics principles and have demonstrated I can apply ML to physics-based simulation effectively."

#### "How do you handle dynamically refining predictions?"

> "This is central to my diffusion research.
>
> In EDISCO, I use continuous-time categorical diffusion over discrete edge variables. The model starts with pure noise (random edges) and iteratively refines toward a valid TSP tour. At each timestep, it conditions on the current noisy state to predict the denoising direction.
>
> In my DAC paper, I go further by computing energy at every step. The per-step reward includes the current placement quality, not just the final outcome. This means the model receives immediate feedback about whether its refinement is helping.
>
> For simulation, I'd envision a similar approach: make initial predictions, observe intermediate results, and update predictions accordingly. This could be framed as online learning within a single simulation run."

---

### Motivation Questions

#### "Why Synopsys?"

> "Synopsys is the leader in EDA and simulation tools. My research on chip placement directly targets problems that Synopsys solvers address, and my work on ML for electromagnetic simulation aligns exactly with this role.
>
> I want to work somewhere my research can have real-world impact. Synopsys tools are used throughout the semiconductor industry, so improvements in simulation speed could affect chip designs globally.
>
> This specific role is exciting because it combines my two research threads: ML for physics simulation and iterative refinement methods. It's rare to find a position that matches so closely."

#### "Why this role specifically?"

> "The job description mentions using ML to 'make predictions on decisions during simulation' and 'dynamically refine predictions.' This is exactly what my research does.
>
> My IEEE JESTIE paper shows I can build ML models that predict physics simulation states in real-time. My diffusion papers show I can design iterative refinement processes with feedback at each step.
>
> I'm also excited about the electromagnetics focus. I have hands-on experience with EMT simulation from my JESTIE work, so I can contribute domain knowledge, not just ML expertise."

#### "What do you hope to learn from this internship?"

> "I want to understand how ML research translates to production simulation tools. In academia, I focus on benchmark datasets and algorithmic innovations. At Synopsys, I'd learn about real-world constraints: what accuracy is needed, what latency is acceptable, how models integrate with existing codebases.
>
> I'm also curious about the specific EM simulation challenges at Synopsys. My power systems background gives me intuition, but I'd learn about different physics regimes and problem scales."

---

## 6. Technical Concepts to Review

Since the role involves physics simulation, brush up on:

| Topic | Key Concepts | Relevance |
|-------|--------------|-----------|
| **Surrogate modeling** | Neural networks approximating expensive simulations; training on physics simulation data | Core approach in your JESTIE paper |
| **Physics-informed neural networks (PINNs)** | Embedding physics constraints (PDEs) into neural network training | Alternative approach to pure data-driven |
| **Neural operators** | DeepONet, Fourier Neural Operator — learn mappings between function spaces | State-of-the-art for PDE solving |
| **Diffusion models** | DDPM, score-based models, continuous-time formulations | Your EDISCO/DAC expertise |
| **Iterative solvers** | Conjugate gradient, GMRES — ML could accelerate convergence | Potential application area |
| **Adaptive mesh refinement** | Could ML predict where to refine mesh? | Potential application area |
| **EM simulation methods** | FDTD, FEM, MoM — different numerical approaches | Background knowledge |

---

## 7. Questions to Ask Them

### About the Team & Projects
1. "What type of simulations does the team work on — is it primarily electromagnetic, or does it span other physics domains like thermal or structural?"

2. "Are you currently using any ML approaches in production, or is this more exploratory research?"

3. "What are the main computational bottlenecks in your simulation workflows that ML could address?"

### About the Role
4. "Would I be working with existing simulation codebases, or developing standalone ML prototypes?"

5. "How does the team balance simulation accuracy versus speedup when introducing ML approximations?"

6. "What does a typical day look like for an intern on this team?"

### About Growth & Conversion
7. "How have past interns' projects influenced Synopsys products or research directions?"

8. "What's the path from internship to full-time? How often do interns convert?"

---

## 8. Logistics & Final Checklist

### Key Facts to Confirm
| Item | Your Answer |
|------|-------------|
| **Start Date** | May 2026 |
| **Duration** | May – August 2026 (full-time, 40 hrs/week) |
| **Location** | Canada (hybrid/remote TBD) |
| **Work Authorization** | Valid Canadian work authorization ✓ |
| **Time Zone** | You're in MST (Edmonton) |

### Pre-Call Checklist
- [ ] Confirm interview time and time zone (4 PM MST = 6 PM EST)
- [ ] Test video/audio if video call
- [ ] Have this document open for reference
- [ ] Have your papers accessible (in case they ask specifics)
- [ ] Prepare 2-3 questions to ask
- [ ] Quiet environment with good internet

### Key Numbers to Remember
| Metric | Value | Paper |
|--------|-------|-------|
| Wind farm speedup | 30× (462 µs → 15 µs) | IEEE JESTIE |
| FTRT ratio | 3.33× | IEEE JESTIE |
| Model accuracy | <0.01% relative error | IEEE JESTIE |
| HPWL improvement | 3.7–5.6% over supervised | DAC |
| Gradient variance reduction | 3.2× | DAC |
| TSP-500 gap | 0.08% (SOTA) | EDISCO |
| TSP-10000 gap | 1.20% (SOTA) | EDISCO |
| Training data reduction | 33–50% less | EDISCO |

### Final Tips
- **Be conversational** — this is a recruiter screen, not a technical deep-dive
- **Lead with IEEE JESTIE** — it's your most relevant work for this role
- **Show enthusiasm** for the specific intersection of ML + physics simulation
- **Keep answers concise** — 1-2 minutes max per question
- **Connect everything back** to what Synopsys does and what the JD asks for

---

*Good luck with the interview!*
