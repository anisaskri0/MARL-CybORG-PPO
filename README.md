# MARL-CybORG-PPO
 
This project is a proof-of-concept implementation of Proximal Policy Optimization (PPO) agents inside the CybORG CAGE Challenge 4 cyber-range.
Its goal is to explore how multi-agent reinforcement learning can automate blue-team deception and red-team attack strategies.

What it does
Environment setup – Wraps CybORG’s EnterpriseScenario with custom observation/action wrappers.

Agent definitions – Implements separate Red and Blue PPO agents (Torch + RLlib), each with a discrete padded action space and optional action masking.

Training loop – Uses Ray RLlib to train 5 parallel agents, logging episodic reward, infection rate, true/false positives, and action distributions.

Evaluation scripts – Runs saved checkpoints against fixed opponents.

Utilities – Helpers for SB3-compatible gym wrappers, seed control, and TensorBoard visualisation.

Why it matters

Research value – Shows how modern RL (PPO) can learn defense & deception-aware policies in a realistic cyber range.

Reproducible baseline – Provides ready-to-run configs and seeds so others can benchmark new algorithms or reward functions.

Foundation for thesis work – Serves as the training backbone for my engineering capstone on intelligent cyber defence.
