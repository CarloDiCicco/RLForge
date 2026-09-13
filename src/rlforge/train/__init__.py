"""
train/
──────
RL post-training (Milestone 3): the reward function and the GRPO run plumbing.

Design boundary worth stating once, because it is the thing that keeps an RL
number comparable to a benchmark number: **this package adds no verification
logic of its own.** The reward is the EXISTING evaluation stack
(extract → compile tests → sandbox) called in the same order `pipeline.py` calls
it, so "reward == 1.0" means exactly "the benchmark harness would have scored
this generation as a pass". Any divergence between the two would make the RL
result incomparable to the baselines it is measured against — the one error this
whole project cannot absorb.
"""

from rlforge.train.reward import VerifierReward

__all__ = ["VerifierReward"]
