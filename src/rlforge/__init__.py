"""
RLForge — verifier-guided post-training for code models.

This package holds the Milestone 1 infrastructure: the canonical data schemas,
the sandbox executor, code extraction, failure classification, the run registry,
and metrics. Everything downstream of ingestion speaks the canonical schemas
defined in `rlforge.schema`; the origin of a sample (real dataset vs synthetic)
is preserved only as a `source` field, never as a code path.
"""

# Single source of truth for the canonical-format version. Every persisted
# schema stamps this so old data stays interpretable as the format evolves.
# Bump only on a breaking schema change, and document the change in PLAN.md.
FORMAT_VERSION = "1.0.0"

__all__ = ["FORMAT_VERSION"]
