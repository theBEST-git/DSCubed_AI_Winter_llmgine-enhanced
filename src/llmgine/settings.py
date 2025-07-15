# ─────────────────────────────────────────────────────────
# Centralised config for engine‑level behaviour.
# Adjust the numbers to taste; they’re safe starting points.

# When the total conversation > MAX_CONTEXT_TOKENS,
# we trigger a summarisation pass.
MAX_CONTEXT_TOKENS: int = 3000

# Each summarisation run tries to condense roughly this many tokens
# (taken from the oldest messages) into a short summary.
PRUNE_TARGET_TOKENS: int = 1500
