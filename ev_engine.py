"""
ev_engine.py — The EV Engine
Implements hypergeometric EV and variance formulas.
This is the mathematical core of the project.

Key formulas
------------
E[settlement | cards_seen] = n_settle × mean(remaining_pool)

Var[settlement]            = n_settle × σ²(remaining) × d / (R − 1)
                             where d = draws left, R = total remaining cards

Spread Width               = k × √Var
"""

import numpy as np
from typing import Dict, Tuple

from game_env import CardGameState


# ---------------------------------------------------------------------------
# Core EV & Variance
# ---------------------------------------------------------------------------

def compute_ev(state: CardGameState, n_to_draw: int = None) -> float:
    """
    Expected settlement sum given the cards revealed so far.

    Parameters
    ----------
    state     : CardGameState — current game state
    n_to_draw : int — draws remaining before settlement.
                      Defaults to state.draws_left.

    Returns
    -------
    float — conditional EV of the settlement sum
    """
    if n_to_draw is None:
        n_to_draw = state.draws_left

    remaining = state.remaining                 # list of card values in deck
    R         = len(remaining)                  # total cards remaining
    n_settle  = R - n_to_draw                   # cards that will settle

    if n_settle <= 0:
        return 0.0

    mean_remaining = float(np.mean(remaining))
    return n_settle * mean_remaining


def compute_variance(state: CardGameState, n_to_draw: int = None) -> float:
    """
    Hypergeometric variance of the settlement sum.

    Var = n_settle × σ²(remaining) × d / (R − 1)

    The factor d/(R-1) is the finite population correction (FPC).
    It collapses to 0 when d → 0 (settlement becomes certain).

    Parameters
    ----------
    state     : CardGameState
    n_to_draw : int — draws remaining (defaults to state.draws_left)

    Returns
    -------
    float — variance of settlement sum
    """
    if n_to_draw is None:
        n_to_draw = state.draws_left

    remaining = state.remaining
    R         = len(remaining)
    n_settle  = R - n_to_draw
    d         = n_to_draw

    if n_settle <= 0 or R <= 1 or d == 0:
        return 0.0

    # Population variance of remaining card values
    sigma2 = float(np.var(remaining))           # ddof=0 (full population)

    # Finite population correction
    fpc = d / (R - 1)

    return n_settle * sigma2 * fpc


def compute_spread(variance: float, k: float = 0.5) -> float:
    """
    Spread width = k × √variance

    Parameters
    ----------
    variance : float — from compute_variance()
    k        : float — aggression parameter (0.3 = tight, 1.0 = wide, 0.5 = default)

    Returns
    -------
    float — total bid-ask spread width
    """
    return k * float(np.sqrt(max(variance, 0.0)))


def compute_quote(
    state:     CardGameState,
    k:         float = 0.5,
    n_to_draw: int   = None,
) -> Tuple[float, float]:
    """
    Compute raw (unadjusted) bid and ask.

    Returns
    -------
    (bid, ask) : tuple of floats
    """
    ev     = compute_ev(state, n_to_draw)
    var    = compute_variance(state, n_to_draw)
    spread = compute_spread(var, k)
    half   = spread / 2.0
    return (ev - half, ev + half)


# ---------------------------------------------------------------------------
# State snapshot — for logging and visualisation
# ---------------------------------------------------------------------------

def get_state_snapshot(state: CardGameState, k: float = 0.5) -> Dict:
    """
    Bundle all numbers at a given step into a single dict.
    Includes ground-truth values for validation / charts.
    """
    n_to_draw = state.draws_left
    remaining = state.remaining
    R         = len(remaining)
    n_settle  = R - n_to_draw

    ev     = compute_ev(state, n_to_draw)
    var    = compute_variance(state, n_to_draw)
    spread = compute_spread(var, k)
    bid, ask = ev - spread / 2, ev + spread / 2

    true_sum = state.true_remaining_sum()

    return {
        "step":          state.n_drawn,
        "draws_left":    n_to_draw,
        "n_remaining":   R,
        "n_settle":      n_settle,
        "ev":            ev,
        "variance":      var,
        "std":           float(np.sqrt(max(var, 0))),
        "spread":        spread,
        "bid":           bid,
        "ask":           ask,
        "mid":           (bid + ask) / 2,
        "true_sum":      true_sum,
        "fpc":           (n_to_draw / (R - 1)) if R > 1 else 0.0,
        "mean_remaining": float(np.mean(remaining)) if remaining else 0.0,
    }


# ---------------------------------------------------------------------------
# Dice game EV (independent draws — no hypergeometric FPC)
# ---------------------------------------------------------------------------

def compute_dice_ev(n_remaining: int) -> float:
    """Expected sum for n_remaining independent d6 rolls."""
    return n_remaining * 3.5


def compute_dice_variance(n_remaining: int) -> float:
    """Variance of sum for n_remaining independent d6 rolls. Var_per = 35/12."""
    return n_remaining * (35 / 12)