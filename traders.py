"""
traders.py — Counterparty Models

Three trader types with increasing informational advantage:

  NoiseTrader        — random, no view on fair value (the MM's revenue source)
  SoftInformedTrader — uses public info more accurately than the MM
  HardInformedTrader — peeks at the next card (adversarial insider)
  MixedFlow          — weighted combination of all three

Default simulation mix: 60% noise / 30% soft / 10% hard
Adversarial stress test:  10% noise / 10% soft / 80% hard
"""

from __future__ import annotations
import random
from typing import Optional

import numpy as np

from game_env import CardGameState
from ev_engine import compute_ev


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseTrader:
    """Abstract interface for all traders."""

    def act(self, bid: float, ask: float, state: CardGameState) -> str:
        """
        Returns 'buy', 'sell', or 'pass'.
        'buy'  = counterparty buys at our ask  (bad for us if mispriced high)
        'sell' = counterparty sells at our bid (bad for us if mispriced low)
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Noise Trader
# ---------------------------------------------------------------------------

class NoiseTrader(BaseTrader):
    """
    Trades randomly with probability trade_prob.
    Buys or sells with equal probability — no view on fair value.
    These traders are the market maker's revenue source.
    Every trade they do pays the full spread.

    Parameters
    ----------
    trade_prob : float — probability of trading at each step (default 0.4)
    seed       : int   — optional random seed for reproducibility
    """

    def __init__(self, trade_prob: float = 0.4, seed: Optional[int] = None):
        self.trade_prob = trade_prob
        self._rng = random.Random(seed)

    def act(self, bid: float, ask: float, state: CardGameState) -> str:
        if self._rng.random() > self.trade_prob:
            return "pass"
        return "buy" if self._rng.random() < 0.5 else "sell"


# ---------------------------------------------------------------------------
# Soft Informed Trader
# ---------------------------------------------------------------------------

class SoftInformedTrader(BaseTrader):
    """
    Computes the true EV from public information and trades when the MM's
    mid-price deviates by more than edge_threshold.

    Represents a fast counterparty who processes information slightly
    more accurately than the MM — no special insider knowledge.

    Parameters
    ----------
    edge_threshold : float — minimum edge to trade (default 0.5)
    trade_prob     : float — probability of acting when edge exists (default 0.7)
    seed           : int
    """

    def __init__(
        self,
        edge_threshold: float = 0.5,
        trade_prob:     float = 0.7,
        seed:           Optional[int] = None,
    ):
        self.edge_threshold = edge_threshold
        self.trade_prob     = trade_prob
        self._rng           = random.Random(seed)

    def act(self, bid: float, ask: float, state: CardGameState) -> str:
        true_ev = compute_ev(state)              # same formula, no noise
        mm_mid  = (bid + ask) / 2.0

        if true_ev - mm_mid > self.edge_threshold:
            # MM is pricing too cheap — buy it
            if self._rng.random() < self.trade_prob:
                return "buy"
        elif mm_mid - true_ev > self.edge_threshold:
            # MM is pricing too rich — sell it
            if self._rng.random() < self.trade_prob:
                return "sell"

        return "pass"


# ---------------------------------------------------------------------------
# Hard Informed Trader (adversarial)
# ---------------------------------------------------------------------------

class HardInformedTrader(BaseTrader):
    """
    Peeks at the next card before it is drawn.
    Simulates the EV *after* that card is revealed and trades if the
    implied EV shift exceeds the spread.

    If the next card is low → remaining pool skews high → EV rises.
    Trader buys before the card drops, profits from the EV jump.

    Parameters
    ----------
    min_edge : float — minimum EV shift needed to trade (default 0.3)
    seed     : int
    """

    def __init__(self, min_edge: float = 0.3, seed: Optional[int] = None):
        self.min_edge = min_edge
        self._rng     = random.Random(seed)

    def act(self, bid: float, ask: float, state: CardGameState) -> str:
        if not state.remaining:
            return "pass"
        if state.draws_left <= 0:
            return "pass"

        # Peek at next card (insider knowledge)
        next_card = state.remaining[0]

        # Current EV
        current_ev = compute_ev(state)

        # Simulated EV after the next card is drawn (removed from pool)
        remaining_after = state.remaining[1:]
        if not remaining_after:
            return "pass"

        n_to_draw_after = state.draws_left - 1
        n_settle_after  = len(remaining_after) - n_to_draw_after

        if n_settle_after <= 0:
            return "pass"

        ev_after   = n_settle_after * float(np.mean(remaining_after))
        ev_shift   = ev_after - current_ev

        if ev_shift > self.min_edge:
            return "buy"        # EV will rise → buy now
        if ev_shift < -self.min_edge:
            return "sell"       # EV will fall → sell now

        return "pass"


# ---------------------------------------------------------------------------
# Mixed Flow
# ---------------------------------------------------------------------------

class MixedFlow:
    """
    Weighted combination of all three trader types.

    Default mix (normal):
        noise=0.60, soft=0.30, hard=0.10

    Adversarial stress test:
        noise=0.10, soft=0.10, hard=0.80

    At each step, one trader type is sampled according to the weights,
    and that trader's act() method is called.
    """

    NORMAL_MIX      = (0.60, 0.30, 0.10)
    ADVERSARIAL_MIX = (0.10, 0.10, 0.80)

    def __init__(
        self,
        noise_weight: float = 0.60,
        soft_weight:  float = 0.30,
        hard_weight:  float = 0.10,
        seed:         Optional[int] = None,
    ):
        total = noise_weight + soft_weight + hard_weight
        self.weights = [noise_weight / total,
                        soft_weight  / total,
                        hard_weight  / total]

        self._rng    = random.Random(seed)
        self.noise   = NoiseTrader(seed=seed)
        self.soft    = SoftInformedTrader(seed=seed)
        self.hard    = HardInformedTrader(seed=seed)
        self._traders = [self.noise, self.soft, self.hard]

    def act(self, bid: float, ask: float, state: CardGameState):
        """Returns (action, trader_name)."""
        idx    = self._rng.choices([0, 1, 2], weights=self.weights, k=1)[0]
        trader = self._traders[idx]
        action = trader.act(bid, ask, state)
        return action, trader.name

    @classmethod
    def normal(cls, seed: Optional[int] = None) -> "MixedFlow":
        return cls(*cls.NORMAL_MIX, seed=seed)

    @classmethod
    def adversarial(cls, seed: Optional[int] = None) -> "MixedFlow":
        return cls(*cls.ADVERSARIAL_MIX, seed=seed)