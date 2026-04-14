"""
market_maker.py — The Quoting Agent

Wraps the EV engine with practical trading logic:
  1. Inventory skew    — shade quotes to attract the opposite side
  2. Informed flow penalty — widen spread after consecutive one-sided hits
  3. Hard inventory limits — stop quoting the problematic side if inventory
                             breaches the hard limit
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from game_env import CardGameState
from ev_engine import compute_ev, compute_variance, compute_spread


# ---------------------------------------------------------------------------
# Fill record
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    step:       int
    side:       str         # 'buy' or 'sell'  (from counterparty perspective)
    price:      float
    trader_type: str


# ---------------------------------------------------------------------------
# MarketMaker
# ---------------------------------------------------------------------------

class MarketMaker:
    """
    Two-way market making agent on a CardGameState.

    Parameters
    ----------
    k             : spread aggression (0.5 = default)
    skew_factor   : inventory skew intensity (0.1 = default)
    hit_penalty   : spread multiplier per extra consecutive hit (1.3 = default)
    inventory_limit : hard inventory cap (6 = default)
    """

    INF = float("inf")

    def __init__(
        self,
        k:               float = 0.5,
        skew_factor:     float = 0.1,
        hit_penalty:     float = 1.3,
        inventory_limit: int   = 6,
    ):
        self.k               = k
        self.skew_factor     = skew_factor
        self.hit_penalty     = hit_penalty
        self.inventory_limit = inventory_limit

        # Runtime state
        self.inventory:         float      = 0.0
        self.cash:              float      = 0.0
        self.fills:             List[Fill] = []
        self.consecutive_hits:  int        = 0   # positive = consecutive buys, negative = sells
        self.pnl_history:       List[float] = []
        self.settled:           bool       = False

    # ------------------------------------------------------------------
    # Reset (called between games in simulation)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.inventory        = 0.0
        self.cash             = 0.0
        self.fills            = []
        self.consecutive_hits = 0
        self.pnl_history      = []
        self.settled          = False

    # ------------------------------------------------------------------
    # Core quoting method
    # ------------------------------------------------------------------
    def quote(
        self,
        state:     CardGameState,
        n_to_draw: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Compute adjusted bid and ask for the current game state.

        Adjustments applied on top of raw EV-centred quote:
          1. Inventory skew
          2. Informed-flow penalty
          3. Hard limit side removal

        Returns
        -------
        (bid, ask) : adjusted quotes. ±inf means that side is closed.
        """
        if n_to_draw is None:
            n_to_draw = state.draws_left

        ev   = compute_ev(state, n_to_draw)
        var  = compute_variance(state, n_to_draw)
        half = compute_spread(var, self.k) / 2.0

        # ---- 1. Raw symmetric quote ----
        bid = ev - half
        ask = ev + half

        # ---- 2. Inventory skew ----
        # Shift both quotes toward reducing inventory
        skew = self.inventory * self.skew_factor * half
        bid -= skew
        ask -= skew

        # ---- 3. Informed-flow penalty ----
        n_consec = abs(self.consecutive_hits)
        if n_consec >= 3:
            penalty = self.hit_penalty ** (n_consec - 2)
            extra   = half * (penalty - 1.0)
            bid    -= extra
            ask    += extra

        # ---- 4. Hard inventory limits ----
        if self.inventory >= self.inventory_limit:
            bid = -self.INF          # refuse to buy more
        if self.inventory <= -self.inventory_limit:
            ask = self.INF           # refuse to sell more

        return (bid, ask)

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------
    def fill_buy(self, ask: float, step: int, trader_type: str = "unknown") -> None:
        """
        Counterparty buys at our ask (we sell).
        Our cash increases, inventory decreases.
        """
        if ask == self.INF:
            return
        self.cash      += ask
        self.inventory -= 1.0
        self.fills.append(Fill(step=step, side="buy", price=ask, trader_type=trader_type))
        self._update_consecutive(-1)

    def fill_sell(self, bid: float, step: int, trader_type: str = "unknown") -> None:
        """
        Counterparty sells at our bid (we buy).
        Our cash decreases, inventory increases.
        """
        if bid == -self.INF:
            return
        self.cash      -= bid
        self.inventory += 1.0
        self.fills.append(Fill(step=step, side="sell", price=bid, trader_type=trader_type))
        self._update_consecutive(+1)

    def _update_consecutive(self, direction: int) -> None:
        """
        Track consecutive one-sided hits.
        +1 = consecutive sell-to-us (we accumulate longs).
        -1 = consecutive buy-from-us (we accumulate shorts).
        Resets if direction reverses.
        """
        if direction == 1:
            self.consecutive_hits = max(0, self.consecutive_hits) + 1
        else:
            self.consecutive_hits = min(0, self.consecutive_hits) - 1

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------
    def settle(self, settlement_price: float) -> float:
        """
        Called at game end.
        P&L = cash_collected_from_spread + inventory × settlement_price

        A long inventory profits if settlement_price > average_price_paid.
        """
        pnl = self.cash + self.inventory * settlement_price
        self.pnl_history.append(pnl)
        self.settled = True
        return pnl

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------
    @property
    def total_pnl(self) -> float:
        return sum(self.pnl_history)

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    def spread_at_step(self, state: CardGameState) -> float:
        bid, ask = self.quote(state)
        if bid == -self.INF or ask == self.INF:
            return self.INF
        return ask - bid

    def __repr__(self) -> str:
        return (f"MarketMaker(k={self.k}, inv={self.inventory:+.1f}, "
                f"cash={self.cash:.2f}, fills={self.n_fills})")