"""
simulator.py — Simulation Loop

run_one_game()     : single 40-draw game, returns GameResult
run_simulation()   : N independent games, returns SimulationResult
run_demo_game()    : single game with step-by-step console narration
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from game_env   import CardGameState
from ev_engine  import compute_ev, compute_variance, get_state_snapshot
from market_maker import MarketMaker
from traders    import MixedFlow


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Single step snapshot stored during a game."""
    step:           int
    ev:             float
    bid:            float
    ask:            float
    spread:         float
    variance:       float
    true_sum:       float
    action:         str
    trader_type:    str
    inventory:      float
    cash:           float


@dataclass
class GameResult:
    pnl:            float
    n_fills:        int
    steps:          List[StepRecord]
    settlement:     float
    final_inventory: float

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class SimulationResult:
    game_results:   List[GameResult]
    pnl_series:     np.ndarray          # cumulative P&L
    per_game_pnl:   np.ndarray

    # Aggregate statistics
    total_pnl:      float
    mean_pnl:       float
    std_pnl:        float
    sharpe:         float
    win_rate:       float
    max_drawdown:   float
    n_games:        int
    elapsed_s:      float

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  SIMULATION RESULTS  ({self.n_games:,} games)\n"
            f"{'='*55}\n"
            f"  Total P&L          : {self.total_pnl:>12,.2f}\n"
            f"  Mean P&L / game    : {self.mean_pnl:>12.2f}\n"
            f"  Std  P&L / game    : {self.std_pnl:>12.2f}\n"
            f"  Sharpe Ratio       : {self.sharpe:>12.2f}\n"
            f"  Win Rate           : {self.win_rate:>11.1%}\n"
            f"  Max Drawdown       : {self.max_drawdown:>12.2f}\n"
            f"  Time               : {self.elapsed_s:>11.1f}s\n"
            f"{'='*55}\n"
        )


# ---------------------------------------------------------------------------
# Single-game runner
# ---------------------------------------------------------------------------

def run_one_game(
    mm:          MarketMaker,
    flow:        MixedFlow,
    state:       CardGameState,
    seed:        Optional[int] = None,
    record_steps: bool = True,
) -> GameResult:
    """
    Execute one full 40-draw game.

    Sequence per step:
      1. Compute bid / ask
      2. Invite flow to respond (action = buy / sell / pass)
      3. Process fill if any
      4. Draw next card (reveals info, updates state)

    At step 40: settle inventory against true remaining sum.
    """
    state.shuffle(seed=seed)
    mm.reset()

    steps: List[StepRecord] = []

    for step_idx in range(CardGameState.N_DRAWS):
        snap         = get_state_snapshot(state, k=mm.k)
        bid, ask     = mm.quote(state)
        action, ttype = flow.act(bid, ask, state)

        if action == "buy":
            mm.fill_buy(ask, step=step_idx, trader_type=ttype)
        elif action == "sell":
            mm.fill_sell(bid, step=step_idx, trader_type=ttype)

        if record_steps:
            steps.append(StepRecord(
                step        = step_idx,
                ev          = snap["ev"],
                bid         = bid,
                ask         = ask,
                spread      = snap["spread"],
                variance    = snap["variance"],
                true_sum    = snap["true_sum"],
                action      = action,
                trader_type = ttype,
                inventory   = mm.inventory,
                cash        = mm.cash,
            ))

        state.draw_card()     # reveal next card

    # Settlement
    settlement   = float(state.true_remaining_sum())
    pnl          = mm.settle(settlement)

    return GameResult(
        pnl              = pnl,
        n_fills          = mm.n_fills,
        steps            = steps,
        settlement       = settlement,
        final_inventory  = mm.inventory,
    )


# ---------------------------------------------------------------------------
# Multi-game simulation
# ---------------------------------------------------------------------------

def run_simulation(
    n_games:     int   = 10_000,
    k:           float = 0.5,
    adversarial: bool  = False,
    seed:        Optional[int] = 42,
    verbose:     bool  = True,
) -> SimulationResult:
    """
    Run N independent games and return aggregate statistics.

    Parameters
    ----------
    n_games     : number of games
    k           : spread aggression parameter
    adversarial : if True, use 80% HardInformed flow
    seed        : master random seed (each game offsets from this)
    verbose     : print progress every 1000 games
    """
    t0    = time.time()
    mm    = MarketMaker(k=k)
    state = CardGameState()

    results:     List[GameResult] = []
    pnl_cumul:   List[float]       = []
    running_pnl: float             = 0.0
    peak:        float             = 0.0
    max_dd:      float             = 0.0

    for i in range(n_games):
        game_seed = (seed + i) if seed is not None else None
        flow      = (MixedFlow.adversarial(seed=game_seed)
                     if adversarial
                     else MixedFlow.normal(seed=game_seed))

        result     = run_one_game(mm, flow, state, seed=game_seed, record_steps=False)
        results.append(result)

        running_pnl += result.pnl
        pnl_cumul.append(running_pnl)
        peak         = max(peak, running_pnl)
        drawdown     = peak - running_pnl
        max_dd       = max(max_dd, drawdown)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Game {i+1:>6,} / {n_games:,}  "
                  f"Cumulative P&L: {running_pnl:>10,.0f}")

    per_game = np.array([r.pnl for r in results])
    mean_pnl = float(per_game.mean())
    std_pnl  = float(per_game.std())
    sharpe   = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
    win_rate = float((per_game > 0).mean())

    elapsed = time.time() - t0

    sim = SimulationResult(
        game_results  = results,
        pnl_series    = np.array(pnl_cumul),
        per_game_pnl  = per_game,
        total_pnl     = running_pnl,
        mean_pnl      = mean_pnl,
        std_pnl       = std_pnl,
        sharpe        = sharpe,
        win_rate      = win_rate,
        max_drawdown  = max_dd,
        n_games       = n_games,
        elapsed_s     = elapsed,
    )

    if verbose:
        print(sim.summary())

    return sim


# ---------------------------------------------------------------------------
# Demo game — step-by-step narrated output
# ---------------------------------------------------------------------------

def run_demo_game(seed: int = 7, k: float = 0.5) -> GameResult:
    """
    Run a single game with detailed console narration of every Bayesian update.
    Mirrors the live demo described in the interview playbook.
    """
    print("\n" + "="*65)
    print("  EV MARKET MAKER — LIVE DEMO")
    print("="*65)
    print(f"  Contract: sum of last {CardGameState.N_SETTLE} cards after "
          f"{CardGameState.N_DRAWS} draws")
    print(f"  Starting EV = {CardGameState.N_SETTLE} × (340/52) = "
          f"{CardGameState.N_SETTLE * 340/52:.2f}")
    print("="*65 + "\n")

    state = CardGameState()
    mm    = MarketMaker(k=k)
    flow  = MixedFlow.normal(seed=seed)

    state.shuffle(seed=seed)
    steps: List[StepRecord] = []

    for step_idx in range(CardGameState.N_DRAWS):
        snap        = get_state_snapshot(state, k=k)
        bid, ask    = mm.quote(state)
        action, tt  = flow.act(bid, ask, state)

        # Console output (every step up to 10, then every 5)
        if step_idx < 10 or step_idx % 5 == 0:
            print(f"  Step {step_idx:>2} | Card drawn so far: {state.n_drawn:>2} | "
                  f"EV={snap['ev']:>7.2f}  "
                  f"Bid={bid:>7.2f}  Ask={ask:>7.2f}  "
                  f"Spread={snap['spread']:>5.2f}  "
                  f"σ={snap['std']:>5.2f}  "
                  f"Action={action:<5} ({tt})")

        if action == "buy":
            mm.fill_buy(ask, step_idx, tt)
        elif action == "sell":
            mm.fill_sell(bid, step_idx, tt)

        steps.append(StepRecord(
            step=step_idx, ev=snap["ev"], bid=bid, ask=ask,
            spread=snap["spread"], variance=snap["variance"],
            true_sum=snap["true_sum"], action=action,
            trader_type=tt, inventory=mm.inventory, cash=mm.cash,
        ))

        card = state.draw_card()

        if step_idx < 10 or step_idx % 5 == 0:
            shift = snap["ev"] - get_state_snapshot(state, k=k)["ev"]
            direction = "↓" if shift > 0 else "↑"
            print(f"        → Revealed: {card:>2}  |  EV moves {direction} "
                  f"({abs(shift):.2f})  |  Inv={mm.inventory:+.0f}")

    settlement = float(state.true_remaining_sum())
    pnl        = mm.settle(settlement)

    print(f"\n  SETTLEMENT: true remaining sum = {settlement}")
    print(f"  Final inventory : {mm.inventory:+.1f}")
    print(f"  Cash collected  : {mm.cash:.2f}")
    print(f"  Settlement P&L  : {mm.inventory * settlement:.2f}")
    print(f"  TOTAL P&L       : {pnl:.2f}")
    print(f"  Total fills     : {mm.n_fills}")
    print("="*65 + "\n")

    return GameResult(pnl=pnl, n_fills=mm.n_fills, steps=steps,
                      settlement=settlement, final_inventory=mm.inventory)