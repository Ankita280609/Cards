"""
visualiser.py — All charts and visualisations

plot_spread_collapse()      : spread width vs cards drawn (the core visual)
plot_ev_convergence()       : EV tracking true sum across a single game
plot_pnl_curve()            : cumulative P&L, drawdown, win-rate
plot_pnl_distribution()     : per-game P&L histogram
plot_adversarial_comparison(): normal vs adversarial side-by-side
plot_variance_decay()       : hypergeometric variance decay curve
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from game_env    import CardGameState
from ev_engine   import compute_ev, compute_variance, compute_spread, get_state_snapshot
from market_maker import MarketMaker
from simulator   import GameResult, SimulationResult


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

BLUE   = "#1565C0"
GREEN  = "#2E7D32"
RED    = "#C62828"
ORANGE = "#E65100"
PURPLE = "#6A1B9A"
GREY   = "#90A4AE"
LIGHT  = "#E3F2FD"

def _style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.color":        "#E0E0E0",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    12,
        "axes.labelsize":    10,
    })

_style()


# ---------------------------------------------------------------------------
# 1. Spread Collapse
# ---------------------------------------------------------------------------

def plot_spread_collapse(k: float = 0.5, seed: int = 42, save_path: str = None):
    """
    Plot spread width vs cards drawn and hypergeometric variance decay.
    This is the core visual demonstrating the mathematical guarantee.
    """
    state = CardGameState()
    state.shuffle(seed=seed)

    spreads   = []
    variances = []
    steps     = []

    for i in range(CardGameState.N_DRAWS):
        snap = get_state_snapshot(state, k=k)
        spreads.append(snap["spread"])
        variances.append(snap["variance"])
        steps.append(i)
        state.draw_card()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Spread Collapse — The Core Visual", fontsize=14, fontweight="bold")

    # Left: spread width
    ax1.fill_between(steps, spreads, alpha=0.25, color=BLUE)
    ax1.plot(steps, spreads, color=BLUE, lw=2.0, label="Spread Width")
    ax1.axvline(x=38, color=RED, ls="--", lw=1.2, label="38 cards drawn")
    ax1.annotate(f"{spreads[0]:.2f}",
                 xy=(0, spreads[0]), xytext=(2, spreads[0]+0.1),
                 fontsize=9, color=BLUE, fontweight="bold")
    ax1.annotate(f"{spreads[-1]:.4f}",
                 xy=(steps[-1], spreads[-1]),
                 xytext=(steps[-1]-8, spreads[-1]+0.15),
                 fontsize=9, color=RED, fontweight="bold")
    ax1.set_xlabel("Cards Drawn")
    ax1.set_ylabel("Spread Width")
    ax1.set_title("Spread Width vs Cards Drawn")
    ax1.legend()

    # Right: variance
    ax2.fill_between(steps, variances, alpha=0.20, color=PURPLE)
    ax2.plot(steps, variances, color=PURPLE, lw=2.0)
    ax2.set_xlabel("Cards Drawn")
    ax2.set_ylabel("Variance")
    ax2.set_title("Hypergeometric Variance Decay")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. EV Convergence (single game)
# ---------------------------------------------------------------------------

def plot_ev_convergence(result: GameResult, save_path: str = None):
    """
    EV (blue) converging to the true settlement sum (green dashed).
    Shaded band shows the bid-ask spread narrowing.
    """
    steps      = [s.step for s in result.steps]
    evs        = [s.ev   for s in result.steps]
    bids       = [s.bid  for s in result.steps]
    asks       = [s.ask  for s in result.steps]
    true_sums  = [s.true_sum for s in result.steps]
    spreads    = [s.spread   for s in result.steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("EV Convergence During a Single Game", fontsize=14, fontweight="bold")

    ax1.fill_between(steps, bids, asks, alpha=0.25, color=BLUE, label="Bid-Ask Band")
    ax1.plot(steps, evs,      color=BLUE,  lw=2.0, label="EV (Mid)")
    ax1.plot(steps, true_sums, color=GREEN, lw=1.5, ls="--", label="True Settlement Sum")
    ax1.set_ylabel("Contract Value")
    ax1.legend()

    ax2.bar(steps, spreads, color=PURPLE, alpha=0.7, width=0.8)
    ax2.set_ylabel("Spread Width")
    ax2.set_xlabel("Cards Drawn")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. P&L Curve (simulation)
# ---------------------------------------------------------------------------

def plot_pnl_curve(sim: SimulationResult, title: str = "10,000-Game P&L Curve",
                   save_path: str = None):
    """
    Three-panel: cumulative P&L / drawdown / rolling win-rate.
    """
    pnl    = sim.pnl_series
    games  = np.arange(len(pnl))

    # Drawdown
    peak   = np.maximum.accumulate(pnl)
    dd     = pnl - peak

    # Rolling win rate (window=300)
    wins   = (sim.per_game_pnl > 0).astype(float)
    window = min(300, len(wins) // 5)
    rolling_win = np.convolve(wins, np.ones(window)/window, mode="valid")
    rw_x        = np.arange(window - 1, len(wins))

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        f"{title}  |  Sharpe {sim.sharpe:.2f}  |  Win Rate {sim.win_rate:.1%}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Cumulative P&L
    ax1.fill_between(games, pnl, alpha=0.15, color=GREEN)
    ax1.plot(games, pnl, color=GREEN, lw=1.5, label="Cumulative P&L")
    peak_line = np.maximum.accumulate(pnl)
    ax1.plot(games, peak_line, color=GREY, lw=0.8, ls="--", label="Running Peak")
    ax1.set_ylabel("Cumulative P&L")
    ax1.legend(fontsize=8)

    # Drawdown
    ax2.fill_between(games, dd, alpha=0.4, color=RED)
    ax2.plot(games, dd, color=RED, lw=0.8)
    ax2.set_ylabel("Drawdown")

    # Rolling win rate
    ax3.axhline(0.5, color=GREY, ls="--", lw=0.8)
    ax3.plot(rw_x, rolling_win, color=ORANGE, lw=1.2,
             label=f"Win Rate (rolling {window}g)")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel(f"Win Rate\n({window}g)")
    ax3.set_xlabel("Game #")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Per-game P&L distribution
# ---------------------------------------------------------------------------

def plot_pnl_distribution(sim: SimulationResult, save_path: str = None):
    """Histogram of per-game P&L with mean / break-even lines."""
    pnl = sim.per_game_pnl

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        f"Distribution of Per-Game P&L ({sim.n_games:,} Games)",
        fontsize=13, fontweight="bold"
    )

    n, bins, _ = ax.hist(pnl, bins=60, color=BLUE, alpha=0.75, edgecolor="white")
    ax.axvline(0,            color=RED,   lw=1.5, ls="--", label="Break-even")
    ax.axvline(pnl.mean(),   color=GREEN, lw=1.5, ls="-",  label=f"Mean = {pnl.mean():.1f}")
    ax.set_xlabel("P&L per Game")
    ax.set_ylabel("Frequency")
    ax.legend()

    txt = (f"μ = {pnl.mean():.2f}\n"
           f"σ = {pnl.std():.2f}\n"
           f"Win% = {(pnl>0).mean():.1%}")
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, va="top", ha="right",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Adversarial comparison
# ---------------------------------------------------------------------------

def plot_adversarial_comparison(
    normal_sim:      SimulationResult,
    adversarial_sim: SimulationResult,
    save_path:       str = None,
):
    """Side-by-side cumulative P&L for normal vs adversarial flow."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Normal Flow vs Adversarial Flow (80% Hard-Informed)",
                 fontsize=13, fontweight="bold")

    def _draw(ax, sim, title, color):
        pnl   = sim.pnl_series
        games = np.arange(len(pnl))
        ax.fill_between(games, pnl, alpha=0.15, color=color)
        ax.plot(games, pnl, color=color, lw=1.5)
        ax.set_title(f"{title}\nSharpe {sim.sharpe:.2f}  |  Win Rate {sim.win_rate:.1%}",
                     fontsize=11)
        ax.set_xlabel("Game #")
        ax.set_ylabel("Cumulative P&L")

    _draw(ax1, normal_sim,      "Normal Flow (60% Noise)",          GREEN)
    _draw(ax2, adversarial_sim, "Adversarial (80% Hard-Informed)",  RED)

    # Metrics table
    metrics = ["Sharpe Ratio", "Win Rate", "Mean P&L/Game",
               "Std P&L/Game", "Max Drawdown"]
    norm_v  = [f"{normal_sim.sharpe:.2f}",
               f"{normal_sim.win_rate:.1%}",
               f"{normal_sim.mean_pnl:.2f}",
               f"{normal_sim.std_pnl:.2f}",
               f"{normal_sim.max_drawdown:.0f}"]
    adv_v   = [f"{adversarial_sim.sharpe:.2f}",
               f"{adversarial_sim.win_rate:.1%}",
               f"{adversarial_sim.mean_pnl:.2f}",
               f"{adversarial_sim.std_pnl:.2f}",
               f"{adversarial_sim.max_drawdown:.0f}"]

    print("\n{:<20} {:>15} {:>15}".format("Metric", "Normal", "Adversarial"))
    print("-" * 52)
    for m, n, a in zip(metrics, norm_v, adv_v):
        print(f"{m:<20} {n:>15} {a:>15}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Finite population correction curve
# ---------------------------------------------------------------------------

def plot_fpc_curve(save_path: str = None):
    """
    Illustrate how the finite population correction d/(R-1) decays
    as more cards are drawn.
    """
    N = CardGameState.DECK_SIZE
    d_vals = np.arange(0, CardGameState.N_DRAWS + 1)
    R_vals = N - d_vals          # cards remaining after d draws
    fpc    = np.where(R_vals > 1, d_vals / (R_vals - 1), 0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title("Finite Population Correction  d / (R − 1)",
                 fontsize=13, fontweight="bold")
    ax.fill_between(d_vals, fpc, alpha=0.2, color=PURPLE)
    ax.plot(d_vals, fpc, color=PURPLE, lw=2.0)
    ax.set_xlabel("Draws Made (d)")
    ax.set_ylabel("FPC value")
    ax.set_ylim(0, 1)
    ax.text(2, 0.85, "FPC → 0 means variance collapses\n"
            "(settlement nearly certain)", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig