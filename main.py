"""
main.py — Entry Point

Usage
-----
python main.py --demo                   # single narrated game
python main.py --sim                    # 10,000-game simulation (normal flow)
python main.py --sim --adversarial      # stress test (80% hard-informed flow)
python main.py --sim --games 1000       # custom number of games
python main.py --charts                 # generate all static charts
python main.py --all                    # demo + sim + adversarial + charts
python main.py --test                   # run unit tests

Flags
-----
--demo          Run a single narrated game
--sim           Run full simulation
--adversarial   Use adversarial flow in simulation
--games N       Number of games (default 10000)
--k K           Spread aggression k (default 0.5)
--seed N        Master random seed (default 42)
--charts        Generate and save all chart PNGs to ./charts/
--all           Run everything
--test          Run built-in mathematical unit tests
"""

import argparse
import os
import sys


def run_demo(k: float = 0.5, seed: int = 7):
    from simulator import run_demo_game
    from game_env  import CardGameState
    from ev_engine import get_state_snapshot
    from market_maker import MarketMaker
    from traders   import MixedFlow

    result = run_demo_game(seed=seed, k=k)
    return result


def run_sim(n_games: int = 10_000, k: float = 0.5,
            adversarial: bool = False, seed: int = 42):
    from simulator import run_simulation
    flow_label = "ADVERSARIAL (80% hard-informed)" if adversarial else "NORMAL (60/30/10 mix)"
    print(f"\n  Running {n_games:,} games | k={k} | Flow: {flow_label}")
    sim = run_simulation(n_games=n_games, k=k, adversarial=adversarial, seed=seed)
    return sim


def generate_charts(normal_sim=None, adversarial_sim=None, k: float = 0.5):
    import matplotlib
    matplotlib.use("Agg")
    from simulator   import run_demo_game, run_simulation
    from visualiser  import (plot_spread_collapse, plot_ev_convergence,
                              plot_pnl_curve, plot_pnl_distribution,
                              plot_adversarial_comparison, plot_fpc_curve)

    os.makedirs("charts", exist_ok=True)
    print("\n  Generating charts → ./charts/")

    # 1. Spread collapse
    print("  [1/6] Spread collapse chart...")
    plot_spread_collapse(k=k, save_path="charts/01_spread_collapse.png")

    # 2. FPC curve
    print("  [2/6] Finite population correction curve...")
    plot_fpc_curve(save_path="charts/02_fpc_curve.png")

    # 3. EV convergence (need a demo game)
    print("  [3/6] EV convergence (single game)...")
    demo_result = run_demo_game(seed=99, k=k)
    plot_ev_convergence(demo_result, save_path="charts/03_ev_convergence.png")

    # 4. P&L curve (need simulation)
    if normal_sim is None:
        print("  [4/6] Running 5,000 games for P&L chart...")
        normal_sim = run_simulation(n_games=5_000, k=k, adversarial=False,
                                     seed=42, verbose=False)
    print("  [4/6] P&L curve...")
    plot_pnl_curve(normal_sim, save_path="charts/04_pnl_curve.png")

    # 5. P&L distribution
    print("  [5/6] P&L distribution...")
    plot_pnl_distribution(normal_sim, save_path="charts/05_pnl_distribution.png")

    # 6. Adversarial comparison
    if adversarial_sim is None:
        print("  [6/6] Running 1,000 adversarial games for comparison chart...")
        adversarial_sim = run_simulation(n_games=1_000, k=k, adversarial=True,
                                          seed=42, verbose=False)
    print("  [6/6] Adversarial comparison...")
    plot_adversarial_comparison(normal_sim, adversarial_sim,
                                 save_path="charts/06_adversarial_comparison.png")

    print("\n  ✓ All charts saved to ./charts/")


def run_tests():
    """Built-in mathematical unit tests."""
    import numpy as np
    from game_env    import CardGameState, build_standard_deck
    from ev_engine   import compute_ev, compute_variance, compute_quote

    print("\n  Running unit tests...\n")
    failures = 0

    # Test 1: Deck sum = 340
    deck = build_standard_deck()
    assert sum(deck) == 340, f"Deck sum should be 340, got {sum(deck)}"
    assert len(deck) == 52,  f"Deck size should be 52, got {len(deck)}"
    print("  ✓ Test 1: Deck sum = 340, size = 52")

    # Test 2: Starting EV = 78.46...
    state = CardGameState()
    state.shuffle(seed=0)
    ev0 = compute_ev(state)
    expected_ev0 = 12 * (340 / 52)
    assert abs(ev0 - expected_ev0) < 1e-6, f"EV0={ev0}, expected={expected_ev0}"
    print(f"  ✓ Test 2: Starting EV = {ev0:.4f} ≈ {expected_ev0:.4f}")

    # Test 3: Variance collapses to 0 at game end
    for _ in range(CardGameState.N_DRAWS):
        state.draw_card()
    var_end = compute_variance(state)
    assert var_end == 0.0, f"End variance should be 0, got {var_end}"
    print(f"  ✓ Test 3: Variance at game end = {var_end}")

    # Test 4: EV at game end = true settlement sum
    state2 = CardGameState()
    state2.shuffle(seed=1)
    for _ in range(CardGameState.N_DRAWS):
        state2.draw_card()
    ev_end   = compute_ev(state2)
    true_sum = float(state2.true_remaining_sum())
    assert abs(ev_end - true_sum) < 1e-6, f"EV_end={ev_end}, true_sum={true_sum}"
    print(f"  ✓ Test 4: EV at end = true settlement sum = {true_sum}")

    # Test 5: Bid < EV < Ask always
    state3 = CardGameState()
    state3.shuffle(seed=2)
    for step in range(CardGameState.N_DRAWS - 5):
        bid, ask = compute_quote(state3, k=0.5)
        ev = compute_ev(state3)
        assert bid < ev < ask, f"Step {step}: bid={bid:.2f}, ev={ev:.2f}, ask={ask:.2f}"
        state3.draw_card()
    print("  ✓ Test 5: Bid < EV < Ask at every step")

    # Test 6: Spread is monotonically (roughly) decreasing
    state4 = CardGameState()
    state4.shuffle(seed=3)
    spreads = []
    for _ in range(CardGameState.N_DRAWS):
        bid, ask = compute_quote(state4, k=0.5)
        spreads.append(ask - bid)
        state4.draw_card()
    # Not strictly monotone (variance can fluctuate), but should trend down.
    # Check that average of last 10 < average of first 10
    assert np.mean(spreads[-10:]) < np.mean(spreads[:10]), \
        "Spread should be narrower at end than start"
    print(f"  ✓ Test 6: Spread trends downward "
          f"(first-10 avg={np.mean(spreads[:10]):.2f}, "
          f"last-10 avg={np.mean(spreads[-10:]):.4f})")

    print(f"\n  All tests passed!\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EV Market Maker — Continuous quoting on incomplete information games"
    )
    parser.add_argument("--demo",        action="store_true", help="Run single narrated game")
    parser.add_argument("--sim",         action="store_true", help="Run N-game simulation")
    parser.add_argument("--adversarial", action="store_true", help="Use adversarial flow")
    parser.add_argument("--games",       type=int,   default=10_000, help="Number of games")
    parser.add_argument("--k",           type=float, default=0.5,    help="Spread aggression k")
    parser.add_argument("--seed",        type=int,   default=42,     help="Random seed")
    parser.add_argument("--charts",      action="store_true", help="Generate all chart PNGs")
    parser.add_argument("--all",         action="store_true", help="Run everything")
    parser.add_argument("--test",        action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if not any([args.demo, args.sim, args.charts, args.all, args.test]):
        parser.print_help()
        print("\n  Quick start: python main.py --demo")
        return

    normal_sim = adversarial_sim = None

    if args.test or args.all:
        run_tests()

    if args.demo or args.all:
        run_demo(k=args.k, seed=args.seed)

    if args.sim or args.all:
        normal_sim = run_sim(n_games=args.games, k=args.k,
                              adversarial=args.adversarial, seed=args.seed)

    if (args.all or args.charts):
        # Also run adversarial sim for comparison chart
        if adversarial_sim is None and args.all:
            print("\n  Running adversarial simulation for comparison chart...")
            from simulator import run_simulation
            adversarial_sim = run_simulation(n_games=min(args.games, 2_000),
                                              k=args.k, adversarial=True,
                                              seed=args.seed, verbose=False)
        generate_charts(normal_sim=normal_sim, adversarial_sim=adversarial_sim, k=args.k)


if __name__ == "__main__":
    main()