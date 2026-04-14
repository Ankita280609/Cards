# Continuous EV Market Maker

An automated trading agent that quotes a live, two-way market on an incomplete information card game. 

This project simulates a real-world market-making environment where information is revealed sequentially. As each card is drawn, the agent recalculates the conditional Expected Value (EV) and tightens its bid-ask spread in proportion to the collapsing variance. 

## 📖 Project Overview

The core objective of the agent is to quote a bid and an ask around a fair value estimate, profit from noise traders paying the spread, and defend against informed traders. The game uses a standard 52-card deck where the contract settlement value is the sum of the 12 cards remaining after 40 draws.

* **Start of Game:** High variance; spread is wide.
* **End of Game:** Variance collapses to zero; spread becomes razor-thin.

##  The Mathematics

The engine runs on conditional probability and hypergeometric sampling (drawing without replacement).

* **Expected Value:** `E[settlement_sum | cards_seen] = n_settle * mean(remaining_pool)`
  
* **Variance (Hypergeometric):** `Var[settlement_sum] = n_settle * population_variance * (d / (R-1))`
  *(The finite population correction factor ensures variance hits exactly zero when all draws are complete).*

* **Spread Width:** `Spread = k * sqrt(Var[settlement_sum])`  
  *(where `k` is the aggression parameter controlling how tight the market maker quotes).*

##  Software Architecture

The project is structured into modular Python files with clear responsibilities:

* `game_env.py`: Manages deck state, card draws, and ground-truth settlement.
* `ev_engine.py`: The math engine computing hypergeometric EV and variance.
* `market_maker.py`: The quoting agent featuring inventory skew, informed flow detection (spread widening on consecutive hits), and hard inventory limits.
* `traders.py`: Simulated counterparties including Noise Traders, Soft-Informed Traders, and Hard-Informed (Adversarial) Traders.
* `simulator.py`: Manages the game loop and runs multi-game Monte Carlo simulations.
* `visualiser.py`: Generates spread collapse charts, P&L curves, and drawdown distributions.

##  Counterparty Models & Risk Management

To test the agent's robustness, it trades against mixed flow:
1. **Noise Traders:** Trade randomly, paying the spread (the primary revenue source).
2. **Soft-Informed Traders:** Trade when the agent's mid-price drifts from the true EV.
3. **Hard-Informed Traders (Adversarial):** Peek at the next card and trade if the EV shift overcomes the spread.

**Risk Defenses:**
* **Inventory Skew:** Shifts quotes downward when long (and upward when short) to attract unwinding trades.
* **Informed Flow Detection:** Exponentially widens the spread after 3+ consecutive same-side hits to choke off toxic flow.

##  Tech Stack
* **Python 3.11** (Core Logic)
* **NumPy / SciPy** (Vectorised math and statistical cross-checks)
* **Matplotlib** (Data visualisation and charting)
* **pytest** (Unit testing for the EV engine)
