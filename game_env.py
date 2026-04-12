"""
game_env.py — Game Environments
Manages the live state of card and dice games.
The agent only reads from 'drawn' — it never peeks at 'remaining'.
"""

import random
from typing import List, Optional


# ---------------------------------------------------------------------------
# Card value assignment
# ---------------------------------------------------------------------------

def build_standard_deck() -> List[int]:
    """
    52-card deck: Ace=1, 2-9=face value, 10/J/Q/K=10.
    Total deck sum = 340.
    """
    values = []
    for suit in range(4):                       # 4 suits
        values.append(1)                        # Ace
        for pip in range(2, 10):                # 2-9
            values.append(pip)
        values.extend([10, 10, 10, 10])         # 10, J, Q, K
    assert len(values) == 52
    assert sum(values) == 340
    return values


# ---------------------------------------------------------------------------
# Card Game State
# ---------------------------------------------------------------------------

class CardGameState:
    """
    Manages a standard 52-card game.

    Public attributes (visible to agent):
        drawn       : list of card values drawn so far
        n_drawn     : number of cards drawn

    Private attributes (ground truth, not shown to agent):
        remaining   : cards still in deck (ordered, next draw = remaining[0])
    """

    DECK_SIZE = 52
    N_DRAWS   = 40          # cards drawn before settlement
    N_SETTLE  = 12          # cards remaining at settlement = DECK_SIZE - N_DRAWS

    def __init__(self):
        self.remaining: List[int] = []
        self.drawn:     List[int] = []

    # ------------------------------------------------------------------
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Reset and shuffle the deck."""
        deck = build_standard_deck()
        rng  = random.Random(seed)
        rng.shuffle(deck)
        self.remaining = deck
        self.drawn     = []

    # ------------------------------------------------------------------
    def draw_card(self) -> int:
        """
        Draw the next card from the top of the deck.
        The drawn card is moved from remaining → drawn.
        Returns the card value (publicly revealed).
        """
        if not self.remaining:
            raise ValueError("No cards left to draw.")
        card = self.remaining.pop(0)
        self.drawn.append(card)
        return card

    # ------------------------------------------------------------------
    @property
    def n_drawn(self) -> int:
        return len(self.drawn)

    @property
    def draws_left(self) -> int:
        """Draws remaining before settlement."""
        return self.N_DRAWS - self.n_drawn

    @property
    def n_remaining(self) -> int:
        return len(self.remaining)

    # ------------------------------------------------------------------
    # Ground truth — NOT visible to the agent in real play
    # ------------------------------------------------------------------
    def true_remaining_sum(self) -> int:
        """True settlement sum (unknown to agent until game end)."""
        settle_cards = self.remaining[self.draws_left:]   # last N_SETTLE cards
        return sum(settle_cards)

    def true_ev(self) -> float:
        """
        True EV the agent *would* compute if it knew all remaining cards.
        Used only for plotting / validation.
        """
        return float(self.true_remaining_sum())

    # ------------------------------------------------------------------
    def is_game_over(self) -> bool:
        return self.n_drawn >= self.N_DRAWS

    def __repr__(self) -> str:
        return (f"CardGameState(drawn={self.n_drawn}, "
                f"remaining={self.n_remaining}, "
                f"draws_left={self.draws_left})")


# ---------------------------------------------------------------------------
# Dice Game State  (extension — independent draws, no hypergeometric FPC)
# ---------------------------------------------------------------------------

class DiceGameState:
    """
    n_rolls independent d6 rolls. Settlement = sum of all rolls.
    Draws are independent so Var = n_remaining * (35/12).
    Used as a contrast case vs the card game.
    """

    N_SIDES  = 6
    N_ROLLS  = 20
    EV_PER   = (1 + 6) / 2          # 3.5
    VAR_PER  = sum((x - 3.5)**2 for x in range(1, 7)) / 6   # 35/12 ≈ 2.9167

    def __init__(self):
        self.rolls:   List[int] = []
        self.n_total: int       = self.N_ROLLS

    def roll(self, seed: Optional[int] = None) -> int:
        rng  = random.Random(seed)
        val  = rng.randint(1, self.N_SIDES)
        self.rolls.append(val)
        return val

    @property
    def n_rolled(self) -> int:
        return len(self.rolls)

    @property
    def n_remaining(self) -> int:
        return self.n_total - self.n_rolled

    def is_game_over(self) -> bool:
        return self.n_rolled >= self.n_total

    def true_ev(self) -> float:
        return self.n_remaining * self.EV_PER

    def true_variance(self) -> float:
        return self.n_remaining * self.VAR_PER