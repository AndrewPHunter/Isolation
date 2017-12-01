# Isolation game agent
[AI Nanodegree Project](https://www.udacity.com/ai)

![Example game of isolation](viz.gif)


<p>
This project was performed as part of the Udacity nanodegree program with the goal of producing a game playing agent for isolation.
Isolation is a deterministic, two-player game of perfect information in which the players alternate turns moving a single piece from one cell to another on a board.
Whenever either player occupies a cell, that cell becomes blocked for the remainder of the game.  The first player with no remaining legal moves loses, and the opponent is declared the winner.
This project uses a version of Isolation where each agent is restricted to L-shaped movements (like a knight in chess) on a rectangular grid (like a chess or checkerboard).
The agents can move to any open cell on the board that is 2-rows and 1-column or 2-columns and 1-row away from their current position on the board. Movements are blocked at the edges of the
board (the board does not wrap around), however, the player can "jump" blocked or occupied spaces (just like a knight in chess).
Additionally, agents will have a fixed time limit each turn to search for the best move and respond.  If the time limit expires during a player's turn, that player forfeits the match, and the
opponent wins.
</p>

The full Udacity project description can be found in the [docs](./docs.md)


### Getting Started
* clone the repo
* restore the project environment

```ss
conda env create -f environment.yml
```

-or -

```sh
pip install -r requirements.txt
```

* run the project

```sh
python solution.py
```

## Main algorithms employed
* [Minimax Algorithm](https://en.wikipedia.org/wiki/Minimax)

```py
    def minimax(self, game, depth):

        """
        Depth limited minimax agent. Uses internal recurse function to implement the recursive
        tree search

        PARAMETERS:
        -------------
            game : `isolation.Board`
                    An instance of `isolation.Board` encoding the current state of the
                    game (e.g., player locations and blocked cells).

            depth : int
                    The maximum search depth to evaluate down the game tree

        RETURNS:
        -------------
            move:   (int, int)
                    The optimum move based on minimax evaluation
        """

        def recurse(game, depth, maximizing=True):

            """
            Depth limited recursive minimax algorithm

            PARAMETERS:
            ----------------
                game : `isolation.Board`
                        An instance of `isolation.Board` encoding the current state of the
                        game (e.g., player locations and blocked cells).

                depth : int
                        The maximum search depth to evaluate down the game tree

                maximizing: bool
                            Indicates whether to evaluate the function from the player perspective (True)
                            or the opponent perspective (False)

            RETURNS:
            --------------
                score, move: float, (int, int)
                             The score from the heuristic function for the given next player position
            """

            max_player = max, float("-inf"), (-1,-1)
            min_player = min, float("inf"), (-1,-1)

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout

            if self.terminal(game) or depth == 0:
                return self.score(game, self), (-1,-1)

            evaluation, value, best_move = max_player if  maximizing else min_player

            for move in game.get_legal_moves():
                score, _ = recurse(game.forecast_move(move), depth-1, not maximizing)
                if evaluation(value, score) == score:
                    value, best_move = score, move

            return value, best_move

        #minimax call
        _, move = recurse(game, depth)

        return move
```

* [Minimax with Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)

```py
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        """
        Depth and time limited alpha beta minimax algorithm

        PARAMETERS:
        ---------------
            game : `isolation.Board`
                    An instance of `isolation.Board` encoding the current state of the
                    game (e.g., player locations and blocked cells).

            depth : int
                    The maximum search depth to evaluate down the game tree

            alpha : float
                    alpha bound for alpha-beta pruning

            beta :  float
                    beta bound for alpha-beta pruning

        RETURNS:
        ----------------
            move:   (int, int)
                    The optimum move based on alpha-beta enhanced minimax evaluation

        """

        def max_value(game, depth, alpha, beta):

            """
            Player perspective evaluation of alpha beta algorithm

            PARAMETERS:
            --------------------
                Same as parent function

            RETURNS:
            ------------------
                score, move: float, (int, int)
                             The score from the heuristic function for the given next player position
            """

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if self.terminal(game) or depth == 0:
                return self.score(game, self), (-1,-1)

            best = float("-inf"), (-1,-1)

            for move in game.get_legal_moves():
                score, _ = min_value(game.forecast_move(move), depth-1, alpha, beta)
                best = max(best, (score, move), key=lambda m: m[0])
                if best[0] >= beta:
                    return best
                alpha = max(alpha, best[0])

            return best


        def min_value(game, depth, alpha, beta):

            """
            Opponent perspective evaluation of alpha beta algorithm

            PARAMETERS:
            --------------------
                Same as parent function

            RETURNS:
            ------------------
                score, move: float, (int, int)
                             The score from the heuristic function for the given next player position
            """

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout

            if self.terminal(game) or depth == 0:
                return self.score(game, self), (-1,-1)

            best = float("inf"), (-1,-1)
            for move in game.get_legal_moves():
                score, _ = max_value(game.forecast_move(move), depth-1, alpha, beta)
                best = min(best, (score, move), key=lambda m: m[0])
                if best[0] <= alpha:
                    return best
                beta = min(beta, best[0])

            return best

        _, move = max_value(game, depth, alpha, beta)

        return move
```

### Contributing

This repository is used for a nanodegree program that I am participating in so I will not be accepting pull requests.