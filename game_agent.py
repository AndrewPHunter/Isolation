"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import itertools

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    def onWall(move):
        return (move[0] == 0 or move[0] == (game.height-1) or move[1] ==0 or move[1] == (game.width-1))

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    def centerDistance():
        w, h = game.width / 2., game.height / 2.
        y, x = game.get_player_location(player)
        return float((h - y) ** 2 + (w - x) ** 2)



    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    open_spaces = len(game.get_blank_spaces())

    own_moves = len(player_moves)
    own_in_wall = len([move for move in player_moves if onWall(move)])
    opp_moves = len(opponent_moves)
    opp_in_wall = len([move for move in opponent_moves if onWall(move)])


    return float(own_moves * (1 + 2 / open_spaces) - opp_moves * (2 - 1 / open_spaces)  - own_in_wall * (1+1/open_spaces) + opp_in_wall * (1 + 1/open_spaces) )


def custom_score_2(game, player):
    """

    Chasing the opponent based heuristic

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    def playerDistance():
        y_opp, x_opp = game.get_player_location(game.get_opponent(player))
        y, x = game.get_player_location(player)
        return float((y-y_opp) ** 2 + (x-x_opp) ** 2)

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))


    return float(own_moves - playerDistance())


def custom_score_3(game, player):
    """

    This hueristic rewards maximizing player moves left at end game

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    def onWall(move):
        return (move[0] == 0 or move[0] == (game.height-1) or move[1] ==0 or move[1] == (game.width-1))

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    def centerDistance():
        w, h = game.width / 2., game.height / 2.
        y, x = game.get_player_location(player)
        return float((h - y) ** 2 + (w - x) ** 2)



    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    own_moves = len(player_moves)
    own_in_wall = len([move for move in player_moves if onWall(move)])
    opp_moves = len(opponent_moves)
    opp_in_wall = len([move for move in opponent_moves if onWall(move)])

    return float(own_moves + centerDistance() + opp_in_wall - 2 * opp_moves - own_in_wall)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """
        Depth limited minimax agent
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

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

    def terminal(self, game):
        """

        Parameters
        -----------
          game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        Returns
        -------
        bool
            True if at terminal node
            False if not at terminal node
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return not bool(game.get_legal_moves())


class AlphaBetaPlayer(IsolationPlayer):
    """
        Iterative deepening minimax search with alpha-beta pruning.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
                result before the time limit expires.

                Modify the get_move() method from the MinimaxPlayer class to implement
                iterative deepening search instead of fixed-depth search.

                **********************************************************************
                NOTE: If time_left() < 0 when this function returns, the agent will
                      forfeit the game due to timeout. You must return _before_ the
                      timer reaches 0.
                **********************************************************************

                Parameters
                ----------
                game : `isolation.Board`
                    An instance of `isolation.Board` encoding the current state of the
                    game (e.g., player locations and blocked cells).

                time_left : callable
                    A function that returns the number of milliseconds left in the
                    current turn. Returning with any less than 0 ms remaining forfeits
                    the game. (EDIT: This will be reported as timeout and not forfeit)

                Returns
                -------
                (int, int)
                    Board coordinates corresponding to a legal move; may return
                    (-1, -1) if there are no available legal moves.
                """
        self.time_left = time_left

        best_move = (-1,-1)

        try:
            for depth in itertools.count(1):
                move = self.alphabeta(game, depth)
                if move == (-1,-1):
                    break
                best_move = move

        except SearchTimeout:
            pass

        # Return the best move from the last completed search iteration
        return best_move

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

    def terminal(self, game):
        """
        Parameters
        -----------
            game : isolation.Board
                   An instance of the Isolation game `Board` class representing the
                   current game state

        Returns
        -------
            bool
                  True if at terminal node
                  False if not at terminal node
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return not bool(game.get_legal_moves())



