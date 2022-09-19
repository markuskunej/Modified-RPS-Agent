from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    YOUR DOCUMENTATION GOES HERE!
    My agent implements the idea of a discrete Markov Chain. The idea is to predict the opponents next move based 
    on the current state, and then select its own move which maximizes the expected payoff. In context with 
    rock, paper, scissors (or similar type games), I represent the current state of the markov chain as the pair 
    of moves from last round. So for R,P,S, there would be 9 possible combinations of moves. From each pair, 
    there are then have 3 possible 'outputs' (aka the moves the opponent can make). Each of these has a
    probability of occuring, with them summing to 1. This structure is represented as self.matrix in the class.
    There is also a n_obs value for each output, which keeps the number of past occurences (used to calculate the probability)
    The self.decay parameter represents the "memory" of the model. A value of 1 perfectly remembers
    all past occurences, while a number between 0 and 1 would forget earlier observations. This can come
    in handy as it allows the agent to adapt to opponents changes faster. I decided to set the decay to 0.8, partly because
    it was acheiving some of my highest autolab marks, but also because it allows the Markov Model to have a fairly good memory,
    but not perfect, so it can't be targeted by purposely anti-Markov/constantly changing agents. 
    The model updates the matrix after each move,
    and then when it comes time to determine what move to do, uses the probabilities of the opponents possible moves
    and the game (payoff) matrix to determine the expected payoff for each possible move, choosing the max of these.
    Another tool I added is a class variable which counts the number of times in a row that the opponent has copied us.
    Any time it is over 10, I overright the make_move function to select the move that maximizes the payoff against our
    previous move. The Markov Model with decay was based on the one from the website below. 
    https://towardsdatascience.com/how-to-win-over-70-matches-in-rock-paper-scissors-3e17e67e0dab

    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        # YOUR CODE GOES HERE
        self.decay = 0.80
        # initialize markov chain state matrix, n_moves x n_moves x n_moves to represent
        # n_moves ^ 2 possible previous move pairs and n_moves probabilities + num of observations for the future
        # move for each of these. The n_moves possible states start at 1/n_moves (even)

        # n_moves probabilities and num of observations represent choosing R, P, or S
        probs = {'prob' : 1 / self.n_moves, 'n_obs' : 0}

        #fill a N_moves x N_moves matrix (represents prev moves) with the probs
        self.matrix = [[[probs.copy() for _ in range (self.n_moves)] for _ in range(self.n_moves)] for _ in range (self.n_moves)]

        # array to keep track of the last move [my move, oppenents move]
        self.prev_pair = []

        #boolean to see if facing against a copycat
        self.copied_in_a_row = 0

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        # Check if first move, then return a random choice
        if (len(self.prev_pair) == 0):
            return np.random.randint(0, self.n_moves)

        #check if facing against copycat
        if (self.copied_in_a_row > 10):
            # return the move that beats our previous move
            next_move = np.argmax(self.game_matrix[:,self.prev_pair[0]])
            return next_move

        
        # GET EXPECTED VALUE FOR PAYOFF FOR EACH POSSIBLE MOVE
        prev_me = self.prev_pair[0]
        prev_opp = self.prev_pair[1]

        # variables to keep track of best move
        highest_expected = 0
        best_move = 0

        #get expected payoff for each move choice
        for my_move in range(0, self.n_moves):
            expected = 0

            for opp_move in range(0, self.n_moves):
                # expected = probability * payoff
                expected += self.matrix[prev_me][prev_opp][opp_move]['prob'] * self.game_matrix[my_move][opp_move]

            if expected > highest_expected:
                #update best move and highest expected value
                best_move = my_move
                highest_expected = expected


        return best_move

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # YOUR CODE GOES HERE

        #skip if this was the first round (prev_pair not established)
        if (len(self.prev_pair) == 0):
            self.prev_pair = [my_move, other_move]
            return

        prev_me = self.prev_pair[0]
        prev_opp = self.prev_pair[1]

        #check if opponent copied us, and update counter
        if prev_me == other_move:
            self.copied_in_a_row = self.copied_in_a_row + 1
        else:
            self.copied_in_a_row = 0

        ## Update Matrix using the previous pair and then opponents resulting move

        #update n_obs for prev_pair based on decay value
        for i in range(0, self.n_moves):
            self.matrix[prev_me][prev_opp][i]['n_obs'] = self.decay * self.matrix[prev_me][prev_opp][i]['n_obs']
        
        #add 1 to n_obs for the opponents resulting move
        self.matrix[prev_me][prev_opp][other_move]['n_obs'] = self.matrix[prev_me][prev_opp][other_move]['n_obs'] + 1

        #get total number of n_obs for the prev pair
        n_total = 0
        for i in range(0, self.n_moves):
            n_total += self.matrix[prev_me][prev_opp][i]['n_obs']

        #update probabilities for the pair
        for i in range(0, self.n_moves):
            self.matrix[prev_me][prev_opp][i]['prob'] = self.matrix[prev_me][prev_opp][i]['n_obs'] / n_total
        
        #update previous pair class variable
        self.prev_pair = [my_move, other_move]

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        ## SET MARKOV CHAIN MATRIX BACK TO DEFAULT
        probs = {'prob' : 1 / self.n_moves, 'n_obs' : 0}

        #fill a n_moves x n_moves matrix (represents prev moves) with the probs
        self.matrix = [[[probs.copy() for _ in range (self.n_moves)] for _ in range(self.n_moves)] for _ in range (self.n_moves)]

        # array to keep track of the last move [my move, oppenents move]
        self.prev_pair = []



if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -2.0, 0.5],
                            [2.0, 0.0, -1.0],
                            [-0.5, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    copycat_player = CopycatPlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, copycat_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))