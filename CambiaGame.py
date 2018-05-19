import Game from Game

import numpy as np

class CambiaGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
      pass
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # the last 4 moves
        # 8 possible cards, 4 for each player, plus one for each player to represent the card they draw
        # 53 channels, one-hot representation for 52 card deck + 1 unknown
        # 54 channels here because we convert to 53 channels before predicting - last 2 channels holds whether card is known by player 1 and player 2
        # use 2x2 convolutions to get history
        return np.zeros(shape=(4, 10, 54))

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (4, 10)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # You can only play one of 4 cards, or the card you draw = 5 actions
        return 5

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        startIndex = 0
        # If player1
        if player == 1:
          startIndex = 0
        else:
          # player2
          startIndex = 5
        # delete the last move in history
        newBoard = np.delete(board, 3)
        # insert a new move that's identical to previous move
        newBoard = np.insert(np.copy(newBoard[0]), 0, axis=0))
        # handle playing own card
        cardPlayed = 0
        if action < 4:
          # play own card and swap with drawn card

          # get own card
          cardToPlayIndex = np.where(np.isin(newBoard[0][action + startIndex], [1.]))[0]
          # get unknown status - index 51 is player1, 52 is player2
          isCardKnown = newBoard[0][action + startIndex][51]
          if player == -1:
            isCardKnown = newBoard[0][action + startIndex][52]

          isCardKnownOpponent = newBoard[0][action + startIndex][52]
          if player == -1:
            isCardKnownOpponent = newBoard[0][action + startIndex][51]

          # our card
          # if card is known play it, otherwise get a random card to play
          # we assume a deck composed of infinite decks so we can uniformly draw
          if isCardKnown == 1.:
            cardPlayed = cardToPlayIndex
          else:
            cardPlayed = np.random(low=0, high=54)

          # replace the current card with the drawn card and remove the drawn card from the board
          # replace played card slot with the drawn card
          newBoard[0][action + startIndex] = np.copy(newBoard[0][4 + startIndex])
          # remove card from drawn card slot
          newBoard[0][4 + startIndex] = np.zeros(shape=(54,))
          # remove opponent's vision of new card
          temp = 0
          if player == -1:
            temp = 1
          newBoard[0][action + startIndex][52 - temp] = 0.
          
        if action == 4:
          # play drawn card
          cardPlayed = np.where(np.isin(newBoard[0][4 + startIndex], [1.]))[0]
          # set drawn card slot to nothing
          newBoard[0][4 + startIndex] = np.zeros(shape=(54,))


        temp = 0
          if player == -1:
            temp = 1
        # regardless of action, play out the effects of the played card
        redKing = False
        if cardPlayed == 38 or cardPlayed == 51:
          redKing = True
        cardPlayed += 1
        cardPlayed %= 13
        if cardPlayed == 7 or cardPlayed == 8:
          # look at one of your own unknown cards
        if cardPlayed == 9 or cardPlayed == 10:
          # look at one of opponent's cards that we don't know
        if cardPlayed == 11 or cardPlayed == 12:
          # blind swap with opponent's cards - choose one of our unknown and lowest known opponent's card, or random, swap
        if cardPlayed == 13:
          # King; check if it's red, if not then we look at an opponent's card and swap
          if !redKing:
            # swap one of our unknown with lowest or random opponent's card
        # then draw a card for the opponent and switch turns
        if player == 1:
          startIndex = 5
        else:
          # player2
          startIndex = 0
        # set opponent's draw
        newDraw = np.random(low=0, high=54)
        newBoard[0][4 + startIndex] = np.zeros(shape=(54,))
        newBoard[0][4 + startIndex][newDraw] = 1.
        # give vision to opponent only
        newBoard[0][4 + startIndex][52 - temp] = 1.
        return newBoard, player * -1


    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def unknownize(self, board, player):
      """
      Input:
        board: current full board
        player: current player (1 or -1)
      Returns:
        unknownizedBoard: returns board with unknown spots blanked. (4x12x53) shape, where 53rd channel is whether a card is known to the player.
      """


    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
