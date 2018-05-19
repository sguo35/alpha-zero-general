import numpy as np
from Game import Game as Game
class CambiaGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # the last 4 moves
        # 8 possible cards, 4 for each player, plus one for each player to represent the card they draw
        # 53 channels, one-hot representation for 52 card deck
        # 56 channels here because we convert to 55 channels before predicting - last 2 channels holds whether card is known by player 1 and player 2
        # use 2x2 convolutions to get history
        newBoard = np.zeros(shape=(4, 10, 56))
        for i in range(4):
            # add p1 cards
            # set starting cards
            newDraw = np.random.randint(low=0, high=54)
            newBoard[0][i] = np.zeros(shape=(56))
            newBoard[0][i][newDraw] = 1.
            if i < 2:
                # give vision
                newBoard[0][i][54] = 1.

            # add p2 cards
            newDraw = np.random.randint(low=0, high=54)
            newBoard[0][i + 5] = np.zeros(shape=(56))
            newBoard[0][i + 5][newDraw] = 1.
            if i < 2:
                # give vision
                newBoard[0][i + 5][55] = 1.

        # p1 starts with draw
        newDraw = np.random.randint(low=0, high=54)
        newBoard[0][4] = np.zeros(shape=(56))
        newBoard[0][4][newDraw] = 1.
        # give vision
        newBoard[0][4][54] = 1.

        return newBoard

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
        newBoard = np.delete(board, 3, axis=0)
        # insert a new move that's identical to previous move
        # handle playing own card
        a = np.copy(newBoard[0])
        newBoard = np.insert(newBoard, 0, a, axis=0)
        cardPlayed=0
        if action < 4:
            # play own card and swap with drawn card

            # get own card
            a = np.where(np.isin(newBoard[0][action + startIndex], [1.]))[0]
            if a.size > 0:
                cardToPlayIndex = a[0]
            # get unknown status - index 51 is player1, 52 is player2
            isCardKnown = newBoard[0][action + startIndex][54]
            if player == -1:
                isCardKnown = newBoard[0][action + startIndex][55]

            isCardKnownOpponent = newBoard[0][action + startIndex][54]
            if player == -1:
                isCardKnownOpponent = newBoard[0][action + startIndex][55]

            # our card
            # if card is known play it, otherwise get a random card to play
            # we assume a deck composed of infinite decks so we can uniformly draw
            if isCardKnown == 1.:
                cardPlayed = cardToPlayIndex
            else:
                cardPlayed = np.random.randint(low=0, high=54)

            # replace the current card with the drawn card and remove the drawn card from the board
            # replace played card slot with the drawn card
            newBoard[0][action + startIndex] = np.copy(newBoard[0][4 + startIndex])
            # remove card from drawn card slot
            newBoard[0][4 + startIndex] = np.zeros(shape=(56))
            # remove opponent's vision of new card
            temp = 0
            if player == -1:
                temp = 1
            newBoard[0][action + startIndex][54 + temp] = 0.
          
        if action == 4:
            # play drawn card
            s = np.where(np.isin(newBoard[0][4 + startIndex], [1.]))[0]
            if s.size > 0:
                cardPlayed = s[0]
            # set drawn card slot to nothing
            newBoard[0][4 + startIndex] = np.zeros(shape=(56))


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
            toLookAtRandom = []
            for i in range(4):
                if newBoard[0][i + startIndex][54 + temp] == 0.:
                    toLookAtRandom.append(i)
            if len(toLookAtRandom) < 1:
                toLookAtRandom.append(np.random.randint(low=0, high=5))
            rand = np.random.randint(low=0, high=len(toLookAtRandom))
            newBoard[0][toLookAtRandom[rand]][54 + temp] = 1.
        if cardPlayed == 9 or cardPlayed == 10:
            # look at one of opponent's cards that we don't know
            toLookAtRandom = []
            if player == 1:
                s = 5
            else:
                # player2
                s = 0
            for i in range(4):
                if newBoard[0][i + s][54 + temp] == 0.:
                    toLookAtRandom.append(i)
            if len(toLookAtRandom) < 1:
                toLookAtRandom.append(np.random.randint(low=0, high=5))
            rand = np.random.randint(low=0, high=len(toLookAtRandom))
            newBoard[0][toLookAtRandom[rand]][54 + temp] = 1.
        if cardPlayed == 11 or cardPlayed == 12:
            # blind swap with opponent's cards - choose one of our unknown and one of opponent's unknown
            myUnknown = 0
            toLookAtRandom = []
            for i in range(4):
                if newBoard[0][i + startIndex][54 + temp] == 0.:
                    toLookAtRandom.append(i)
            # if there's no unknown, randomly append one
            if len(toLookAtRandom) < 1:
                toLookAtRandom.append(np.random.randint(low=0, high=5))
            rand = np.random.randint(low=0, high=len(toLookAtRandom))
            myUnknown = toLookAtRandom[rand]

            oppUnknown = 5
            toLookAtRandom = []
            if player == 1:
                s = 5
            else:
                # player2
                s = 0
            for i in range(4):
                if newBoard[0][i + s][54 + temp] == 0.:
                    toLookAtRandom.append(i)
            if len(toLookAtRandom) < 1:
                toLookAtRandom.append(np.random.randint(low=0, high=5))
            rand = np.random.randint(low=0, high=len(toLookAtRandom))
            oppUnknown = toLookAtRandom[rand]

            # swap cards
            oppCard = np.copy(newBoard[0][oppUnknown])
            newBoard[0][oppUnknown] = np.copy(newBoard[0][myUnknown])
            newBoard[0][myUnknown] = np.copy(oppCard)

        if cardPlayed == 13:
          # King; check if it's red, if not then we look at an opponent's card and swap
          if not redKing:
            # swap one of our unknown with lowest or random opponent's card
            # blind swap with opponent's cards - choose one of our unknown and one of opponent's unknown
            myUnknown = 0
            toLookAtRandom = []
            for i in range(4):
                if newBoard[0][i + startIndex][54 + temp] == 0.:
                    toLookAtRandom.append(i)
            if len(toLookAtRandom) < 1:
                toLookAtRandom.append(np.random.randint(low=0, high=5))
            rand = np.random.randint(low=0, high=len(toLookAtRandom))
            myUnknown = toLookAtRandom[rand]

            oppUnknown = 5
            toLookAtRandom = []
            if player == 1:
                s = 5
            else:
                # player2
                s = 0
            for i in range(4):
                if newBoard[0][i + s][54 + temp] == 0.:
                    toLookAtRandom.append(i)
            if len(toLookAtRandom) < 1:
                toLookAtRandom.append(np.random.randint(low=0, high=5))
            rand = np.random.randint(low=0, high=len(toLookAtRandom))
            oppUnknown = toLookAtRandom[rand]

            # swap cards
            oppCard = np.copy(newBoard[0][oppUnknown])
            oppCard[54 + temp] = 1.
            newBoard[0][oppUnknown] = np.copy(newBoard[0][myUnknown])
            newBoard[0][myUnknown] = np.copy(oppCard)


        # play all identical cards
        for j in range(10):
            # loop through and look for cards that are similar
            for index in np.where(np.isin(newBoard[0][j], [1.]))[0]:
                # don't play red kings or jokers
                if index != 38 and index != 51 and index < 52:
                    # if the card is similar to current card, play it
                    c = index + 1
                    c %= 13
                    if c == cardPlayed:
                        newBoard[0][j] = np.zeros(shape=(56))

        # then draw a card for the opponent and switch turns
        # swap startIndex and temp
        if player == 1:
            startIndex = 5
        else:
            # player2
            startIndex = 0

        temp = 1
        if player == -1:
            temp = 0
        
        # set opponent's draw
        newDraw = np.random.randint(low=0, high=54)
        newBoard[0][4 + startIndex] = np.zeros(shape=(56))
        newBoard[0][4 + startIndex][newDraw] = 1.
        # give vision to opponent only
        newBoard[0][4 + startIndex][54 + temp] = 1.

        # update turn count
        newBoard[2][0][0] = newBoard[3][0][0] + 1
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
        outArr = np.zeros(shape=(5), dtype='int')
        if player == 1:
            startIndex = 0
        else:
            # player2
            startIndex = 5
        for i in range(4):
            if np.where(np.isin(board[0][i + startIndex], [1.]))[0].size > 0 and np.where(np.isin(board[0][i + startIndex], [1.]))[0][0] < 54:
                outArr[i] = 1
        outArr[4] = 1
        #print(outArr)
        return outArr

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        # if player1 or player2 is 1 or lower, game is over
        player1, player2 = self.computeScore(board)
        if player1 <= 1:
            return 1
        if player2 <= 1:
            return -1
        if board[2][0][0] > 52:
            # else, game ends after ~54 turns
            if player1 > player2:
                return -1
            else:
                return 1
        return 0

    
    def computeScore(self, board):
        """
        Input:
            board: current board
        Returns:
            score1: player1's score
            score2: player2's score
        """
        player1 = 0
        player2 = 0

        startIndex = 0
        for i in range(4):
            s = np.where(np.isin(board[0][i + startIndex], [1.]))[0]
            if s.size > 0:
                player1 += self.getCardScore(s[0])

        startIndex = 5
        for i in range(4):
            s = np.where(np.isin(board[0][i + startIndex], [1.]))[0]
            if s.size > 0:
                player2 += self.getCardScore(s[0])
        
        return player1, player2

    def getCardScore(self, num):
        """
        Input:
            num: an index of a card from 0-53
        Returns:
            value: the card's value
        """
        if num == 38 or num == 51:
            return -1
        if num == 52 or num == 53:
            return 0
        num += 1
        num %= 13
        return num


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
        newArr = np.zeros(shape=(4, 10, 56))
        # swap 0-4 with 5-9
        for i in range(4):
            for j in range(5):
                newArr[i][j + 5] = board[i][j]
        for i in range(4):
            for j in range(5):
                newArr[i][j] = board[i][j + 5]

        # swap vision columns
        for i in range(4):
            for j in range(10):
                temp = newArr[i][j][54]
                newArr[i][j][54] = newArr[i][j][55]
                newArr[i][j][55] = temp
        
        return newArr

    def unknownize(self, board, player):
        """
        Input:
            board: current full board
            player: current player (1 or -1)
        Returns:
            unknownizedBoard: returns board with unknown spots blanked. (4x12x55) shape
        """
        index = 55
        if player == -1:
            index = 54
        newBoard = np.delete(board, index, axis=2)
        #print(newBoard)
        #print(newBoard.shape)
        for i in range(newBoard.shape[0]):
            for j in range(newBoard[i].shape[0]):
                #print(newBoard.size)
                s = np.where(np.isin(board[i][j], [1.]))
                known = False
                #print(s[0])
                for indice in range(s[0].size):
                    #print(indice)
                    if indice == 54:
                        # if it's known, don't black card out
                        known = True
                if not known and s[0].size > 0:
                    #print(s[0])
                    if s[0][0] > 54:
                        s[0][0] = 54
                    newBoard[i][j][s[0][0]] = 0.
        return newBoard


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
        # We don't need to input symmetries because not having them doesn't prevent convergence
        # See AlphaZero Reddit AMA
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return np.array_str(board)
