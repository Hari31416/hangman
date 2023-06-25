import random
from tqdm.auto import tqdm


class HangmanGame:
    """The class provides a simple CLI interface for the Hangman game.
    It also supports giving a custom guessing algorithm to play the game as well as playing multiple games.


    Attributes
    ----------
    words_location : str
        The location of the file that contains the list of words
    words : list
        The list of words that are used in the game
    TRIES : int
        The number of incorrect guesses that are allowed
    guesses : int
        The number of incorrect guesses that are made
    guessed_letters : list
        The list of letters that are guessed
    """

    def __init__(self, words_location="words_train.txt", tries=6):
        self.words_location = words_location
        self.words = open(words_location, "r").read().split("\n")
        self.words = [word for word in self.words if len(word) > 0]
        self.TRIES = tries

    def check_win(self, current_hint):
        """Checks if the user has won the game

        Parameters
        ----------
        current_hint : str
            The word that is displayed to the user (with the letters that are not guessed hidden)

        Returns
        -------
        bool
            True if the user has won the game, False otherwise
        """
        if current_hint.find("_ ") == -1:
            return True
        return False

    def display_word(self, word, letter_to_show=[]):
        """Displays the word with the letters that are not guessed hidden

        Parameters
        ----------
        word : str
            The word to be displayed
        letter_to_show : list
            The list of letters that are not hidden

        Returns
        -------
        str
            The word with the letters that are not guessed hidden
        """
        letters_to_hide = [l for l in word if l not in letter_to_show]
        result = word
        for l in letters_to_hide:
            result = result.replace(l, "_ ")
        return result

    def start_game(
        self,
        guessing_algorithm=None,
        show_hint=True,
        show_word=False,
        verbose=1,
    ):
        """Chooses a random word from the list of words and starts the game

        Parameters
        ----------
        guessing_algorithm : function
            The function that is used to guess the letters. If None, the user will be asked to input the letters
            The function should take as input the HangmanGame object and return the letter that is guessed
        show_hint : bool
            If True, the hint will be displayed
        show_word : bool
            If True, the word will be displayed
        verbose : int
            If 0, nothing will be printed. If 1, the game will be printed

        Returns
        -------
        bool
            True if the user has won the game, False otherwise
        """
        random_word = random.choice(self.words)
        if show_word and verbose:
            print(random_word)
        if show_hint and verbose:
            print("The current hint is: ", self.display_word(random_word))
        self.guesses = 0
        self.guessed_letters = []
        while self.guesses < self.TRIES:
            if guessing_algorithm is not None:
                cur_letter = guessing_algorithm(self)
            else:
                cur_letter = input("Input the letter ").strip()
            if verbose:
                print(f"You guessed the letter {cur_letter}")
            if len(cur_letter) > 1:
                print("You can only input one letter")
                continue
            self.guessed_letters.append(cur_letter)

            if cur_letter in random_word:
                self.guesses -= 1
            current_hint = self.display_word(random_word, self.guessed_letters)
            if verbose:
                print(f"Number of incorrect guesses: {self.guesses+1}")
            if show_hint and verbose:
                print(f"\nThe current hint is: {current_hint}")
            if self.check_win(current_hint=current_hint):
                if verbose:
                    print("You Won")
                return True
            self.guesses += 1
        if verbose:
            print("You lose")
        return False

    def play_multiple_games(self, num_games, guessing_algorithm=None, **kwargs):
        """Plays multiple games of Hangman

        Parameters
        ----------
        num_games : int
            The number of games to be played
        guessing_algorithm : function
            The function that is used to guess the letters. If None, the user will be asked to input the letters
            The function should take as input the HangmanGame object and return the letter that is guessed
        **kwargs : dict
            The arguments that are passed to the `start_game` function
        Returns
        -------
        int
            The number of games won
        """
        num_wins = 0
        for i in tqdm(range(num_games)):
            if self.start_game(guessing_algorithm=guessing_algorithm, **kwargs):
                num_wins += 1
        score = num_wins / num_games
        print(f"Score: {score}")
        return num_wins


if __name__ == "__main__":
    hangman = HangmanGame()
    hangman.start_game()
