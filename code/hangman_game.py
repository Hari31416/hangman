import random


class HangmanGame:
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

    def start_game(self, guessing_algorithm=None):
        """Chooses a random word from the list of words and starts the game

        Parameters
        ----------
        guessing_algorithm : function
            The function that is used to guess the letters. If None, the user will be asked to input the letters
            The function should take as input the HangmanGame object and return the letter that is guessed

        Returns
        -------
        bool
            True if the user has won the game, False otherwise
        """
        random_word = random.choice(self.words)
        # print(random_word)
        print("The current hint is:")
        print(self.display_word(random_word))
        self.guesses = 0
        self.guessed_letters = []
        while self.guesses < self.TRIES:
            if guessing_algorithm is not None:
                cur_letter = guessing_algorithm(self)
            else:
                cur_letter = input("Input the letter ").strip()
            if len(cur_letter) > 1:
                print("You can only input one letter")
                continue
            if cur_letter in random_word:
                self.guessed_letters.append(cur_letter)
                self.guesses -= 1
            current_hint = self.display_word(random_word, self.guessed_letters)
            print(f"Number of incorrect guesses: {self.guesses+1}")
            print(f"The current hint is: {current_hint}")
            if self.check_win(current_hint=current_hint):
                print("You Won")
                return True
            self.guesses += 1
        print("You lose")
        return False


if __name__ == "__main__":
    hangman = HangmanGame()
    hangman.start_game()
