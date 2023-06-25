import collections
import random
import re
from itertools import product
import tqdm
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, KneserNeyInterpolated
from nltk.util import everygrams, flatten
import numpy as np
import pandas as pd

TRIES = 6


class HangmanModel:
    """Creates a hangman model that can be used to play hangman.
    The model uses ngrams along with frequency of letters in the dictionary to guess the next letter.
    """

    def __init__(self, dictionary_dir="words_train.txt") -> None:
        """Initializes the model.

        Parameters
        ----------
        dictionary_dir : str
            Path to the dictionary file.
        """

        self.dictionary_dir = dictionary_dir
        self.full_dictionary = self.build_dictionary()
        self.train_model()

        self.dictionary_og = self.train_data
        self.current_dictionary = []

        self.full_dictionary_common_letter_sorted = collections.Counter(
            "".join(self.full_dictionary)
        ).most_common()
        self.full_dictionary_common_letter_sorted = [
            a[0] for a in self.full_dictionary_common_letter_sorted
        ]

        # initializing the hyperparameters
        self.frequency_guess_percentage = 0.5
        self.w1 = 0.05
        self.w2 = 0.1
        self.w3 = 0.2
        self.w4 = 0.2
        self.w5 = 0.3

    def train_model(self, split=0.8):
        """Trains the model on the dictionary. Uses five grams and only a part of the
        dictionary to train the model."""
        len_dataset = len(self.full_dictionary)
        full_dictionary_shuffled = self.full_dictionary.copy()
        random.shuffle(full_dictionary_shuffled)
        train_dataset_length = int(len_dataset * split)
        self.train_data = full_dictionary_shuffled[:train_dataset_length]
        self.test_data = full_dictionary_shuffled[train_dataset_length:]

        n = 5
        train_dataset = (list(everygrams(word, max_len=n)) for word in self.train_data)
        vocabulary = flatten([list(w) for w in self.train_data])
        self.model = MLE(n)
        print("Training Model. This may take a while.")
        self.model.fit(train_dataset, vocabulary)
        print("Training Done!")

    def guess_unigram(self, current_hint, probabilities, letters_to_guess):
        """Use the unigram to guess the next letter

        Parameters
        ----------
        current_hint : str
            The current hint of the game. (e.g. "_ _ _ _ _ _ _ _ _ _")
        probabilities : np.array
            The probabilities of each letter being the next letter.
        letters_to_guess : list
            The list of letters that can be guessed.

        Returns
        -------
        np.array
            The updated probabilities of each letter being the next letter.
        """
        current_hint = current_hint.replace(" ", "")
        posterior = np.zeros(len(letters_to_guess))
        # run through the current hint and update the probability
        for i, char in enumerate(letters_to_guess):
            if char not in current_hint:
                posterior[i] = self.model.score(char)
        # update the probabilities and return
        return probabilities + self.w1 * posterior

    def guess_bigram(self, current_hint, probabilities, letters_to_guess):
        """Use the bigram to guess the next letter

        Parameters
        ----------
        current_hint : str
            The current hint of the game. (e.g. "_ _ _ _ _ _ _ _ _ _")
        probabilities : np.array
            The probabilities of each letter being the next letter.
        letters_to_guess : list
            The list of letters that can be guessed.

        Returns
        -------
        np.array
            The updated probabilities of each letter being the next letter.
        """
        current_hint = current_hint.replace(" ", "")
        first_empty_place = []
        second_empty_place = []

        # determine the first empty places (e.g. _ n)
        for i in range(1, len(current_hint)):
            if current_hint[i - 1] == "_" and current_hint[i] != "_":
                first_empty_place.append(i - 1)

        # determine the second empty places (e.g. a _)
        for i in range(len(current_hint) - 1):
            if current_hint[i + 1] == "_" and current_hint[i] != "_":
                second_empty_place.append(i + 1)

        # run through the current hint and update the probability
        posterior_1 = np.zeros(len(letters_to_guess))
        for i in first_empty_place:
            second_letter = current_hint[i + 1]
            for j, letter in enumerate(letters_to_guess):
                first_letter = letter
                posterior_1[j] = self.model.score(second_letter, [first_letter])

        posterior_2 = np.zeros(len(letters_to_guess))
        for i in second_empty_place:
            first_letter = current_hint[i - 1]
            for j, letter in enumerate(letters_to_guess):
                second_letter = letter
                posterior_2[j] = self.model.score(second_letter, [first_letter])

        # update the probability
        posterior = np.array([posterior_1, posterior_2])
        posterior = posterior.mean(axis=0)

        return probabilities + self.w2 * posterior

    def guess_trigram(self, current_hint, probabilities, letters_to_guess):
        """Use the trigram to guess the next letter

        Parameters
        ----------
        current_hint : str
            The current hint of the game. (e.g. "_ _ _ _ _ _ _ _ _ _")
        probabilities : np.array
            The probabilities of each letter being the next letter.
        letters_to_guess : list
            The list of letters that can be guessed.

        Returns
        -------
        np.array
            The updated probabilities of each letter being the next letter.
        """
        current_hint = current_hint.replace(" ", "")
        first_empty_place = []
        second_empty_place = []
        third_empty_place = []
        posterior = np.zeros(len(letters_to_guess))

        # same as bigram, but with 3 empty places
        for i in range(2, len(current_hint)):
            if (
                current_hint[i - 2] == "_"
                and current_hint[i - 1] != "_"
                and current_hint[i] != "_"
            ):
                first_empty_place.append(i - 2)

        for i in range(1, len(current_hint) - 1):
            if (
                current_hint[i - 1] != "_"
                and current_hint[i] == "_"
                and current_hint[i + 1] != "_"
            ):
                second_empty_place.append(i)

        for i in range(len(current_hint) - 2):
            if (
                current_hint[i] != "_"
                and current_hint[i + 1] != "_"
                and current_hint[i + 2] == "_"
            ):
                third_empty_place.append(i + 2)

        posterior_1 = np.zeros(len(letters_to_guess))
        for i in first_empty_place:
            second_letter = current_hint[i + 1]
            third_letter = current_hint[i + 2]

            for j, letter in enumerate(letters_to_guess):
                first_letter = letter
                try:
                    posterior_1[j] += self.model.score(
                        third_letter, [first_letter, second_letter]
                    )
                except ZeroDivisionError:
                    pass

        posterior_2 = np.zeros(len(letters_to_guess))
        for i in second_empty_place:
            first_letter = current_hint[i - 1]
            third_letter = current_hint[i + 1]

            for j, letter in enumerate(letters_to_guess):
                second_letter = letter
                try:
                    posterior_2[j] += self.model.score(
                        third_letter, [first_letter, second_letter]
                    )
                except ZeroDivisionError:
                    pass

        posterior_3 = np.zeros(len(letters_to_guess))
        for i in third_empty_place:
            first_letter = current_hint[i - 1]
            second_letter = current_hint[i - 2]

            for j, letter in enumerate(letters_to_guess):
                third_letter = letter
                try:
                    posterior_3[j] += self.model.score(
                        third_letter, [first_letter, second_letter]
                    )
                except ZeroDivisionError:
                    pass

        # update the probability
        posterior = np.array(
            [
                posterior_1,
                posterior_2,
                posterior_3,
            ]
        )
        posterior = posterior.mean(axis=0)
        return probabilities + self.w3 * posterior

    def guess_fourgram(self, current_hint, probabilities, letters_to_guess):
        """Use the fourgram to guess the next letter

        Parameters
        ----------
        current_hint : str
            The current hint of the game. (e.g. "_ _ _ _ _ _ _ _ _ _")
        probabilities : np.array
            The probabilities of each letter being the next letter.
        letters_to_guess : list
            The list of letters that can be guessed.

        Returns
        -------
        np.array
            The updated probabilities of each letter being the next letter.
        """
        # remove the spaces
        current_hint = current_hint.replace(" ", "")
        first_empty_place = []
        second_empty_place = []
        third_empty_place = []
        fourth_empty_place = []
        posterior = np.zeros(len(letters_to_guess))

        for i in range(3, len(current_hint)):
            if (
                current_hint[i - 3] == "_"
                and current_hint[i - 2] != "_"
                and current_hint[i - 1] != "_"
                and current_hint[i] != "_"
            ):
                first_empty_place.append(i - 3)

        for i in range(2, len(current_hint) - 1):
            if (
                current_hint[i - 2] != "_"
                and current_hint[i - 1] == "_"
                and current_hint[i] != "_"
                and current_hint[i + 1] != "_"
            ):
                second_empty_place.append(i - 1)

        for i in range(1, len(current_hint) - 2):
            if (
                current_hint[i - 1] != "_"
                and current_hint[i] != "_"
                and current_hint[i + 1] == "_"
                and current_hint[i + 2] != "_"
            ):
                third_empty_place.append(i + 1)

        for i in range(len(current_hint) - 3):
            if (
                current_hint[i] != "_"
                and current_hint[i + 1] != "_"
                and current_hint[i + 2] != "_"
                and current_hint[i + 3] == "_"
            ):
                fourth_empty_place.append(i + 3)

        posterior_1 = np.zeros(len(letters_to_guess))
        for i in first_empty_place:
            second_letter = current_hint[i + 1]
            third_letter = current_hint[i + 2]
            fourth_letter = current_hint[i + 3]

            for j, letter in enumerate(letters_to_guess):
                first_letter = letter
                try:
                    posterior_1[j] += self.model.score(
                        fourth_letter, [first_letter, second_letter, third_letter]
                    )
                except ZeroDivisionError:
                    pass

        posterior_2 = np.zeros(len(letters_to_guess))
        for i in second_empty_place:
            first_letter = current_hint[i - 1]
            third_letter = current_hint[i + 1]
            fourth_letter = current_hint[i + 2]

            for j, letter in enumerate(letters_to_guess):
                second_letter = letter
                try:
                    posterior_2[j] += self.model.score(
                        fourth_letter, [first_letter, second_letter, third_letter]
                    )
                except ZeroDivisionError:
                    pass

        posterior_3 = np.zeros(len(letters_to_guess))
        for i in third_empty_place:
            first_letter = current_hint[i - 2]
            second_letter = current_hint[i - 1]
            fourth_letter = current_hint[i + 1]

            for j, letter in enumerate(letters_to_guess):
                third_letter = letter
                try:
                    posterior_3[j] += self.model.score(
                        fourth_letter, [first_letter, second_letter, third_letter]
                    )
                except ZeroDivisionError:
                    pass

        posterior_4 = np.zeros(len(letters_to_guess))
        for i in fourth_empty_place:
            first_letter = current_hint[i - 3]
            second_letter = current_hint[i - 2]
            third_letter = current_hint[i - 1]

            for j, letter in enumerate(letters_to_guess):
                fourth_letter = letter
                try:
                    posterior_4[j] += self.model.score(
                        fourth_letter, [first_letter, second_letter, third_letter]
                    )
                except ZeroDivisionError:
                    pass

        # update the probability
        posterior = np.array(
            [
                posterior_1,
                posterior_2,
                posterior_3,
                posterior_4,
            ]
        )
        posterior = posterior.mean(axis=0)
        return probabilities + self.w4 * posterior

    def guess_fivegram(self, current_hint, probabilities, letters_to_guess):
        """Use the fivegram to guess the next letter

        Parameters
        ----------
        current_hint : str
            The current hint of the game. (e.g. "_ _ _ _ _ _ _ _ _ _")
        probabilities : np.array
            The probabilities of each letter being the next letter.
        letters_to_guess : list
            The list of letters that can be guessed.

        Returns
        -------
        np.array
            The updated probabilities of each letter being the next letter.
        """
        # remove the spaces
        current_hint = current_hint.replace(" ", "")
        first_empty_place = []
        second_empty_place = []
        third_empty_place = []
        fourth_empty_place = []
        fifth_empty_place = []
        posterior = np.zeros(len(letters_to_guess))

        for i in range(4, len(current_hint)):
            if (
                current_hint[i - 4] == "_"
                and current_hint[i - 3] != "_"
                and current_hint[i - 2] != "_"
                and current_hint[i - 1] != "_"
                and current_hint[i] != "_"
            ):
                first_empty_place.append(i - 4)

        for i in range(3, len(current_hint) - 1):
            if (
                current_hint[i - 3] != "_"
                and current_hint[i - 2] == "_"
                and current_hint[i - 1] != "_"
                and current_hint[i] != "_"
                and current_hint[i + 1] != "_"
            ):
                second_empty_place.append(i - 2)

        for i in range(2, len(current_hint) - 2):
            if (
                current_hint[i - 2] != "_"
                and current_hint[i - 1] != "_"
                and current_hint[i] == "_"
                and current_hint[i + 1] != "_"
                and current_hint[i + 2] != "_"
            ):
                third_empty_place.append(i)

        for i in range(1, len(current_hint) - 3):
            if (
                current_hint[i - 1] != "_"
                and current_hint[i] != "_"
                and current_hint[i + 1] != "_"
                and current_hint[i + 2] == "_"
                and current_hint[i + 3] != "_"
            ):
                fourth_empty_place.append(i + 2)

        for i in range(len(current_hint) - 4):
            if (
                current_hint[i] != "_"
                and current_hint[i + 1] != "_"
                and current_hint[i + 2] != "_"
                and current_hint[i + 3] != "_"
                and current_hint[i + 4] == "_"
            ):
                fifth_empty_place.append(i + 4)

        posterior_1 = np.zeros(len(letters_to_guess))
        for i in first_empty_place:
            second_letter = current_hint[i + 1]
            third_letter = current_hint[i + 2]
            fourth_letter = current_hint[i + 3]
            fifth_letter = current_hint[i + 4]

            for j, letter in enumerate(letters_to_guess):
                first_letter = letter
                try:
                    posterior_1[j] += self.model.score(
                        fifth_letter,
                        [first_letter, second_letter, third_letter, fourth_letter],
                    )
                except ZeroDivisionError:
                    pass

        posterior_2 = np.zeros(len(letters_to_guess))
        for i in second_empty_place:
            first_letter = current_hint[i - 1]
            third_letter = current_hint[i + 1]
            fourth_letter = current_hint[i + 2]
            fifth_letter = current_hint[i + 3]

            for j, letter in enumerate(letters_to_guess):
                second_letter = letter
                try:
                    posterior_2[j] += self.model.score(
                        fifth_letter,
                        [first_letter, second_letter, third_letter, fourth_letter],
                    )
                except ZeroDivisionError:
                    pass

        posterior_3 = np.zeros(len(letters_to_guess))
        for i in third_empty_place:
            first_letter = current_hint[i - 2]
            second_letter = current_hint[i - 1]
            fourth_letter = current_hint[i + 1]
            fifth_letter = current_hint[i + 2]

            for j, letter in enumerate(letters_to_guess):
                third_letter = letter
                try:
                    posterior_3[j] += self.model.score(
                        fifth_letter,
                        [first_letter, second_letter, third_letter, fourth_letter],
                    )
                except ZeroDivisionError:
                    pass

        posterior_4 = np.zeros(len(letters_to_guess))
        for i in fourth_empty_place:
            first_letter = current_hint[i - 3]
            second_letter = current_hint[i - 2]
            third_letter = current_hint[i - 1]
            fifth_letter = current_hint[i + 1]

            for j, letter in enumerate(letters_to_guess):
                fourth_letter = letter
                try:
                    posterior_4[j] += self.model.score(
                        fifth_letter,
                        [first_letter, second_letter, third_letter, fourth_letter],
                    )
                except ZeroDivisionError:
                    pass

        posterior_5 = np.zeros(len(letters_to_guess))
        for i in fifth_empty_place:
            first_letter = current_hint[i - 4]
            second_letter = current_hint[i - 3]
            third_letter = current_hint[i - 2]
            fourth_letter = current_hint[i - 1]

            for j, letter in enumerate(letters_to_guess):
                fifth_letter = letter
                try:
                    posterior_5[j] += self.model.score(
                        fifth_letter,
                        [first_letter, second_letter, third_letter, fourth_letter],
                    )
                except ZeroDivisionError:
                    pass

        # update the probability
        posterior = np.array(
            [
                posterior_1,
                posterior_2,
                posterior_3,
                posterior_4,
                posterior_5,
            ]
        )
        posterior = posterior.mean(axis=0)
        return probabilities + self.w5 * posterior

    def guess_ngram(self, current_hint):
        """Guesses the next letter using n-gram model

        Parameters
        ----------
        current_hint : str
            The current hint

        Returns
        -------
        str
            The next letter to guess
        """
        # initialize probabilities
        probabilities = np.zeros(len(self.letters_to_guess))

        # start with 5-gram
        probabilities = self.guess_fivegram(
            current_hint, probabilities, self.letters_to_guess
        )

        # pass the update probabilities to 4-gram as prior
        probabilities = self.guess_fourgram(
            current_hint, probabilities, self.letters_to_guess
        )

        # pass the update probabilities to 3-gram as prior
        probabilities = self.guess_trigram(
            current_hint, probabilities, self.letters_to_guess
        )
        probabilities = self.guess_bigram(
            current_hint, probabilities, self.letters_to_guess
        )

        # pass the update probabilities to 1-gram as prior
        probabilities = self.guess_unigram(
            current_hint, probabilities, self.letters_to_guess
        )

        probabilities_max = np.max(probabilities)
        # use frequency if all probabilities are 0
        if probabilities_max == 0:
            return self.guess_frequency(current_hint)
        # otherwise return the letter with the highest probability
        return np.array(self.letters_to_guess)[np.argsort(probabilities)[::-1]][0]

    def build_dictionary(self):
        dictionary = open(self.dictionary_dir, "r").read().split("\n")
        dictionary = [word for word in dictionary if len(word) > 0]
        return dictionary

    def check_win(self, displayed_word):
        if displayed_word.find("_ ") == -1:
            return True
        return False

    def display_word(self, word, letter_to_show=[]):
        letters_to_hide = [l for l in word if l not in letter_to_show]
        result = word
        for l in letters_to_hide:
            result = result.replace(l, "_ ")
        return result

    def guess_frequency(self, word):
        """Guesses the next letter using frequency

        Parameters
        ----------
        current_hint : str
            The current hint

        Returns
        -------
        str
            The next letter to guess
        """
        word_pattern = word.replace("_ ", ".")
        # find length of passed word
        len_word = len(word_pattern)

        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []

        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue

            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(word_pattern, dict_word):
                new_dictionary.append(dict_word)

        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary

        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)

        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()

        guess_letter = "!"

        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter, instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                # self.guessed_letters.append(guess_letter)
                break

        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == "!":
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    # self.guessed_letters.append(guess_letter)
                    break

        return guess_letter

    def update_dictionary(self, letter_to_exclude):
        """Updates the dictionary to exclude words with the letter to exclude

        Parameters
        ----------
        letter_to_exclude : str
            The letter to exclude
        """
        self.current_dictionary = [
            word for word in self.current_dictionary if letter_to_exclude not in word
        ]

    def start_game(self, verbose=True, random_word=None):
        """Starts the game

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the game progress, by default True
        random_word : str, optional

        Returns
        -------
        int: 0 if the game is lost, 1 if the game is won
        """

        self.guessed_letters = []
        self.current_dictionary = self.dictionary_og
        if random_word is None:
            random_word = random.choice(self.dictionary_og)
        if verbose:
            print(random_word)
            print("Word is")
        displayed_word = self.display_word(random_word)
        word_length = displayed_word.count("_")
        if verbose:
            print(displayed_word)
        trial = 0

        # Start the game
        while trial < TRIES:
            # A list of letters that have not been guessed yet
            self.letters_to_guess = [
                letter
                for letter in self.full_dictionary_common_letter_sorted
                if letter not in self.guessed_letters
            ]

            # number of correct guesses used to determine whether to guess by frequency or ngram
            correct_guesses = word_length - displayed_word.count("_")
            tries_remaining = TRIES - trial
            if verbose:
                print("Correct guesses: ", correct_guesses)

            # if the number of correct guesses is greater than the word length times the frequency guess percentage and have more than 2 tries remaining guess by frequency
            if (
                int(word_length * self.frequency_guess_percentage) >= correct_guesses
            ) and (tries_remaining > 2):
                if verbose:
                    print("Guessing by frequency")
                cur_letter = self.guess_frequency(displayed_word)
            # otherwise guess by ngram
            else:
                if verbose:
                    print("Guessing by ngram")
                cur_letter = self.guess_ngram(
                    displayed_word,
                )

            if verbose:
                print("Guessed letter ", cur_letter)

            # make sure the letter is only one character
            if len(cur_letter) > 1:
                raise ValueError("Letter is more than one character")

            # increase trial count if the letter is not in the word
            # Also update the dictionary to exclude words with the letter
            if cur_letter not in random_word:
                trial += 1
                self.update_dictionary(cur_letter)

            # add the letter to the guessed letters list and update the displayed word
            self.guessed_letters.append(cur_letter)
            displayed_word = self.display_word(random_word, self.guessed_letters)
            if verbose:
                print(f"After Trial Number, {trial}")
                print(f"Displayed word is: {displayed_word}")
                print(len(self.current_dictionary))

            # if the displayed word has no more underscores, the game is won
            if self.check_win(displayed_word=displayed_word):
                if verbose:
                    print("You Won")
                return 1
        # if the game is lost
        if verbose:
            print("You lose")
        return 0

    def test(self, n=100):
        """Tests the model on the test data

        Parameters
        ----------
        n : int, optional
            Number of games to play, by default 100

        Returns
        -------
        float
            The percentage of games won
        """
        won_count = 0
        for i in tqdm.tqdm(range(n)):
            random_word = random.choice(self.test_data)
            won = self.start_game(verbose=False, random_word=random_word)
            won_count += won
        return won_count / n

    def _test_one(self, params, n=500):
        """Tests the model on the train and the test data with given parameters. Used for hyperparameter tuning

        Parameters
        ----------
        params : list
            List of parameters to test
        n : int, optional
            Number of games to play, by default 500

        Returns
        -------
        dict
            Dictionary of results
        """

        # unpack parameters
        p, w1, w2, w3, w4, w5 = params
        # set parameters
        self.frequency_guess_percentage = p
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        # Perform training data
        train_won_count = 0
        for _ in range(n):
            train_won_count += self.start_game(verbose=False)
        train_percentage_won = train_won_count / n

        # Perform test data
        test_won_count = 0
        for i in range(n):
            random_word = random.choice(self.test_data)
            test_won_count += self.start_game(verbose=False, random_word=random_word)
        test_percentage_won = test_won_count / n

        result_dictionary = {
            "Parameters": params,
            "Train Score": train_percentage_won,
            "Test Score": test_percentage_won,
        }
        return result_dictionary

    def _list_to_dictionary(self, params):
        return {
            "frequency_guess_percentage": params[0],
            "w1": params[1],
            "w2": params[2],
            "w3": params[3],
            "w4": params[4],
            "w5": params[5],
        }

    def update_info(self, i, current_result, current_parameters):
        """Updates the info.txt file with the results of the current run

        Parameters
        ----------
        i : int
            Run number
        current_result : dict
            Dictionary of results
        current_parameters : list
            List of parameters used
        """
        train_score = current_result["Train Score"]
        test_score = current_result["Test Score"]
        p = self._list_to_dictionary(current_parameters)

        with open("info.txt", "+a") as f:
            f.write(f"Run Number: {i}\n")
            f.write(f"Parameters: {p}\n")
            f.write(f"Train Score: {train_score}\n")
            f.write(f"Test Score: {test_score}\n\n")

    def tune_hyperparameters(
        self,
        all_params_list,
        n=500,
        verbose=True,
        start=0,
        end=None,
        start_id=1,
    ):
        """Tunes the hyperparameters of the model

        Parameters
        ----------
        all_params_list : list
            List of lists of parameters to test. Each list should contain the parameters to test for that parameter
            The funtion will test all combinations of the parameters and test them
        n : int, optional
            Number of games to play, by default 500
        verbose : bool, optional
            Whether to print the results of each run, by default True
        start : int, optional
            The run number to start at, by default 0. Useful if the function is stopped and needs to be restarted
        end : int, optional
            The run number to end at, by default None which means the function will run until the end of the list
        start_id : int, optional
            The start id to be written on the info.txt file, by default 1

        Returns
        -------
        list
            List of dictionaries of results
        """
        params = list(product(*all_params_list))
        results = []
        start = start
        end = end if end is not None else len(params)
        for i in tqdm.tqdm(range(start, end), desc="Tuning..."):
            p = params[i]
            if verbose:
                print(f"Current Parameters: {self._list_to_dictionary(p)}")
            res = self._test_one(p, n=n)
            if verbose:
                print(f"Current Results:")
                print(f"\tTrain Score: {res['Train Score']}")
                print(f"\tTest Score: {res['Test Score']}\n")
            results.append(res)
            self.update_info(start_id, res, p)
            start_id += 1
        return results
