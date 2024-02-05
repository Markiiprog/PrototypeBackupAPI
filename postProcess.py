from nltk.tokenize import sent_tokenize
from fastpunct import FastPunct

import nltk
nltk.download('punkt')

import re
from spellchecker import SpellChecker

def perform_spell_check(text):
    spell = SpellChecker(distance=1)

    # Split the text into lines
    lines = text.split('\n')

    corrected_lines = []
    for line in lines:
        words = re.findall(r'\b\w+\b', line)  # Extract individual words

        # Handle None values from spell.correction()
        corrected_words = []
        for word in words:
            correction = spell.correction(word)
            corrected_words.append(correction if correction is not None else word)

        corrected_line = ' '.join(corrected_words)
        corrected_lines.append(corrected_line)

    corrected_text = '\n'.join(corrected_lines)
    return corrected_text


def perform_punctuation_check(text):
    # Split the text into lines
    lines = text.split('\n')

    # Initialize the FastPunct model outside the loop
    punctuator = FastPunct()

    corrected_lines = []
    for line in lines:
        # Split the line into sentences using nltk sentence tokenization
        sentences = sent_tokenize(line)

        # Use list comprehension for concise code
        corrected_sentences = [punctuator.punct(sentence) for sentence in sentences]

        corrected_lines.append('\n'.join(corrected_sentences))

    corrected_text = '\n'.join(corrected_lines)
    return corrected_text

