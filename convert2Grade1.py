
alphaBraille = ['⠁', '⠃', '⠉', '⠙', '⠑', '⠋', '⠛', '⠓', '⠊', '⠚', '⠅', '⠇',
                '⠍', '⠝', '⠕', '⠏', '⠟', '⠗', '⠎', '⠞', '⠥', '⠧', '⠺', '⠭', 
                '⠽', '⠵', ' ']
numBraille = ['⠁', '⠃', '⠉', '⠙', '⠑', '⠋', '⠛', '⠓', '⠊', '⠚']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
punctuation = [',', ';', ':', '.', '?', '!', '(', ')', '/', '-']
punctuationBraille = ['⠂', '⠆', '⠒', '⠲', '⠦', '⠖', '⠐⠣', '⠐⠜', '⠸⠌', '⠤']
characters = ['&', '*', '@', '©', '®', '™', '°']
characterBraille = ['⠈⠯', '⠐⠔', '⠈⠁', '⠘⠉', '⠘⠗', '⠘⠞', '⠘⠚']

# Grade 1 UEB prefixes
grade1UEBPrefixes = {
    "capital": "⠠",
    "numeric": "⠼",
    "grade1" : "⠰",
    "capital_terminator" : "⠠⠄"
}

def translate_to_braille(text):
    if not text:
        return ' '  # Return a space if the input text is empty or null
    
    inputString = ''
    words = text.split()
    for word_index, word in enumerate(words):
        numeric_prefix_used = False
        double_capital_used = False
        for i, char in enumerate(word):
            if char.isdigit():
                # If the character is a digit, handle it separately
                if not numeric_prefix_used:
                    # Use numeric prefix only once for a sequence of numbers
                    inputString += grade1UEBPrefixes["numeric"]
                    numeric_prefix_used = True
                inputString += numBraille[nums.index(char)]
            elif char.isupper():
                # If the character is uppercase, apply the appropriate capitalization rules
                if not double_capital_used:
                    # Check if there are consecutive uppercase characters following
                    has_consecutive_upper = any(c.isupper() for c in word[i+1:])
                    if has_consecutive_upper:
                        inputString += grade1UEBPrefixes["capital"] * 2
                        double_capital_used = True
                    else:
                        inputString += grade1UEBPrefixes["capital"]
                    inputString += alphaBraille[alphabet.index(char.lower())]
                elif not double_capital_used:
                    # Use capital prefix for the first uppercase character in the word
                    inputString += grade1UEBPrefixes["capital"]
                inputString += alphaBraille[alphabet.index(char.lower())]
            elif char in alphabet:
                # If the character is in the alphabet, directly convert it to Braille
                if double_capital_used:
                    # Found a character that is not uppercase, insert capital terminator
                    inputString += grade1UEBPrefixes["capital_terminator"]
                    double_capital_used = False
                    inputString += alphaBraille[alphabet.index(char)]
                else:
                    inputString += alphaBraille[alphabet.index(char)]
            elif char in punctuation:
                # If the character is a punctuation mark, convert it to Braille
                inputString += punctuationBraille[punctuation.index(char)]
            elif char in characters:
                # If the character is a special character, convert it to Braille
                inputString += characterBraille[characters.index(char)]

            # Check the next character to determine prefix usage
            if i < len(word) - 1:
                next_char = word[i + 1]
                if next_char.isdigit() and not numeric_prefix_used:
                    # If the next character is a digit and numeric prefix is not used, add numeric prefix
                    inputString += grade1UEBPrefixes["numeric"]
                    numeric_prefix_used = True
                elif not next_char.isdigit() and not next_char.isupper() and next_char in 'abcdefghij' and numeric_prefix_used and word[i].isdigit():
                    # If the next character is non-numeric and belongs to 'a-j' range and numeric prefix was used, add grade1 prefix
                    inputString += grade1UEBPrefixes["grade1"]
                    numeric_prefix_used = False
            
        inputString += ' ' # Add space between words

    return inputString.strip() # Remove trailing space