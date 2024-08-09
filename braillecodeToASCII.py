# ASCII and Braille symbols combined into a dictionary
braille_to_ascii = {
    '⠀': ' ', '⠮': '!', '⠐': '"', '⠼': '#', '⠫': '$', '⠩': '%', '⠯': '&', '⠄': """'""",
    '⠷': '(', '⠾': ')', '⠡': '*', '⠬': '+', '⠠': ',', '⠤': '-', '⠨': '.', '⠌': '/',
    '⠴': '0', '⠂': '1', '⠆': '2', '⠒': '3', '⠲': '4', '⠢': '5', '⠖': '6', '⠶': '7',
    '⠦': '8', '⠔': '9', '⠱': ':', '⠰': ';', '⠣': '<', '⠿': '=', '⠜': '>', '⠹': '?',
    '⠈': '@', '⠁': 'a', '⠃': 'b', '⠉': 'c', '⠙': 'd', '⠑': 'e', '⠋': 'f', '⠛': 'g',
    '⠓': 'h', '⠊': 'i', '⠚': 'j', '⠅': 'k', '⠇': 'l', '⠍': 'm', '⠝': 'n', '⠕': 'o',
    '⠏': 'p', '⠟': 'q', '⠗': 'r', '⠎': 's', '⠞': 't', '⠥': 'u', '⠧': 'v', '⠺': 'w',
    '⠭': 'x', '⠽': 'y', '⠵': 'z', '⠪': '[', '⠳': '\\', '⠻': ']', '⠘': '^', '⠸': '_'
}

def braille_to_ascii_conversion(braille_string):
    ascii_string = ''
    for symbol in braille_string:
        if symbol in braille_to_ascii:
            ascii_string += braille_to_ascii[symbol]
    return ascii_string
