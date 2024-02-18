from pybraille import pybrl as brl
from braillecodeToASCII import braille_to_ascii_conversion as b2t

def convert_to_braille(transcripted_text):
    lines = transcripted_text.split('\n')
    translated_lines_pef = []
    translated_lines_brf = []
    for line in lines:
        brltext = brl.translate(line) 
        brltext = brl.toUnicodeSymbols(brltext, flatten=True)
        brfText = b2t(brltext)
        
        translated_lines_brf.append(brfText)
        translated_lines_pef.append(brltext)
    
    pef = '\n'.join(translated_lines_pef)
    brf = '\n'.join(translated_lines_brf)
    brf = f"\n{brf.upper()}\n"

    return brf, pef