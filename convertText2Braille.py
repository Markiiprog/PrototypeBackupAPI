from pybrl import pybrl as brl
from braillecodeToASCII import braille_to_ascii_conversion as b2t
#from pybraille import convertText
from convert2Grade1 import translate_to_braille as convertText
import louis

def convert_to_braille(transcripted_text):

    lines = transcripted_text.split('\n')

    translated_lines_pef_g1 = []
    translated_lines_brf_g1 = []

    translated_lines_pef_g2 = []
    translated_lines_brf_g2 = []

    for line in lines:
        
        #brltext_g1 = convertText(line)
        #brltext_g2 = brl.translate(line) 
        #brltext_g2 = brl.toUnicodeSymbols(brltext_g2, flatten=True)
        brltext_g1 = louis.translateString(["braille-patterns.cti", "en-ueb-g1.ctb"], line)
        brltext_g2 = louis.translateString(["braille-patterns.cti", "en-ueb-g2.ctb"], line)
        brfText_g1 = b2t(brltext_g1)
        brfText_g2 = b2t(brltext_g2)
        
        #Grade 1 PEF / BRF
        translated_lines_brf_g1.append(brfText_g1)
        translated_lines_pef_g1.append(brltext_g1)

        #Grade 2 PEF / BRF
        translated_lines_brf_g2.append(brfText_g2)
        translated_lines_pef_g2.append(brltext_g2)

    # Grade 1 Files
    pef_g1 ='\n'.join(translated_lines_pef_g1)
    brf_g1 = '\n'.join(translated_lines_brf_g1)
    brf_g1 = f"\n{brf_g1.upper()}\n"
    
    # Grade 2 Files
    pef_g2 = '\n'.join(translated_lines_pef_g2)
    brf_g2 = '\n'.join(translated_lines_brf_g2)
    brf_g2 = f"\n{brf_g2.upper()}\n"
    
    return brf_g1, brf_g2, pef_g1, pef_g2