from pybrl import pybrl as brl
from braillecodeToASCII import braille_to_ascii_conversion as b2t
import louis

def convert_to_braille(transcripted_text):
        
    brltext_g1 = louis.translateString(["braille-patterns.cti", "en-ueb-g1.ctb"],transcripted_text)
    brltext_g2 = louis.translateString(["braille-patterns.cti", "en-ueb-g2.ctb"],transcripted_text)
    brfText_g1 = b2t(brltext_g1)
    brfText_g2 = b2t(brltext_g2)
    
    # Grade 1 Files
    pef_g1 = brltext_g1
    brf_g1 = f"\n{brfText_g1.upper()}\n"

    # Grade 2 Files
    pef_g2 = brltext_g2
    brf_g2 = f"\n{brfText_g2.upper()}\n"

    
    return brf_g1, brf_g2, pef_g1, pef_g2
