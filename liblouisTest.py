import louis


test = input("Enter Text To be Converted")
print(louis.translateString(["braille-patterns.cti", "en-us-g2.ctb"], test))