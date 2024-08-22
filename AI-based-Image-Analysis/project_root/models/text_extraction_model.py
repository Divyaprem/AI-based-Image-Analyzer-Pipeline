import easyocr
import numpy as np

def load_easyocr_model():
    reader = easyocr.Reader(['en'])
    return reader

def extract_text(reader, img):
    result = reader.readtext(np.array(img))
    return [text[1] for text in result]
