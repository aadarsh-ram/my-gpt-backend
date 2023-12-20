import fitz
from multi_column import column_boxes
import re
import pdf2image
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pprint import pprint

def column_cleaning(text):
    new_text = text.replace('-\n', '')
    new_text = new_text.replace('.\n', '#')
    new_text = new_text.replace('\n', ' ')
    new_text = new_text.replace('#', '\n') + '\n'
    new_text = new_text.replace('= \n', '= ')
    return new_text

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)

def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text

def print_pages(pdf_file):
    images = pdf_to_img(pdf_file)
    text = ''
    for pg, img in enumerate(images):
        text+= ocr_core(img)
    return text

def pdf_to_ocr(path):
    return print_pages(path)

def pdf_to_ocr_fitz(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text+= page.get_text()
    return text

if __name__ == '__main__':
    pdf_path = "./samples/tiny-attention.pdf"
    res = pdf_to_ocr(pdf_path)
    print (res)