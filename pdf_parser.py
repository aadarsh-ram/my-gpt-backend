import fitz
from multi_column import column_boxes
import re

def column_cleaning(text):
    new_text = text.replace('-\n', '')
    new_text = new_text.replace('.\n', '#')
    new_text = new_text.replace('\n', ' ')
    new_text = new_text.replace('#', '\n') + '\n'
    new_text = new_text.replace('= \n', '= ')
    return new_text

def pdf_to_ocr(path):
    doc = fitz.open(path)

    all_tables_text = []
    all_text = []

    for page in doc:
        # Find tables and associated text
        table_finder = page.find_tables()
        for table in table_finder.tables:
            table_text = page.get_text(clip=table.bbox)
            all_tables_text.append(table_text)
        
        # Single column research paper
        bboxes = column_boxes(page, no_image_text=True)
        for rect in bboxes:
            text = page.get_text(clip=rect, sort=True)
            # Remove tables
            for table_text in all_tables_text:
                if (table_text in text):
                    text = text.replace(table_text, '')
            
            new_text = column_cleaning(text)
            all_text.append(new_text)

    full_text = ''.join(all_text)

    # Remove references
    pos = re.findall('references', full_text, flags=re.IGNORECASE)
    exact_word = ''
    for ind in range(len(pos)-1, -1, -1):
        if (pos[ind] == 'REFERENCES'):
            exact_word = 'REFERENCES'
            break
        if (pos[ind] == 'References'):
            exact_word = 'References'
            break

    word_ind = full_text.rfind(exact_word)
    full_text = full_text[:word_ind]

    print (full_text.encode('ascii', errors='ignore').decode())
    return (full_text.encode('ascii', errors='ignore').decode())

if __name__ == '__main__':
    pdf_path = "./samples/tiny-attention.pdf"
    res = pdf_to_ocr(pdf_path)
    print (res)