from docx import Document

def create_word_document(file_name, content):
    doc = Document()
    doc.add_paragraph(content)
    doc.save(file_name)

def create_pef_file(file_name, content):
    try:
        with open(file_name, 'w',encoding='utf-8') as file:
            file.write(content)
        print(f"Braille Ready File '{file_name}' created successfully.")
    except Exception as e:
        print(f"Error creating Braille Ready File: {e}")

def create_brf_file(file_name, content):
    try:
        with open(file_name, 'w') as file:
            file.write(content)
        print(f"Braille Ready File '{file_name}' created successfully.")
    except Exception as e:
        print(f"Error creating Braille Ready File: {e}")

