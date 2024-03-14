from docx import Document

def create_word_document(file_name, content):
    doc = Document()
    doc.add_paragraph(content)
    doc.save(file_name)

def format_content(content, line_length=42, lines_per_page=25):
    formatted_content = []
    words = content.split()
    current_line = ''
    line_count = 0

    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            if current_line:
                current_line += ' '
            current_line += word
        else:
            formatted_content.append(current_line)
            current_line = word
            line_count += 1
            if line_count == lines_per_page:
                formatted_content.append('\f')  # Form feed character to indicate page break
                line_count = 0

    if current_line:
        formatted_content.append(current_line)

    return '\n'.join(formatted_content)

def create_pef_file(file_name, content):
    try:
        formatted_content = format_content(content)
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(formatted_content)
        print(f"Braille Ready File '{file_name}' created successfully.")
    except Exception as e:
        print(f"Error creating Braille Ready File: {e}")

def create_brf_file(file_name, content):
    try:
        formatted_content = format_content(content)
        with open(file_name, 'w') as file:
            file.write(formatted_content)
        print(f"Braille Ready File '{file_name}' created successfully.")
    except Exception as e:
        print(f"Error creating Braille Ready File: {e}")


"""def create_pef_file(file_name, content):
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
"""
