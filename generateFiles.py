from docx import Document

def create_word_document(file_name, content):
    doc = Document()
    doc.add_paragraph(content)
    doc.save(file_name)

def format_content(content, line_length=40, lines_per_page=24):
    formatted_content = []
    words = content.split()
    current_line = ''
    line_count = 0

    for word in words:
        # Check if adding the word would exceed line length
        if len(current_line) + len(word) + 1 > line_length:
            # Check if the word itself exceeds line length
            if len(word) > line_length:
                # If word exceeds line length, add line break before it
                if current_line:
                    formatted_content.append(current_line)
                current_line = word
                line_count += 1
                if line_count == lines_per_page:
                    formatted_content.append('\f')  # Form feed character to indicate page break
                    line_count = 0
            else:
                # If word doesn't exceed line length, add current line and start a new one with the word
                formatted_content.append(current_line)
                current_line = word
                line_count += 1
                if line_count == lines_per_page:
                    formatted_content.append('\f')  # Form feed character to indicate page break
                    line_count = 0
        else:
            # If adding the word doesn't exceed line length, add it to the current line
            if current_line:
                current_line += ' '
            current_line += word

    # Add the last line if it exists
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
