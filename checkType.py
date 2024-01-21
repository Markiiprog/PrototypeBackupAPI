import mimetypes

def get_file_type(file_path):
    mime, encoding = mimetypes.guess_type(file_path)
    return mime