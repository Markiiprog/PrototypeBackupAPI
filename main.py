import speechbrain as sb
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import os
import shutil

from preprocess import MP32Wav, Video2Wav
from OCR import perform_ocr
from loadModels import OCR_Model, ASR_Model
from generateFiles import create_word_document, create_brf_file, create_pef_file
from docInput import extract_text_from_file
from convertText2Braille import convert_to_braille
from typing import Dict

app = FastAPI()  # uvicorn main:app --reload (This runs starts a local instance of the server)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = OCR_Model()
asr_model = ASR_Model()

OUTPUTDIR = 'outputs/'
AUDIODIR = 'audio_cache/'


def get_download_links(filename: str) -> dict:
    # base_url = "http://localhost:8000"
    base_url = "http://34.126.91.78:8000"  # Change this to your FastAPI server address
    download_links = {
        "docx": f"{base_url}/download/outputs/{filename}(transcription).doc",
        "pef_g1": f"{base_url}/download/outputs/{filename}(transcription).pef",
        "brf_g1": f"{base_url}/download/outputs/{filename}(transcription).brf",
        "pef_g2": f"{base_url}/download/outputs/{filename}_g2(transcription).pef",
        "brf_g2": f"{base_url}/download/outputs/{filename}_g2(transcription).brf"
    }
    return download_links

def get_response_content(filename: str, transcription: str,  pef_g1: str, pef_g2:str) -> Dict[str, str]:
    download_links = get_download_links(filename)
    response_content = {
        "Transcription": transcription,
        "Braille": pef_g1,
        "Braille_G2": pef_g2,
        "download_links": download_links
    }
    return response_content


@app.get('/')
async def root():
    return {
        'ASR API': 'Active'
    }


@app.post('/transcribe/audio')
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Ensure the directory exists; create it if necessary
        os.makedirs(OUTPUTDIR, exist_ok=True)
        
        # Save the uploaded file to the specified directory
        file_path = os.path.join(AUDIODIR, file.filename)
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())
        
        # Check file extension and process accordingly
        name, ext = os.path.splitext(file_path)
        if ext == ".mp3":
            # Convert MP3 to WAV if necessary
            # Replace this with your conversion function (MP32Wav)
            file_path = MP32Wav(file_path, OUTPUTDIR, f"{name}.wav")
            if not file_path:
                return {"error": "Failed to convert MP3 to WAV"}
        
        # Transcribe audio file
        transcription = asr_model.transcribe_file(file_path)
        brf_g1,brf_g2,pef_g1,pef_g2 = convert_to_braille(transcription.lower())

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')
        pef_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).pef')
        brf_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).brf')

        create_word_document(docx_filename,transcription.lower())
        create_pef_file(pef_g1_filename,pef_g1)
        create_brf_file(brf_g1_filename,brf_g1)
        create_pef_file(pef_g2_filename,pef_g2)
        create_brf_file(brf_g2_filename,brf_g2)
        

        os.remove(new_file_path)
        os.remove(f"{name}.wav")

        response_content = get_response_content(name, transcription,pef_g1,pef_g2)

        #returnJSON with braille and transcription
        return JSONResponse(content=response_content)
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing file: {str(e)}")
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(f'/transcribe/video')
async def transcribe_video(file: UploadFile = File(...)):
    try:
        # Ensure the directory exists; create it if necessary
        os.makedirs(OUTPUTDIR, exist_ok=True)
        
        # Save the uploaded file to the specified directory
        file_path = os.path.join(AUDIODIR, file.filename)
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())
        
        # Check file extension and process accordingly
        name, ext = os.path.splitext(file_path)
        if ext == ".mp4":
            # Convert MP3 to WAV if necessary
            # Replace this with your conversion function (MP32Wav)
            file_path = Video2Wav(file_path, OUTPUTDIR, f"{name}.wav")
            if not file_path:
                return {"error": "Failed to convert MP4 to WAV"}

        transcripted_text = asr_model.transcribe_file(file_path)
        brf_g1,brf_g2,pef_g1,pef_g2 = convert_to_braille(transcripted_text.lower())
        

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')
        pef_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).pef')
        brf_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).brf')

        create_word_document(docx_filename,transcripted_text.lower())
        create_pef_file(pef_g1_filename,pef_g1)
        create_brf_file(brf_g1_filename,brf_g1)
        create_pef_file(pef_g2_filename,pef_g2)
        create_brf_file(brf_g2_filename,brf_g2)
        
        os.remove(new_file_path)
        os.remove(f"{name}.wav")
        response_content = get_response_content(name, transcripted_text,pef_g1,pef_g2)

        #return JSON with braille and transcription
        return JSONResponse(content=response_content)
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing file: {str(e)}")
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/transcribe/image')
async def transcribe_image(file: UploadFile = File(...)):
    try:
        # Ensure the directory exists; create it if necessary
        os.makedirs(OUTPUTDIR, exist_ok=True)
        
        # Save the uploaded file to the specified directory
        file_path = os.path.join(OUTPUTDIR, file.filename)
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())
        
        # Check file extension and process accordingly
        name, ext = os.path.splitext(file_path)

        # Perform transcription using the full file path
        transcripted_text = perform_ocr(file_path,reader)
        brf_g1,brf_g2,pef_g1,pef_g2 = convert_to_braille(transcripted_text)

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')
        pef_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).pef')
        brf_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).brf')

        create_word_document(docx_filename,transcripted_text)
        create_pef_file(pef_g1_filename,pef_g1)
        create_brf_file(brf_g1_filename,brf_g1)
        create_pef_file(pef_g2_filename,pef_g2)
        create_brf_file(brf_g2_filename,brf_g2)

        os.remove(new_file_path)

        response_content = get_response_content(name, transcripted_text,pef_g1,pef_g2)

        #returnJSON with braille and transcription
        return JSONResponse(content=response_content)
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing file: {str(e)}")
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/transcribe/docs')
async def transcribe_documents(file: UploadFile = File(...)):
    try:
        os.makedirs(OUTPUTDIR, exist_ok=True)

        file_path = os.path.join(OUTPUTDIR, file.filename)
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        name, ext = os.path.splitext(file_path)

        transcripted_text = extract_text_from_file(file_path)
        brf, pef = convert_to_braille(transcripted_text)

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name, _ = os.path.splitext(file.filename)

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')

        create_word_document(docx_filename, transcripted_text)
        create_pef_file(pef_filename, pef)
        create_brf_file(brf_filename, brf)

        os.remove(new_file_path)

        response_content = get_response_content(name, transcripted_text, brf, pef)

        return JSONResponse(content=response_content)

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
@app.post('/transcribe/docs') 
async def transcribe_documents(file: UploadFile = File(...)): 
    
    try:
        # Ensure the directory exists; create it if necessary
        os.makedirs(OUTPUTDIR, exist_ok=True)
        
        # Save the uploaded file to the specified directory
        file_path = os.path.join(OUTPUTDIR, file.filename)
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())
        
        # Check file extension and process accordingly
        name, ext = os.path.splitext(file_path)

        # Perform transcription using the full file path
        transcripted_text = extract_text_from_file(file_path)
        brf_g1,brf_g2,pef_g1,pef_g2 = convert_to_braille(transcripted_text)

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')
        pef_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).pef')
        brf_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).brf')

        create_word_document(docx_filename,transcripted_text)
        create_pef_file(pef_g1_filename,pef_g1)
        create_brf_file(brf_g1_filename,brf_g1)
        create_pef_file(pef_g2_filename,pef_g2)
        create_brf_file(brf_g2_filename,brf_g2)

        os.remove(new_file_path)

        response_content = get_response_content(name, transcripted_text,pef_g1,pef_g2)

        #returnJSON with braille and transcription
        return JSONResponse(content=response_content)
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing file: {str(e)}")
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/transcribe/text')
async def transcribe_text(request_data: dict):
    if 'input_string' not in request_data:
        raise HTTPException(status_code=400, detail="Input string not provided")

    input_string = request_data['input_string']

    brf_g1,brf_g2,pef_g1,pef_g2 = convert_to_braille(input_string)

    name = 'text_input'
    docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
    pef_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
    brf_g1_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')
    pef_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).pef')
    brf_g2_filename = os.path.join(OUTPUTDIR, name + '_g2(transcription).brf')

    create_word_document(docx_filename,input_string)
    create_pef_file(pef_g1_filename,pef_g1)
    create_brf_file(brf_g1_filename,brf_g1)
    create_pef_file(pef_g2_filename,pef_g2)
    create_brf_file(brf_g2_filename,brf_g2)
    
    response_content = get_response_content(name, input_string,pef_g1,pef_g2)

    #returnJSON with braille and transcription
    return JSONResponse(content=response_content)


@app.get('/download/outputs/{file_path:path}')
async def download_file(file_path: str):
    file_full_path = os.path.join(OUTPUTDIR, file_path)
    if not os.path.isfile(file_full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_full_path,
                        background=BackgroundTask(os.remove, file_full_path))  # deletes temp file after download.
