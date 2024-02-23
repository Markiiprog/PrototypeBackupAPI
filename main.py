import speechbrain as sb
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import os
import shutil
import base64

from preprocess import MP32Wav, Video2Wav
from OCR import perform_ocr
from loadModels import OCR_Model, ASR_Model
from generateFiles import create_word_document,create_brf_file,create_pef_file
from docInput import extract_text_from_file
from convertText import convert_to_braille
from typing import Dict

app = FastAPI()  #uvicorn main:app --reload (This runs starts a local instance of the 

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
    base_url = "http://34.142.200.21:8000"  # Change this to your FastAPI server address
    download_links = {
        "doc": f"{base_url}/download/outputs/{filename}(transcription).doc",
        "pef": f"{base_url}/download/outputs/{filename}(transcription).pef",
        "brf": f"{base_url}/download/outputs/{filename}(transcription).brf"
    }
    return download_links

def get_response_content(filename: str, transcription: str, brf: str, pef: str) -> Dict[str, str]:
    download_links = get_download_links(filename)
    response_content = {
        "Transcription": transcription,
        "Braille": pef,
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
        print(file_path)
        transcription = asr_model.transcribe_file(file_path)
        # Assuming asr_model is properly defined elsewhere
        transcription = transcription.lower()
        brf,pef = convert_to_braille(transcription.lower())

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')

        create_word_document(docx_filename,transcription)
        create_pef_file(pef_filename,pef)
        create_brf_file(brf_filename,brf)

        os.remove(new_file_path)

        # download_links = get_download_links(name)

        response_content = get_response_content(name, transcription, brf, pef)

        #returnJSON with braille and transcription
        return JSONResponse(content=response_content)

        # Return JSON response with download links
        # return JSONResponse(content=download_links)
    
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
        transcripted_text = transcripted_text.lower()
        brf,pef = convert_to_braille(transcripted_text.lower())

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')

        create_word_document(docx_filename,transcripted_text)
        create_pef_file(pef_filename,pef)
        create_brf_file(brf_filename,brf)

        os.remove(new_file_path)

        # download_links = get_download_links(name)

        response_content = get_response_content(name, transcripted_text, brf, pef)

        #returnJSON with braille and transcription
        return JSONResponse(content=response_content)

        # Return JSON response with download links
        # return JSONResponse(content=download_links)
    
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
        brf,pef = convert_to_braille(transcripted_text)  

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')

        create_word_document(docx_filename,transcripted_text)
        create_pef_file(pef_filename,pef)
        create_brf_file(brf_filename,brf)

        os.remove(new_file_path)

        # download_links = get_download_links(name)

        response_content = get_response_content(name, transcripted_text, brf, pef)

        #returnJSON with braille and transcription
        return JSONResponse(content=response_content)

        # Return JSON response with download links
        # return JSONResponse(content=download_links)
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing file: {str(e)}")
        # Raise an HTTPException with a 500 status code
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
        brf,pef = convert_to_braille(transcripted_text)

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        name,_ = os.path.splitext(file.filename) 

        docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
        pef_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
        brf_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')

        create_word_document(docx_filename,transcripted_text)
        create_pef_file(pef_filename,pef)
        create_brf_file(brf_filename,brf)

        os.remove(new_file_path)

        # download_links = get_download_links(name)

        response_content = get_response_content(name, transcripted_text, brf, pef)

        #returnJSON with braille and transcription
        return JSONResponse(content=response_content)

        # Return JSON response with download links
        # return JSONResponse(content=download_links)
    
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

    brf,pef = convert_to_braille(input_string)

    name = 'text_input'
    docx_filename = os.path.join(OUTPUTDIR, name + '(transcription).doc')
    pef_filename = os.path.join(OUTPUTDIR, name + '(transcription).pef')
    brf_filename = os.path.join(OUTPUTDIR, name + '(transcription).brf')

    create_word_document(docx_filename,input_string)
    create_pef_file(pef_filename,pef)
    create_brf_file(brf_filename,brf)


    # download_links = get_download_links(name)
    
    response_content = get_response_content(name, input_string, brf, pef)

    #returnJSON with braille and transcription
    return JSONResponse(content=response_content)

    # Return JSON response with download links
    # return JSONResponse(content=download_links)

@app.get('/download/outputs/{file_path:path}')
async def download_file(file_path: str):
    file_full_path = os.path.join(OUTPUTDIR, file_path)
    if not os.path.isfile(file_full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_full_path,
                       background = BackgroundTask(os.remove,file_full_path) #deletes temp file after download.
                    )