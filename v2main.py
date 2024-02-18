import speechbrain as sb
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from preprocess import MP32Wav, Video2Wav
from OCR import perform_ocr
from loadModels import OCR_Model, ASR_Model
from generateFiles import create_word_document,create_brf_file
from pybraille import pybrl as brl

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
AUDIODIR = 'audio_cache'

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
        brltext = brl.translate(transcription) 
        brltext = brl.toUnicodeSymbols(brltext, flatten=True)

        # Move the file to the output directory
        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)
        
        # Generate document
        docx_filename = os.path.join(OUTPUTDIR, os.path.splitext(os.path.basename(file_path))[0] + '.doc')
        create_word_document(docx_filename, transcription)
        
        print("Transcription:"+ transcription)
        print("Braille:" + brltext)

        os.remove(new_file_path)

        # return FileResponse(
        #     docx_filename,
        #     filename=os.path.basename(docx_filename),
        #     media_type="application/msword"
        # )

        return {"Transcription": transcription, "Braille": brltext}
    
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

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
        brltext = brl.translate(transcripted_text) 
        brltext = brl.toUnicodeSymbols(brltext, flatten=True)

        new_file_path = os.path.join(OUTPUTDIR, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)

        docx_filename = os.path.join(OUTPUTDIR, os.path.splitext(os.path.basename(file_path))[0] + '.doc')
        create_word_document(docx_filename,transcripted_text)

        print("Transcription:"+ transcripted_text)
        print("Braille:" + brltext)

        os.remove(new_file_path)

        # return FileResponse(
        #     docx_filename,
        #     filename=name+'.doc',
        #     media_type="application/msword",
        # )
        return {"Transcription": transcripted_text, "Braille": brltext}
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post('/transcribe/image')
async def transcribe_image(file: UploadFile = File(...)):
    try:
        # Define a directory to save uploaded files
        upload_dir = OUTPUTDIR
      
        name, _ = os.path.splitext(file.filename) 

        # Ensure the directory exists; create it if necessary
        os.makedirs(upload_dir, exist_ok=True)

        # Combine the directory and the file name to get the full file path
        file_path = os.path.join(upload_dir, file.filename)
        
        # Save the uploaded file to the specified directory
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        # Perform transcription using the full file path
        transcripted_text = perform_ocr(file_path,reader)
        #braille_text = alpha_to_braille(transcripted_text)
        
        brltext = brl.translate(transcripted_text) 
        brltext = brl.toUnicodeSymbols(brltext, flatten=True)   

        docx_filename  = upload_dir + name + '.doc'
        #brf_filename  = name +'. brf'

        create_word_document(docx_filename,transcripted_text)
        #create_brf_file(brf_filename,brf_filename)

        # Remove the temporary file
        os.remove(file_path)
        #os.remove(name + '.doc')
        print("Transcription:"+ transcripted_text)
        print("Braille:" +brltext)

        # return FileResponse(
        #     docx_filename,
        #     filename=name+'.doc',
        #     media_type="application/msword",
        # )

        return {"Transcription": transcripted_text, "Braille": brltext}
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing file: {str(e)}")
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")