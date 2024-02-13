import speechbrain as sb
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

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

@app.get('/')
async def root():
    return {
        'ASR API': 'Active'
    }

@app.post(f'/transcribe/audio')
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the audios directory
        file_path = file.filename
        temp_filepath = file_path

        upload_dir = OUTPUTDIR
      
        name, _ = os.path.splitext(file.filename) 
        
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        file_name, file_extension = os.path.splitext(file_path)

        if file_extension == ".mp3":
           file_path = MP32Wav(file_path, "audios", f"{file_name}.wav")
        else:
            return {"error": "Unsupported file type"}
        
        transcription = asr_model.transcribe_file(file_path)
        brltext = brl.translate(transcription) 
        brltext = brl.toUnicodeSymbols(brltext, flatten=True)

        docx_filename  = upload_dir + name + '.doc'
        create_word_document(docx_filename,transcription)
        # Remove the temporary filez
        os.remove(temp_filepath)
        os.remove(file_path)
        
        print("Transcription:"+ transcription)
        print("Braille:" +brltext)

        # return FileResponse(
        #     docx_filename,
        #     filename=name+'.doc',
        #     media_type="application/msword",
        # )

        return {"Transcription": transcription, "Braille": brltext}
    
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post(f'/transcribe/video')
async def transcribe_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the audios directory
        file_path = file.filename
        temp_filepath = file_path
        
        upload_dir = OUTPUTDIR
      
        name, _ = os.path.splitext(file.filename) 


        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        file_name, file_extension = os.path.splitext(file_path)
        if file_extension == ".mp4":
            file_path = Video2Wav(file_path, "audios", f"{file_name}.wav")
        else:
            return {"error": "Unsupported file type"}
        
        transcripted_text = asr_model.transcribe_file(file_path)
        brltext = brl.translate(transcripted_text) 
        brltext = brl.toUnicodeSymbols(brltext, flatten=True)

        # docx_filename  = upload_dir + name + '.doc'
        # create_word_document(docx_filename,transcripted_text)
        # Remove the temporary file
        os.remove(temp_filepath)
        os.remove(file_path)

        print("Transcription:"+ transcripted_text)
        print("Braille:" + brltext)

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