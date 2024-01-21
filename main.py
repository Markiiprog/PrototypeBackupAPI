import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from preprocess import MP32Wav, Video2Wav
from fastapi.middleware.cors import CORSMiddleware
from PredictImages import predict


from datetime import datetime


app = FastAPI()  #uvicorn main:app --reload (This runs starts a local instance of the 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIODIR = 'audios/'

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
        
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        file_name, file_extension = os.path.splitext(file_path)

        if file_extension == ".mp3":
           file_path = MP32Wav(file_path, "audios", f"{file_name}.wav")
        else:
            return {"error": "Unsupported file type"}
        
        # Transcribe the audio using the ASR model
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
        )
        transcription = asr_model.transcribe_file(file_path)
        transcripted_text = asr_model.transcribe_file(file_path)

        # Remove the temporary file
        os.remove(temp_filepath)
        os.remove(file_path)

        return {"Transcription": transcripted_text}

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post(f'/transcribe/video')
async def transcribe_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the audios directory
        file_path = file.filename
        temp_filepath = file_path
        
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        file_name, file_extension = os.path.splitext(file_path)
        if file_extension == ".mp4":
            file_path = Video2Wav(file_path, "audios", f"{file_name}.wav")
        else:
            return {"error": "Unsupported file type"}

        # Transcribe the audio using the ASR model
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
        )
        transcripted_text = asr_model.transcribe_file(file_path)

        # Remove the temporary file
        os.remove(temp_filepath)
        os.remove(file_path)

        return {"Transcription": transcripted_text}

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post('/transcribe/image')
async def transcribe_image(file: UploadFile = File(...)):
    try:
        # Define a directory to save uploaded files
        upload_dir = 'path/to/uploaded/files'
        
        # Ensure the directory exists; create it if necessary
        os.makedirs(upload_dir, exist_ok=True)

        # Combine the directory and the file name to get the full file path
        file_path = os.path.join(upload_dir, file.filename)

        # Save the uploaded file to the specified directory
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        # Perform transcription using the full file path
        transcripted_text = predict(file_path)

        # Remove the temporary file
        os.remove(file_path)

        return {"Transcription": transcripted_text}

    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing file: {str(e)}")
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")