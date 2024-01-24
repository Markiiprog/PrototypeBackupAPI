# main.py

import os
from sys import argv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.pretrained import EncoderDecoderASR
from preprocess import MP32Wav, Video2Wav
from PredictImages import predict
import alphaToBraille
import brailleToAlpha

app = FastAPI(openapi_url="/api/v1/openapi.json", docs_url="/api/v1/docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIODIR = 'audios/'

def user_braille(braille_text):
    return brailleToAlpha.translate(braille_text)

def user_text(text):
    return alphaToBraille.translate(text)

def transcribe_audio(file_path):
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
    )
    transcripted_text = asr_model.transcribe_file(file_path)
    braille_text = user_text(transcripted_text)
    return {"Transcription": transcripted_text, "Braille": braille_text}

def transcribe_video(file_path):
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
    )
    transcripted_text = asr_model.transcribe_file(file_path)
    braille_text = user_text(transcripted_text)
    return {"Transcription": transcripted_text, "Braille": braille_text}

def transcribe_image(file_path):
    transcripted_text = predict(file_path)
    braille_text = user_text(transcripted_text)
    return {"Transcription": transcripted_text, "Braille": braille_text}

@app.get('/')
async def root():
    return {
        'ASR API': 'Active'
    }

@app.post(f'/transcribe/audio')
async def transcribe_audio_api(file: UploadFile = File(...)):
    try:
        file_path = file.filename
        temp_filepath = file_path
        
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        file_name, file_extension = os.path.splitext(file_path)
        if file_extension == ".mp3":
            file_path = MP32Wav(file_path, "audios", f"{file_name}.wav")
        else:
            return {"error": "Unsupported file type"}
        
        result = transcribe_audio(file_path)

        os.remove(temp_filepath)
        os.remove(file_path)

        return result

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post(f'/transcribe/video')
async def transcribe_video_api(file: UploadFile = File(...)):
    try:
        file_path = file.filename
        temp_filepath = file_path
        
        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        file_name, file_extension = os.path.splitext(file_path)
        if file_extension == ".mp4":
            file_path = Video2Wav(file_path, "audios", f"{file_name}.wav")
        else:
            return {"error": "Unsupported file type"}

        result = transcribe_video(file_path)

        os.remove(temp_filepath)
        os.remove(file_path)

        return result

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post('/transcribe/image')
async def transcribe_image_api(file: UploadFile = File(...)):
    try:
        upload_dir = 'path/to/uploaded/files'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        result = transcribe_image(file_path)

        os.remove(file_path)

        return result

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
