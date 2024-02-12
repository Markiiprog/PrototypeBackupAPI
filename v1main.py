import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.pretrained import EncoderDecoderASR
from preprocess import MP32Wav, Video2Wav
from OCR_Predict import Predict
from alphaToBraille import translate as alpha_to_braille
from brailleToAlpha import translate as braille_to_alpha
from postProcess import perform_spell_check, perform_punctuation_check

app = FastAPI()

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

        # Convert transcription to Braille
        braille_text = alpha_to_braille(transcription)

        # Remove the temporary files
        os.remove(temp_filepath)
        os.remove(file_path)

        return {"Transcription": transcription, "Braille": braille_text}

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

        # Transcribe the video using the ASR model
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
        )
        transcripted_text = asr_model.transcribe_file(file_path)

        # Convert transcription to Braille
        braille_text = alpha_to_braille(transcripted_text)

        # Remove the temporary files
        os.remove(temp_filepath)
        os.remove(file_path)

        return {"Transcription": transcripted_text, "Braille": braille_text}

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post('/transcribe/image')
async def transcribe_image(file: UploadFile = File(...)):
    try:
        upload_dir = 'path/to/uploaded/files'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, 'wb') as file_output:
            file_output.write(file.file.read())

        # Perform OCR on the image
        transcripted_text = Predict(file_path)

        # Perform spell check
        transcripted_text = perform_spell_check(transcripted_text)

        # Perform punctuation check
        transcripted_text = perform_punctuation_check(transcripted_text)

        # Convert transcription to Braille
        braille_text = alpha_to_braille(transcripted_text)

        # Remove the temporary file
        os.remove(file_path)

        return {"Transcription": transcripted_text, "Braille": braille_text}

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
