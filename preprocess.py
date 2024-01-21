
import moviepy.editor
import os
import torchaudio
from torchaudio.transforms import Resample
import librosa
import soundfile as sf


def ExtractAudio(video_file_path):
  file_name, _ = os.path.splitext(video_file_path)

  video = moviepy.editor.VideoFileClip(file_name + ".mp4")
  audio = video.audio
  audiofilename = file_name + ".mp3"
  audio.write_audiofile(audiofilename)


  return audiofilename

def convert_audio(input_path, output_path, target_sr=16000):
    # Load the audio file using librosa
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # Export the processed audio to a WAV file using soundfile
    sf.write(output_path, audio, target_sr)

def preprocess_for_asr(input_path, output_path):
    # Load the WAV file using torchaudio
    waveform, sample_rate = torchaudio.load(input_path)

    # Check if the waveform is loaded successfully
    if waveform.numel() == 0:
        print("Error: Unable to load audio file.")
        return

    # Save the preprocessed audio to a new WAV file in the same directory
    torchaudio.save(output_path, waveform, sample_rate)

def Video2Wav(input,output_dir,output_filename): #Converting Video(MP4) to .WAV
  temp_ext_mp3 = ExtractAudio(input)
  temp_wav_path = os.path.join(output_dir, "temp_audio.wav")
 
  convert_audio(temp_ext_mp3, temp_wav_path)

  preprocess_for_asr(temp_wav_path, output_filename)

  os.remove(temp_wav_path)
  os.remove(temp_ext_mp3)

  return output_filename

def MP32Wav(input,output_dir,output_filename): #Converting .MP3 to .WAV

  temp_wav_path = os.path.join(output_dir, "temp_audio.wav")

  convert_audio(input, temp_wav_path)

  preprocess_for_asr(temp_wav_path, output_filename)
  
  os.remove(temp_wav_path)

  return output_filename


