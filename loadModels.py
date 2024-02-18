import easyocr
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR


#To remove GPU Util remove gpu=True and run_opts = {"device":"cuda"}

def OCR_Model():

    # Create an OCR reader with the specified language (Pretrained model) 
    #reader =  easyocr.Reader(['en'],gpu=True) 

    # Create an OCR reader with the specified language (custom recognition model)
    custom_model_path = 'ocr_finetuned'

    reader =  easyocr.Reader(['en'],recog_network='fine_tuned_recognition',model_storage_directory=custom_model_path,user_network_directory=custom_model_path,gpu=True) 

    return reader

def ASR_Model():
    asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech",
    savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
    run_opts = {"device": "cuda"}
)
    return asr_model