import easyocr
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR

def OCR_Model():

    # Create an OCR reader with the specified language (Pretrained model) 
    reader =  easyocr.Reader(['en']) 

    # Create an OCR reader with the specified language (custom recognition model)
    #custom_model_path = 'custom_model3'
    #reader =  easyocr.Reader(['en'],recog_network='best_accuracy',model_storage_directory=custom_model_path,user_network_directory=custom_model_path) 

    return reader

def ASR_Model():
    asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech",
    savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
)
    return asr_model