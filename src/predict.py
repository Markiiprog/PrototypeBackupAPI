import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

from .config import common_config as config
from .dataset import Synth90kDataset, synth90k_collate_fn
from .ctc_decoder import ctc_decode

def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

    return all_preds

def show_result(paths, preds):
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
    return text

def perform_prediction(images, crnn, decode_method, beam_size):
    img_height = config['img_height']
    img_width = config['img_width']

    predict_dataset = Synth90kDataset(paths=images,
                                      img_height=img_height, img_width=img_width)

    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=len(images),  # Process all images at once
        shuffle=False)

    preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size)

    return show_result(images, preds)
