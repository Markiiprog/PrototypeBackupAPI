import torch

from src.config import common_config as config
from src.dataset import Synth90kDataset, synth90k_collate_fn
from src.model import CRNN

img_height = config['img_height']
img_width = config['img_width']

def LoadModel():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = len(Synth90kDataset.LABEL2CHAR) + 1

    # Load your CRNN model here if not already loaded
    crnn = CRNN(1, img_height, img_width, num_class, map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'], leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load('checkpoints/crnn_synth90k.pt', map_location=device))
    crnn.to(device)

    return crnn