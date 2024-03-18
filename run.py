# File where to run the model
from scipy.misc import imread, imresize

import numpy as np 
import torch

from dataloader.transformers import test_transform



def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3, device):
    k = beam_size
    vocab_size = len(word_map)
    
    # Read image an process 
    img = imread(image_path)
    if len(img.shape) == 1:
        img = img[:, :, None]
        img = np.concatenate([img, img, img], axis=2)
    
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255 
    img = torch.FloatTensor(img).to(device)
    
    image = test_transform(img)
    
    # Encode 
    image = image.unsqueeze(0)
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    
    # Flatten encoding 
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)
    
    # Tread the problem as having a batch size of k
    # (K, N_pixels, Encoder)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    
    # Tensor to store top k
    
    
    
    
    
    
    # Read image and process 
    img = 
    return seq, alphas


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
