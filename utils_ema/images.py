import numpy as np
import torch
import cv2


def show_image( name, image, wk=0 ):
    if torch.is_tensor(image):
        image = image.numpy()
    if wk is not None:
        cv2.imshow( name, image )
        cv2.waitKey(wk)

def sample_from_image_pdf( probabilities, num_samples ):

    if( torch.is_tensor(probabilities) ):
        probabilities = probabilities.numpy()

    flattened_probs = probabilities.flatten()
    remaining_probs = flattened_probs.copy()
    sampled_pixels = []
    
    for _ in range(num_samples):
        normalized_probs = remaining_probs / np.sum(remaining_probs)
        cdf = np.cumsum(normalized_probs)

        random_num = np.random.random()
        sampled_pixel_flat = np.searchsorted(cdf, random_num, side='right')
        sampled_pixel = np.unravel_index(sampled_pixel_flat, probabilities.shape)
        sampled_pixels.append(sampled_pixel)
        
        # Remove the selected pixel's probability
        remaining_probs[sampled_pixel_flat] = 0
    
    return sampled_pixels
