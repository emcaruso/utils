import numpy as np
import torch


def show_image( name, image, wk=0 ):
    if torch.is_tensor(image):
        image = image.numpy()
    if wk is not None:
        cv2.imshow( name, image )
        cv2.waitKey(wk)

def sample_from_image_pdf( probabilities, num_samples ):

    if( torch.is_tensor(image_prob) ):
        imabe_prob = image_prob.numpy()

    flattened_probs = probabilities.flatten()
    remaining_probs = flattened_probs.copy()
    sampled_pixels = []
    
    for _ in range(num_samples):
        normalized_probs = remaining_probs / np.sum(remaining_probs)
        cdf = np.cumsum(normalized_probs)
        
        random_num = np.random.random()
        sampled_pixel_index = np.searchsorted(cdf, random_num, side='right')
        sampled_pixels.append(sampled_pixel_index)
        
        # Remove the selected pixel's probability
        remaining_probs[sampled_pixel_index] = 0
    
    sampled_pixels = np.array(sampled_pixels).reshape(probabilities.shape)
    return sampled_pixels
