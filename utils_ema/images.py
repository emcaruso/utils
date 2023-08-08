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

    # Generate random numbers for all samples at once
    random_nums = np.random.random(num_samples)
    
    # Calculate CDF for the remaining probabilities
    normalized_probs = flattened_probs / np.sum(flattened_probs)
    cdf = np.cumsum(normalized_probs)
    
    # Find the indices of the sampled pixels
    sampled_pixel_indices = np.searchsorted(cdf, random_nums, side='left')
    sampled_pixel_indices = np.clip(sampled_pixel_indices, 0, len(cdf)-1)
    
    # Convert flattened indices to 2D indices
    x, y = np.unravel_index(sampled_pixel_indices, probabilities.shape)

    sampled_indices = np.stack((x, y), axis=-1)

    return torch.LongTensor(sampled_indices)
    

def show_pixs( pixs, shape, wk=0 ):
    img = np.zeros( shape )
    img[pixs[:,0],pixs[:,1]]=1
    if wk is not None:
        cv2.imshow("shown pixs", img)
        cv2.waitKey(wk)

