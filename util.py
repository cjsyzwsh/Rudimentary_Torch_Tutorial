import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# simulate data sets.
def gen_multiple_choice_data(N, D, K, b):
    '''
    gen choice model data for simulation
        D is the dim of X; 
        b: D * K, with the first column equals to zero
    Outputs:
        Choice: k = 0, 1, ... K - 1 (total K)
        X: N * D
    '''
#    np.random.seed(np.random.choice(N))
    X = np.random.normal(0,1, (N,D))
    e = np.random.gumbel(0,1, (N,K))

    Xb = np.dot(X, b)
#    Xb_with_K_classes = np.concatenate((np.zeros((N, 1)), Xb), axis = 1)
    Xb = Xb + e # add noise
    y = np.argmax(Xb, axis = 1)
    return y, X


# imshow
def imshow(image, *args, **kwargs):
    if len(image.shape) == 3:
      # Height, width, channels
      # Assume BGR, do a conversion since 
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
      # Height, width - must be grayscale
      # convert to RGB, since matplotlib will plot in a weird colormap (instead of black = 0, white = 1)
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Draw the image
    plt.imshow(image, *args, **kwargs)
    # We'll also disable drawing the axes and tick marks in the plot, since it's actually an image
    plt.axis('off')
    # Make sure it outputs
    plt.show()
