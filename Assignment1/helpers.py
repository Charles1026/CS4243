import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""" Util Functions """""""""""""""""""""""

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)


def plotAndSaveHistogram(img: np.ndarray, title: str, path = None):
    fig, axs = plt.subplots(1, 1)
    axs.hist(img.flatten(), 256, [0, 256], color = 'r')
    if (title != None):
        axs.set_title(title)
    
    if path != None:
        plt.savefig(path + title + ".png")
    else:
        plt.savefig(title + ".png")
    
def plotAndSaveHistogramWithAxesLim(img: np.ndarray, title: str, path = None, xLim: int = None, yLim: int = None):
    fig, axs = plt.subplots(1, 1)
    axs.hist(img.flatten(), 256, [0, 256], color = 'r')
    
    if yLim != None and yLim > 0:
        axs.set_ylim(0, yLim)
        
    if xLim != None and xLim > 0:
        axs.set_xlim(0, xLim)
    
    if (title != None):
        axs.set_title(title)
    
    if path != None:
        plt.savefig(path + title + ".png")
    else:
        plt.savefig(title + ".png")