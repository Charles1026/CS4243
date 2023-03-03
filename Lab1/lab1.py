""" CS4243 Lab 1: Template Matching
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: Image Preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    """ Your code starts here """
    def rbgCell2GrayCell(cell : np.array(int)) -> float:
        if cell.shape[0] != 3:
            print('RGB Cell should have 3 channels')
            return -1
        for value in cell:
            if value < 0 or value > 255:
                print('RGB Value should be >=0 and <=255')
                return -1
        return (0.299 * cell[0]) + (0.587 * cell[1]) + (0.114 * cell[2])
    
    img_gray = np.array([[rbgCell2GrayCell(cell) for cell in row] for row in img], dtype = float)
    
    """ Your code ends here """
    return img_gray


def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    

    """ Your code starts here """ 
    
    if len(img.shape) != 2:
        print(gray2grad.func_name + ": only works with 2 dimensional arrays.")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    def flip2DArray(array):
        return [row[::-1] for row in array[::-1]]

    def convolve(imgPiece : np.array(float), reversedFilter : np.array(float)) -> float:
        if (imgPiece.shape != reversedFilter.shape):
            print("Img Piece and filter must be same size to convolve")
            return 0
        
        sum : float = 0
        for i in range(imgPiece.shape[0]):
            for j in range(imgPiece.shape[1]):
                sum += imgPiece[i][j] * reversedFilter[i][j]
        return sum
    
    
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    
    def convolveWithFilter(filter : np.array(float)) -> np.array(float): 
        filterHeight = filter.shape[0]
        filterWidth = filter.shape[1]
        
        paddedImg = pad_zeros(img, math.floor(filterHeight / 2), math.floor(filterHeight / 2), 
                math.floor(filterWidth / 2), math.floor(filterWidth / 2))
        
        returnImg = np.empty((imgHeight, imgWidth), dtype = float)
        
        if imgHeight < filterHeight or imgWidth < filterWidth:
            print(gray2grad.func_name + ": input array must at least be of size " + filterHeight + "x" + filterWidth)
            return returnImg

        for i in range(imgHeight):
            for j in range(imgWidth):
                imgPiece = np.array(paddedImg[i:(i + filterHeight), j:(j + filterWidth)], dtype=float)
                convolved = convolve(imgPiece, filter)
                returnImg[i][j] = convolved 

        return returnImg
    
    flippedSobelh = np.array(flip2DArray(sobelh), dtype=float)
    flippedSobelv = np.array(flip2DArray(sobelv), dtype=float)
    flippedSobeld1 = np.array(flip2DArray(sobeld1), dtype=float)
    flippedSobeld2 = np.array(flip2DArray(sobeld2), dtype=float)
    
    img_grad_h = convolveWithFilter(flippedSobelh)
    img_grad_v = convolveWithFilter(flippedSobelv)
    img_grad_d1 = convolveWithFilter(flippedSobeld1)
    img_grad_d2 = convolveWithFilter(flippedSobeld2)

    """ Your code ends here """
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    """ Your code starts here """
    img_pad[pad_height_bef:pad_height_bef + height, pad_width_bef:pad_width_bef + width] = img
    img_pad = img_pad.astype(img.dtype)
    
    """ Your code ends here """
    return img_pad




##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 4, 5 or 6 nested for-loops. 
    The loops should at least include the height and width of the output and height and width of the template.
    When it is 5 or 6 loops, the channel of the output and template may be included.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    response = np.empty((Ho, Wo), dtype = float)
    
    if len(img.shape) != len(template.shape):
        print("Image and Template of different dimensions")
        return response
    # iterate through the output image size
    for outI in range(Ho):
        for outJ in range(Wo):
            squaredTemplateMagnitude : float = 0.0
            squaredImageMagnitude : float = 0.0
            correlation :float = 0.0
            # Iterate through template
            for iterI in range(Hk):
                for iterJ in range(Wk):
                    # If rgb channels 
                    if len(img.shape) == 3 and img.shape[2] == template.shape[2]:
                        for rgb in range(img.shape[2]):
                            templateValue : float = float(template[iterI, iterJ, rgb])
                            imgValue : float = float(img[outI + iterI, outJ + iterJ, rgb])
                            correlation += templateValue * imgValue
                            squaredTemplateMagnitude += templateValue ** 2
                            squaredImageMagnitude += imgValue ** 2
                    # Grayscale image
                    elif len(img.shape) == 2:
                        templateValue : float = float(template[iterI, iterJ])
                        imgValue : float = float(img[outI + iterI, outJ + iterJ])
                        correlation += templateValue * imgValue
                        squaredTemplateMagnitude += templateValue ** 2
                        squaredImageMagnitude += imgValue ** 2
                    else:
                        print("Error with cell: ", outI + iterI, " + ", outJ + iterJ)
            # sqrt first to prevent overflow
            templateMagnitude = math.sqrt(squaredTemplateMagnitude)
            imageMagnitude = math.sqrt(squaredImageMagnitude)
            denominator = templateMagnitude * imageMagnitude
            normalisedCorrelation = correlation / denominator
            response[outI, outJ] = normalisedCorrelation
    """ Your code ends here """
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    response = np.empty((Ho, Wo), dtype = float)
    
    if len(img.shape) != len(template.shape):
        print("Image and Template of different dimensions")
        return response
    
    # convert template to float type
    floatTemplate = template.astype(float)
    floatImg = img.astype(float)
    
    # iterate through the output image size
    for outI in range(Ho):
        for outJ in range(Wo):
            floatWindow = floatImg[outI : outI + Hk, outJ : outJ + Wk]
            
            crossCorrelatedValue      : float = np.sum(np.multiply(floatWindow, floatTemplate))
            tempalateSquaredMagnitude : float = np.sum(np.multiply(floatTemplate, floatTemplate))
            windowSquaredMagnitude    : float = np.sum(np.multiply(floatWindow, floatWindow))
        
            templateMagnitude : float = math.sqrt(tempalateSquaredMagnitude)
            windowMagnitude   : float = math.sqrt(windowSquaredMagnitude)
            
            response[outI, outJ] = crossCorrelatedValue / (templateMagnitude * windowMagnitude)
    """ Your code ends here """
    return response


def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    response = np.zeros((Ho, Wo), dtype = float)
    
    if len(img.shape) != len(template.shape):
        print("Image and Template of different dimensions")
        return response
    
    # convert template to float type
    floatTemplate = template.astype(float)
    floatImg = img.astype(float)
    channels : int = floatTemplate.shape[2] if len(floatTemplate.shape) == 3 else 1
    
    # Only works for arrays of size Hk x Wk
    def flattenArray(array, asRow : bool):
        # Reshaped to Hk * Wk X channels, collapsed all pixels into 1 row
        tempMatrix = np.reshape(array, (Hk * Wk, channels), order="C")
        # Reshaped to Hk * Wk * channels X 1, 1D array sorted by channel then pixel order
        if asRow:
            return np.reshape(tempMatrix, (1, Hk * Wk * channels), order="F")
        else:
            return np.reshape(tempMatrix, (Hk * Wk * channels, 1), order="F")
            

    def arrayMagnitudeSquared(array, isRow) -> float:
        transposedArray = np.reshape(array,(array.shape[1], array.shape[0]))
        if isRow:
            return np.dot(array, transposedArray)[0,0]
        else:
            return np.dot(transposedArray, array)[0,0]

    reshapedTemplateMatrix = flattenArray(floatTemplate, False)

    reshapedImageList = []
    windowMagnitudeMatrix = np.empty((Ho, Wo))
    for outI in range(Ho):
        for outJ in range(Wo):
            # Get reshaped window
            floatWindow = floatImg[outI : outI + Hk, outJ : outJ + Wk]
            reshapedWindow = flattenArray(floatWindow, True)
            
            # Add window magnitude
            windowMagnitudeMatrix[outI, outJ] = math.sqrt(arrayMagnitudeSquared(reshapedWindow, True))
            
            # Append window to reshaped image
            reshapedImageList.append(reshapedWindow)
    # Convert image list to np array
    reshapedImageMatrix = np.array(reshapedImageList, dtype = float)
    
    reshapedCorrelationMatrix = np.reshape(np.matmul(reshapedImageMatrix, reshapedTemplateMatrix, dtype = float), (Ho, Wo))
    
    templateMagnitude = math.sqrt(arrayMagnitudeSquared(reshapedTemplateMatrix, False))
    
    divisor = windowMagnitudeMatrix * templateMagnitude
    response = reshapedCorrelationMatrix / divisor

    """ Your code ends here """
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    
    """ Your code starts here """
    res = np.copy(response)
    height = res.shape[0]
    width  = res.shape[1]
    threshold = 0 if threshold == None else threshold
    windowHalfHeight = suppress_range[0]
    windowHalfWidth = suppress_range[1]
    
    # high pass filter
    for i in range(height):
        for j in range(width):
            value = res[i, j]
            res[i, j] = 0 if value < threshold else value
    
    hasNonZeroVariable = True
    localMaximaList = []
    while hasNonZeroVariable:
        # Reset break condition
        hasNonZeroVariable = False
        # value, x coord, y coord
        globalMaxima : tuple(float, int, int) = (0.0, 0, 0)
        for i in range(height):
            for j in range(width):
                value = res[i, j]
                # check break condition
                hasNonZeroVariable = value > 0 or hasNonZeroVariable
                # Find global maxima
                if (value > globalMaxima[0]):
                    globalMaxima = (value, i, j)
        localMaximaList.append(globalMaxima)
        
        # Set area around global maxima to 0
        iMin = max(0, globalMaxima[1] - windowHalfHeight)
        iMax = min(res.shape[0], globalMaxima[1] + windowHalfHeight)
        jMin = max(0, globalMaxima[2] - windowHalfWidth)
        jMax = min(res.shape[1], globalMaxima[2] + windowHalfWidth)
        
        res[iMin : iMax, jMin : jMax] = np.zeros((iMax - iMin, jMax - jMin))
        
    for point in localMaximaList:
        res[point[1], point[2]] = 255.0
    
    """ Your code ends here """
    return res

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    response = np.zeros((Ho, Wo), dtype = float)
    
    if len(img.shape) != len(template.shape):
        print("Image and Template of different dimensions")
        return response
    
    # convert template to float type
    floatTemplate = template.astype(float)
    floatImg = img.astype(float)
    
    def zeroCenterArray(array):
        zeroCenteredArray = np.empty(array.shape, dtype = float)
        if len(array.shape) == 3:
            arraySumList = [np.mean(array[:, :, channel]) for channel in range(array.shape[2])]
            zeroCenteredArray = array - np.array(arraySumList)
        else:
            zeroCenteredArray = array - np.mean(array)
        return zeroCenteredArray
    
    # zero centering the template
    zeroCenteredTemplate = zeroCenterArray(floatTemplate)
    
    # Finding Magnitude of Template
    zeroCenteredTemplateMagnitude = math.sqrt(np.sum(zeroCenteredTemplate ** 2))
    
    # iterate through the output image size
    for outI in range(Ho):
        for outJ in range(Wo):
            floatWindow = floatImg[outI : outI + Hk, outJ : outJ + Wk]
            zeroCenteredWindow = zeroCenterArray(floatWindow)
            crossCorrelatedValue = np.sum(zeroCenteredWindow * zeroCenteredTemplate)
            
            zeroCenteredWindowMagnitude = math.sqrt(np.sum(zeroCenteredWindow ** 2))
            
            response[outI, outJ] = crossCorrelatedValue / (zeroCenteredTemplateMagnitude * zeroCenteredWindowMagnitude)
            
    
    """ Your code ends here """
    return response




"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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

def show_img_with_points(response, img_ori=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.circle(response, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        if img_ori is not None:
            img_ori = cv2.circle(img_ori, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)


