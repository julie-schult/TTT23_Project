
import cv2 as cv
import numpy as np
def paddImage(img, new_shape):
    """
    Cuts or pads the image with zeros to fit the shape new_shape

    img: np grayscale image
    new_shape: tuple with two values (height, width)
    
    return: an image with size new_shape
    """
    if img.shape[0] > new_shape[0]:
        img = img[:new_shape[0]]
    if img.shape[1] > new_shape[1]:
        img = img[:, :new_shape[1]]
    
    bottom = new_shape[0] - img.shape[0]
    right = new_shape[1] - img.shape[1]

    return cv.copyMakeBorder(img, 0, bottom, 0, right, cv.BORDER_CONSTANT)

def setGrayToBlack(img, threshold=150):
    """
    Sets the gray values of img to black. 

    img: Needs to be a bgr numpy array
    threshold: value between 0-255, saturation values under this value gets set to black    
    """

    h, s, v = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV)) #Convert image to hsv
    gray_mask = s < threshold # Set a binary mask where saturation is under threshold
    v[gray_mask] = 0 # Set the value to 0 (set to black) where the mask is True
    out_image = cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2BGR) #Convert back to bgr
    return out_image



