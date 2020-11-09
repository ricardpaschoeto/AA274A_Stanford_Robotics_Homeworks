#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    downscaled_image = image[1::2, 1::2,:]
    
    return downscaled_image
    ########## Code ends here ##########


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    blur_image = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.7)
    downscaled_image = half_downscale(blur_image)

    return downscaled_image
    ########## Code ends here ##########


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    ########## Code starts here ##########
    upscaled_image = np.repeat(np.repeat(image, repeats=2, axis=0), repeats=2, axis=1)

    return upscaled_image
    ########## Code ends here ##########


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape

    f = (1./scale) * np.convolve(np.ones((scale, )), np.ones((scale, )))
    f = np.expand_dims(f, axis=0) # Making it (1, (2*scale)-1)-shaped
    filt = f.T * f
    
    ########## Code starts here ##########
    Iscaled = np.zeros(((image.shape[0]*scale, image.shape[1]*scale, image.shape[2])))

    Iscaled[::scale,::scale,:] = image

    upscaled_image = cv2.filter2D(Iscaled,-1,filt)

    return upscaled_image
    ########## Code ends here ##########


def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png')[..., ::-1].astype(float)
    favicon = cv2.imread('favicon-16x16.png')[..., ::-1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    ########## Code starts here ##########
    # plt.imshow(test_card)
    # plt.show()
    # img1 = half_downscale(test_card)
    # plt.imshow(img1)
    # plt.show()
    # img2 = half_downscale(img1)
    # plt.imshow(img2)
    # plt.show()
    # img3 = half_downscale(img2)
    # plt.imshow(img3)
    # plt.show()
    ## =========================================##
    # plt.imshow(test_card)
    # plt.show()
    # img1 = blur_half_downscale(test_card)
    # plt.imshow(img1)
    # plt.show()
    # img2 = blur_half_downscale(img1)
    # plt.imshow(img2)
    # plt.show()
    # img3 = blur_half_downscale(img2)
    # plt.imshow(img3)
    # plt.show()
    ## =========================================##
    # plt.imshow(favicon)
    # plt.show()
    # img1 = two_upscale(favicon)
    # plt.imshow(img1)
    # plt.show()
    # img2 = two_upscale(img1)
    # plt.imshow(img2)
    # plt.show()
    # img3 = two_upscale(img2)
    # plt.imshow(img3)
    # plt.show()
    ## =========================================##
    plt.imshow(favicon)
    plt.show()
    img1 = bilinterp_upscale(favicon, 8)
    plt.imshow(img1)
    plt.show()
    ########## Code ends here ##########


if __name__ == '__main__':
    main()
