#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    k , ell, c = F.shape[0],F.shape[1],F.shape[2]
    f = np.reshape(F.flatten('F'), (1,k*ell*c))
    v_pad = (k//2)
    h_pad = (ell//2)
    Ipad = np.pad(I,((v_pad,h_pad),(v_pad,h_pad),(0,0)), mode='constant', constant_values=0)
    G = np.zeros((I.shape[0],I.shape[1]))

    for ii in range(I.shape[0]):
        for jj in range(I.shape[1]):
            try:
                tij = np.reshape(Ipad[ii:ii+k,jj:jj+ell,:].flatten('F'), (k*ell*c,1))
                G[ii,jj] = np.dot(f,tij)
            except :
                break           

    return G
    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    k , ell, c = F.shape[0],F.shape[1],F.shape[2]
    f = np.reshape(F.flatten('F'), (1,k*ell*c))
    f_T = f.T
    G_corr = corr(F, I)
    
    v_pad = (k//2)
    h_pad = (ell//2)
    Ipad = np.pad(I,((v_pad,h_pad),(v_pad,h_pad),(0,0)), mode='constant', constant_values=0)
    G = np.zeros(G_corr.shape)
    for ii in range(G_corr.shape[0]):
        for jj in range(G_corr.shape[1]):
            try:
                tij = np.reshape(Ipad[ii:ii+k,jj:jj+ell,:].flatten('F'), (k*ell*c,1))
                G[ii,jj] = G_corr[ii,jj]/(np.linalg.norm(f_T)*np.linalg.norm(tij))
            except :
                break

    return G
    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    ### Begin My test ###
    # filt = list()

    # f1 = np.array([[0,0,0],[0,1,0],[0,0,0]])
    # f1 = np.expand_dims(f1, -1)
    
    # f2 = np.array([[0,0,0],[0,0,1],[0,0,0]])
    # f2 = np.expand_dims(f2, -1)
    
    # f3 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # f3 = np.expand_dims(f3, -1)
    
    # f4 = np.round(np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32)/16, decimals=4)
    # f4 = np.expand_dims(f4, -1)
    
    # filt = np.concatenate([f4,f4,f4], axis = -1)

    # I = np.zeros((3,3,3))
    # img = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # I[:,:,0] = img
    # I[:,:,1] = img
    # I[:,:,2] = img
    
    # corr_img = corr(filt,I)
    # plt.imshow(corr_img, interpolation='none')
    ### End My test ####

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    plt.imshow(test_card, interpolation='none')
    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        #norm_img = norm_cross_corr(filt,test_card)
        stop = time.time()
        print 'Correlation function runtime:', stop - start, 's'
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
