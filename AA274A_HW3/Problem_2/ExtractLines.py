#!/usr/bin/env python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-slitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        startIdx: starting index of segment to be split.
        endIdx: ending index of segment to be split.
        params: dictionary of parameters.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    
    alpha_, r_ = FitLine(theta[startIdx:endIdx],rho[startIdx:endIdx])

    alpha = [alpha_]
    r = [r_]
    idx = [(startIdx, endIdx)]
    
    if (endIdx - startIdx) <= params['MIN_POINTS_PER_SEGMENT']:
        return np.asarray(alpha), np.asarray(r), np.asarray(idx)
    
    s = FindSplit(theta[startIdx:endIdx],rho[startIdx:endIdx],alpha_,r_,params)
    if s == -1:
        return np.asarray(alpha), np.asarray(r), np.asarray(idx)
    
    alpha1, r1, i1 = SplitLinesRecursive(theta,rho,startIdx,startIdx+s, params)
    alpha2, r2, i2 = SplitLinesRecursive(theta,rho,startIdx + s,endIdx, params)

    return np.concatenate([alpha1,alpha2]), np.concatenate([r1,r2]), np.concatenate([i1,i2])

def FindSplit(theta, rho, alpha, r, params):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        params: dictionary of parameters.
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).
    '''
    d = rho*np.cos(theta - alpha) - r
    d_list = d.tolist()
    n = len(d_list)

    while(n > 0):
        farthest = max(d_list, key=abs)
        splitIdx = np.where(d == farthest)[0][0]
        n_pts_seg1 = len(theta[:splitIdx])
        n_pts_seg2 = len(theta[splitIdx:])
        
        if np.abs(d[splitIdx]) > params['LINE_POINT_DIST_THRESHOLD'] and n_pts_seg1 >= params['MIN_POINTS_PER_SEGMENT'] and n_pts_seg2 >= params['MIN_POINTS_PER_SEGMENT']:
            return splitIdx            
        else:
            d_list.remove(farthest)
            n = len(d_list)            
    
    return -1

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads).
        r: 'r' of best fit for range data (1 number) (m).
    '''
    num_sum1 = 0
    num_sum2 = 0
    den_sum1 = 0
    den_sum2 = 0
    sum_r = 0
    n = len(rho)
    for ii in range(n):
        num_sum1 += (rho[ii]**2)*np.sin(2*theta[ii])
        den_sum1 += (rho[ii]**2)*np.cos(2*theta[ii])
    
    for ii in range(n):
        for jj in range(n):
            num_sum2 += rho[ii]*rho[jj]*np.cos(theta[ii])*np.sin(theta[jj])
            den_sum2 += rho[ii]*rho[jj]*np.cos(theta[ii] + theta[jj])
        
    alpha = (0.5)*np.arctan2((num_sum1 - 2*num_sum2/n),(den_sum1 - den_sum2/n)) + np.pi/2
    
    for ii in range(n):
        sum_r += rho[ii]*np.cos(theta[ii] - alpha)
    
    r = sum_r/n

    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########
    alphaOut = []
    rOut = []
    pointIdxOut = []
    for ii in range(len(pointIdx)-1):
        pt1 = pointIdx[ii]
        pt2 = pointIdx[ii+1]
        
        start = pt1[0]
        end = pt2[1]
        
        alpha_, r_ = FitLine(theta[start:end], rho[start:end])
        split = FindSplit(theta[start:end], rho[start:end], alpha_, r_, params)
        
        if split == -1:
            alphaOut.append(alpha_)
            rOut.append(r_)
            pointIdxOut.append((start,end))
        else:
            return alpha, r, pointIdx
        
    return alphaOut, rOut, pointIdxOut
        
    # # ########## Code ends here ##########

#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 2  # minimum number of points per line segment
    MAX_P2P_DIST = 0.25  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    # filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    # filename = 'rangeData_7_2_90.csv'
    
    files = ['rangeData_5_5_180.csv', 'rangeData_4_9_360.csv', 'rangeData_7_2_90.csv']

    # Import Range Data
    
    for filename in files:
        RangeData = ImportRangeData(filename)
    
        params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
                  'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
                  'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
                  'MAX_P2P_DIST': MAX_P2P_DIST}
    
        alpha, r, segend, pointIdx = ExtractLines(RangeData, params)
    
        ax = PlotScene()
        ax = PlotData(RangeData, ax)
        ax = PlotRays(RangeData, ax)
        ax = PlotLines(segend, ax)
    
        plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
