import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize_coords(pts: np.ndarray):
    """
    calculates normalization of point correspondence matching following term:
        1. centroid of the reference points is at the origin
        2. root mean square difference from origin is sqrt(2)
    finds a transformation matrix that performs the normalization: norm_x = T @ x
    :param pts: array representing pts shape (2, num_pts)
    :return: normalized pt_correspondences, normalization matrix T
    """
    assert pts.shape[0] == 2
    # pad to homogeneous coordinates
    pts = np.concatenate([pts, np.ones((1, pts.shape[1]))])

    # get centroid
    centroid = np.mean(pts, 1)

    # compute the RMS distance to centroid
    RMS = np.sqrt(np.sum(np.mean((pts - centroid.reshape(3, 1)) ** 2, axis=1)))

    # calculate matrix
    T = np.asarray([[np.sqrt(2) / RMS, 0, - centroid[0] * np.sqrt(2) / RMS],
                    [0, np.sqrt(2) / RMS, - centroid[1] * np.sqrt(2) / RMS],
                    [0, 0, 1]]).reshape(3, 3)

    return np.matmul(T, pts), T


def find_matrix_by_pt_matches(pts1: np.ndarray, pts2: np.ndarray, zero_eigen=True):
    """
    uses svd decomposition to estimate 3X3 matrix from pt correspondence
    :param pts1: pt set 1 of shape (3 , num_pts) or (num_pts, 3) // in homogeneous coordinates
    :param pts2: pt set 2 of shape (3 , num_pts) or (num_pts, 3) // in homogeneous coordinates
    :param zero_eigen: boolean flag -> enforce last eigenvalue = 0
    :return: F (3x3 Matrix) where for all matching x, x' in pts1, pts2 : x'.T @ F @ x = 0
    """
    assert pts1.shape[0] == pts2.shape[0] == 3 or pts1.shape[1] == pts2.shape[1] == 3, "illegal input"

    # enforce shape (num_pts, 3)
    if pts1.shape[0] == 3:
        pts1, pts2 = pts1.T, pts2.T

    # build constraint matrix

    A = np.asarray([pts1[:, 0] * pts2[:, 0],
                        pts1[:, 1] * pts2[:, 0],
                        pts1[:, 0],
                        pts1[:, 0] * pts2[:, 1],
                        pts1[:, 1] * pts2[:, 1],
                        pts2[:, 1],
                        pts1[:, 0],
                        pts1[:, 1],
                        pts1[:, 0] / pts1[:, 0]]).T
    A = A.reshape(8, 9)

    # SVD -> take smallest eigenvector
    U, S, VT = np.linalg.svd(A)
    f = VT.T[:, -1]

    # setup vector in matrix
    F = f.reshape((3, 3))

    if zero_eigen:
        U, S, VT = np.linalg.svd(F)
        S[-1] = 0
        F = np.matmul(U, np.matmul(np.diag(S), VT))

    return F
