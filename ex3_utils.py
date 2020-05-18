import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches):
    """
    :param kp1: list of KeyPoint objects
    :param kp2: list of KeyPoint objects
    :param matches: list of DMatch objects
    :return: np.ndarray of homogeneous pt correspondence
    """
    output = [[[], [], []], [[], [], []]]
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        output[0][0].append(x1)
        output[0][1].append(y1)
        output[1][0].append(x2)
        output[1][1].append(y2)
        output[0][2].append(1)
        output[1][2].append(1)
    output = np.asarray(output).reshape((2, 3, -1))
    return output


def get_norm_matrix(pts: np.ndarray):
    """
    calculates normalization of point correspondence matching following term:
        1. centroid of the reference points is at the origin
        2. root mean square difference from origin is sqrt(2)
    finds a transformation matrix that performs the normalization: norm_x = T @ x
    :param pts: array representing pts shape (3, num_pts)
    :return: homogeneous pt_correspondences, normalization matrix T
    """
    assert pts.shape[0] == 3

    # get centroid
    centroid = np.mean(pts, 1)

    # compute the RMS distance to centroid
    RMS = np.sqrt(np.sum(np.mean((pts - centroid.reshape(3, 1)) ** 2, axis=1)))

    # calculate matrix
    T = np.asarray([[np.sqrt(2) / RMS,  0,                  - centroid[0] * np.sqrt(2) / RMS],
                    [0,                 np.sqrt(2) / RMS,   - centroid[1] * np.sqrt(2) / RMS],
                    [0,                 0,                  1                               ]]).reshape(3, 3)

    return pts, T


def build_matrix_from_pts(pts1: np.ndarray, pts2: np.ndarray, T1=np.eye(3), T2=np.eye(3), zero_eigen=True):
    """
    uses svd decomposition to estimate 3X3 matrix from pt correspondence
    :param pts1: pt set 1 of shape (num_pts, 3) // in homogeneous coordinates
    :param pts2: pt set 2 of shape (num_pts, 3) // in homogeneous coordinates
    :param T1: normalization matrix (3,3) for pts1
    :param T2: normalization matrix (3,3) for pts2
    :param zero_eigen: boolean flag -> enforce last eigenvalue = 0
    :return: F (3x3 Matrix) where for all matching x, x' in pts1, pts2 : x'.T @ F @ x = 0
    """
    assert pts1.shape[1] == pts2.shape[1] == 3
    assert T1.shape == T2.shape == (3,3)

    # build constraint matrix
    x1 = np.matmul(T1, pts1.T)
    x2 = np.matmul(T2, pts2.T)
    A = np.asarray([x1[:, 0] * x2[:, 0],
                    x1[:, 1] * x2[:, 0],
                    x1[:, 0],
                    x1[:, 0] * x2[:, 1],
                    x1[:, 1] * x2[:, 1],
                    x2[:, 1],
                    x1[:, 0],
                    x1[:, 1],
                    x1[:, 0] / x1[:, 0]]).T
    A = A.reshape(-1, 9)

    # SVD -> take smallest eigenvector
    U, S, VT = np.linalg.svd(A)
    f = VT.T[:, -1]

    # setup vector in matrix
    F = f.reshape((3, 3))

    # enforce det(F) = 0
    if zero_eigen:
        U, s, VT = np.linalg.svd(F)
        s[-1] = 0
        F = np.matmul(U, np.matmul(np.diag(s), VT))

    return F


def discard_outliers(kp1, kp2, matches, tol=None):
    """
    currently filters matching score <= tol
    :param F: fundamental matrix shape (3,3)
    :param kp1: list of KeyPoint objects
    :param kp2: list of KeyPoint objects
    :param matches: list of DMatch objects
    :param tol: tolerance factor, 1e-3 by default
    :return: inlier pts for which x'^T @ F @ x <= tol
    """
    if tol is None:
        tol = np.mean([m.distance for m in matches])
    idxs = np.where(np.array([m.distance for m in matches]).flatten() <= tol)[0]
    if idxs.shape[0] < 8:
        return kp1, kp2, matches[:8]
    rel_matches = []
    for i in idxs:
        rel_matches.append(matches[i])
    return kp1, kp2, rel_matches
