from ex3_utils import *


def find_pts_correspondences(img1: np.ndarray, img2: np.ndarray):
    # find pt correspondence using orb
    pass


def find_fundamental_matrix(pt_correspondence: np.ndarray):
    """
    :param pt_correspondence: 8 point correspondences in non-homogeneous coordinates
    :return:
    """
    assert pt_correspondence.shape == (2, 2, 8)
    # normalize coordinates + add homogeneous coordinates
    T1, X1 = normalize_coords(pt_correspondence[0])
    T2, X2 = normalize_coords(pt_correspondence[1])

    # find matrix
    # -- linear solution + constraint enforce
    F_hat = find_matrix_by_pt_matches(X1, X2, zero_eigen=True)

    # de-normalize
    F = np.matmul(T2.T, np.matmul(F_hat, T1))

    return F

