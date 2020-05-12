from ex3_utils import *


def find_fundamental_matrix(pt_correspondence: np.ndarray):
    """
    :param pt_correspondence: 8 point correspondences in non-homogeneous coordinates
    :return:
    """
    assert pt_correspondence.shape == (2, 8) or pt_correspondence.shape == (8, 2)
    # normalize coordinates + add homogeneous coordinates

    # find matrix
    # -- linear solution
    # -- constraint enforcement

    # de-normalize

    pass
