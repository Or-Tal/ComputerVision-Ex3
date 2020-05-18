from ex3_utils import *


def find_pts_correspondences(img1: np.ndarray, img2: np.ndarray, show=False):
    """
    find matches between two grayscale images using ORB +
    :param img1: np.ndarray representing img2
    :param img2: np.ndarray representing img2
    :param show: boolean, True = display image
    :return: kp1: KeyPoint list relative to img 1
             kp2: KeyPoint list relative to img 2
             matches: DMatch list of these correspondences
    """
    # init detector
    orb = cv2.ORB_create()

    # find the keypoints
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)

    # calculate descriptors
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if show:
        # draw match lines
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.imshow(res)
        plt.title("all matches found")
        plt.show()

    return kp1, kp2, matches


def find_fundamental_matrix(kp1, kp2, matches, show=False, filter_func=None):
    """
    :param kp1: KeyPoint list relative to img 1
    :param kp2: KeyPoint list relative to img 2
    :param show: boolean flag -> show inlier pt correspondence
    :param filter_func: outlier filtering function:
                    func(Fundamental matrix, KeyPoint list 1, KeyPoint list 2, DMatch list)
                    returns: KeyPoint list 1, KeyPoint list 2, DMatch list (of inliers)
                    # note - DMatch list is at least of length 8
    :param matches: DMatch list of these correspondences
    :return: Fundamental matrix (3,3)
    """
    # retrieve pt_correspondence
    pt_correspondence = get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches)

    # calc normalize matrix, get homogeneous coordinates
    X1, T1 = get_norm_matrix(pt_correspondence[0])
    X2, T2 = get_norm_matrix(pt_correspondence[1])

    if filter_func is not None:
        # find matrix
        # comment out in case of different implementation of outlier filtering
        # -- enforce shape (num_pts, 3)
        if X1.shape[0] == 3:
            X1, X2 = X1.T, X2.T

        # -- gen F matrix
        F = build_matrix_from_pts(X1, X2, T1, T2)

        # -- remove outliers
        F = np.matmul(T2.T, np.matmul(F, T1))
        kp1, kp2, matches = filter_func(F, kp1, kp2, matches)
        #
    else:
        # case where filtering by match score
        # -- remove outliers
        kp1, kp2, matches = discard_outliers(kp1, kp2, matches)

    # -- re-gen pts
    pt_correspondence = get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches)
    X1, X2 = pt_correspondence[0], pt_correspondence[1]
    # ---- enforce shape (num_pts, 3)
    if X1.shape[0] == 3:
        X1, X2 = X1.T, X2.T

    # -- re-gen F matrix
    F = build_matrix_from_pts(X1, X2, T1, T2)

    # de-normalize
    F = np.matmul(T2.T, np.matmul(F, T1))

    return F, kp1, kp2, matches


def calc_fundamental_matrix(img_path1: str, img_path2: str, show=False):
    """
    main function - estimate fundamental matrix from two images
    :param img_path1: path to img1 file
    :param img_path2: path to img2 file
    :param show: boolean, True = display image
    :return: fundamental matrix
    """
    # load images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find matches
    kp1, kp2, matches = find_pts_correspondences(img1, img2, show)

    # find matrix, inlier matching points
    F, kp1, kp2, matches = find_fundamental_matrix(kp1, kp2, matches)
    if show:
        # draw match lines
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.imshow(res)
        plt.title("filtered inliers found")
        plt.show()
    print(F)
    return F


if __name__ == "__main__":
    calc_fundamental_matrix("./external/oxford1.jpg", "./external/oxford2.jpg", show=True)
