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

    # # find the keypoints
    # kp1 = orb.detect(img1, None)
    # kp2 = orb.detect(img2, None)
    #
    # # calculate descriptors
    # kp1, des1 = orb.compute(img1, kp1)
    # kp2, des2 = orb.compute(img2, kp2)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    if show:
        # draw match lines
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.figure(figsize=(15,10))
        plt.imshow(res)
        plt.title("all matches found", fontsize=20)
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

    # find fundamental matrix
    if filter_func is not None:

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

    # -- gen pts
    pt_correspondence = get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches)

    # -- calc normalize matrix, get homogeneous coordinates
    X1, T1 = get_norm_matrix(pt_correspondence[0])
    X2, T2 = get_norm_matrix(pt_correspondence[1])

    # -- enforce shape (num_pts, 3)
    if X1.shape[0] == 3:
        X1, X2 = X1.T, X2.T

    # --gen F matrix
    F = build_matrix_from_pts(X1, X2, T1, T2)

    # de-normalize
    F = np.matmul(T2.T, np.matmul(F, T1))

    return F, kp1, kp2, matches


def main_func(img_path1: str, img_path2: str, show=False):
    """
    main function - estimate fundamental matrix from two images
    :param img_path1: path to img1 file
    :param img_path2: path to img2 file
    :param show: boolean, True = display image inliers found, print fundamental matrix
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
        plt.figure(figsize=(15,10))
        plt.imshow(res)
        plt.title("filtered inliers found", fontsize=20)
        plt.show()

        # print the fundamental matrix, centroid,
        print("fundamental matrix:\n{}".format(F))

    draw_epilines(img1, img2, F, kp1, kp2, matches)
    return


def draw_epilines(im1: np.ndarray, im2: np.ndarray, F: np.ndarray, kp1, kp2, matches):
    # get pt coordinates
    pts = get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches)
    pts1, pts2 = pts[0], pts[1]  # assuming these coords are homogeneous with z = 1
    pts1, pts2 = (pts1[:2, :].astype(int)).T, (pts2[:2, :].astype(int)).T
    # find corresponding epilines
    # lines1 = np.matmul(F.T, pts2).T
    # lines2 = np.matmul(F, pts1).T
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # draw lines on images
    img1 = draw_lines(im1, lines1, pts1)
    img2 = draw_lines(im2, lines2, pts2)

    # plot
    fig, axs = plt.subplots(2,1, figsize=(15, 10))
    plt.suptitle("our result: epipolar lines", fontsize=20)
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    plt.show()

    # -------------- compare to open-cv result --------------

    # find matches
    kp1, kp2, matches = find_pts_correspondences(img1, img2, show=False)
    pts = get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches)
    pts1, pts2 = (pts[0, :2, :].astype(int)).T, (pts[1, :2, :].astype(int)).T  # assuming these coords are homogeneous with z = 1

    # find fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # find corresponding epilines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # draw lines on images
    img1 = draw_lines(im1, lines1, pts1)
    img2 = draw_lines(im2, lines2, pts2)

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    plt.suptitle("opencv result: epipolar lines", fontsize=20)
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    plt.show()


if __name__ == "__main__":
    main_func("./external/img1.jpeg", "./external/img2.jpeg", show=True)
