from ex3_utils import *


def eight_point(kp1, kp2, matches):
    """
    implement eight point algorithm
    :param kp1: key points in 1st image (KeyPoint list)
    :param kp2: key points in 2nd image (KeyPoint list)
    :param matches: mapping between the above points (DMatch list)
    :return: fundamental matrix
    """
    # gen points from matches
    pts = get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches)

    # get normalization matrices
    T1, T2 = get_norm_matrix(pts[0]), get_norm_matrix(pts[1])

    # find fundamental matrix corresponding to norm pts, this enforces pt normalization and det(F)=0
    F = build_matrix_from_pts(pts[0], pts[1], T1, T2)

    # denormalize
    F = np.matmul(T2.T, np.matmul(F, T1))
    # F = np.matmul(np.linalg.inv(T2).T, np.matmul(F, np.linalg.inv(T1)))
    return F, T1, T2


def find_matches(img1:np.ndarray, img2:np.ndarray, func=discard_outliers, show=True):
    """
    find keypoint matches between images using ORB, discard outliers
    :param img1: np.ndarray representing img2
    :param img2: np.ndarray representing img2
    :param func: filter function to remove outliers default=discard_outliers (by score)
    :return: kp1 <KeyPoint list>, kp2 <KeyPoint list>, matches <DMatch list>
    """
    # get orb object
    orb = cv2.ORB_create()

    # get key points
    kp1, kp2 = orb.detect(img1, None), orb.detect(img2, None)

    # get descriptors
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    if show:
        # draw match lines
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.figure(figsize=(15, 10))
        plt.imshow(res)
        plt.title("all matches found", fontsize=20)
        plt.show()

    if func is not None:
        kp1, kp2, matches = func(kp1, kp2, matches)

    if show:
        # draw inliers matching lines
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.figure(figsize=(15, 10))
        plt.imshow(res)
        plt.title("inliers matches found", fontsize=20)
        plt.show()

    return kp1, kp2, matches


def draw_epilines(im1: np.ndarray, im2: np.ndarray, F: np.ndarray,T1: np.ndarray, T2: np.ndarray, kp1, kp2, matches):
    # get pt coordinates
    pts = get_pt_correspondence_from_keypoint_matches(kp1, kp2, matches)
    pts1, pts2 = pts[0], pts[1]  # assuming these coords are homogeneous with z = 1
    pts1, pts2 = (pts1[:2, :].astype(int)).T, (pts2[:2, :].astype(int)).T
    # find corresponding epilines
    # lines1 = np.matmul(np.matmul(np.linalg.inv(T1), F.T), pts2)
    # lines2 = np.matmul(np.matmul(np.linalg.inv(T2),F), pts1)
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
    kp1, kp2, matches = find_matches(img1, img2, show=False)
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

    # get keypoints and matches
    kp1, kp2, matches = find_matches(img1, img2)

    # 8 point algorithm
    F, T1, T2 = eight_point(kp1, kp2, matches)

    draw_epilines(img1, img2, F, T1, T2, kp1, kp2, matches)


if __name__ == "__main__":
    main_func("./external/img1.jpeg", "./external/img2.jpeg", show=True)
