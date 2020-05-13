from ex3_utils import *


def find_pts_correspondences(img1: np.ndarray, img2: np.ndarray, show=False):
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

    # TODO: think of better way to eliminate outliers
    # eliminate outliers
    matches = sorted(matches, key=lambda x: x.distance)[:8]
    # TODO: maybe this?
    # # Apply ratio test
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append((m, n))
    # return good

    if show:
        # draw match lines
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.imshow(res)
        plt.show()

    output = [[[], []], [[], []]]
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        output[0][0].append(x1)
        output[0][1].append(y1)
        output[1][0].append(x2)
        output[1][1].append(y2)
    return output


def find_fundamental_matrix(pt_correspondence: np.ndarray):
    """
    :param pt_correspondence: 8 point correspondences in non-homogeneous coordinates
    :return:
    """
    assert pt_correspondence.shape == (2, 2, 8)
    # normalize coordinates + add homogeneous coordinates
    X1, T1 = normalize_coords(pt_correspondence[0])
    X2, T2 = normalize_coords(pt_correspondence[1])

    # find matrix
    # -- linear solution + constraint enforce
    F_hat = find_matrix_by_pt_matches(X1, X2, zero_eigen=True)

    # de-normalize
    F = np.matmul(T2.T, np.matmul(F_hat, T1))
    return F


def calc_fundamental_matrix(img_path1: str, img_path2: str, show=False):
    # load images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find matches
    matches = find_pts_correspondences(img1, img2, show)
    matches = np.asarray(matches).reshape((2, 2, 8))

    # find matrix
    F = find_fundamental_matrix(matches)
    print(F)
    return F


if __name__ == "__main__":
    calc_fundamental_matrix("./external/oxford1.jpg", "./external/oxford2.jpg", show=True)
