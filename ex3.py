import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread


def compute_transformation_and_normalize_points(image_points: np.ndarray) -> (np.ndarray, np.ndarray):
    assert image_points is not None
    assert type(image_points) is np.ndarray
    assert len(image_points.shape) == 2
    assert image_points.shape[1] == 2
    n = image_points.shape[0]

    mean = np.mean(image_points, axis=0)
    std = np.std(image_points, axis=0)
    rms = (std[0] ** 2 + std[1] ** 2) ** 0.5
    transformation = np.array([[np.sqrt(2) / rms, 0, -mean[0]],
                               [0, np.sqrt(2) / rms, -mean[1]],
                               [0, 0, 1]])
    homogeneous_points = np.hstack((image_points, np.ones(n).reshape(n, 1)))
    homogeneous_normalized_points = (transformation @ homogeneous_points.T).T
    normalized_points = homogeneous_normalized_points[:, :2] / homogeneous_normalized_points[:, -1].reshape(n, 1)
    return normalized_points, transformation


def from_normalized_points_to_regular_points(normalized_points, transformation):
    assert normalized_points is not None
    assert type(normalized_points) is np.ndarray
    assert len(normalized_points.shape) == 2
    assert normalized_points.shape[1] == 2
    n = normalized_points.shape[0]

    homogeneous_normalized_points = np.hstack((normalized_points, np.ones(n).reshape(n, 1)))
    homogeneous_points = (np.linalg.inv(transformation) @ homogeneous_normalized_points.T).T
    image_points = homogeneous_points[:, :2] / homogeneous_points[:, -1].reshape(n, 1)
    return image_points


def from_normalized_lines_to_regular_lines(normalized_lines, transformation):
    assert normalized_lines is not None
    assert type(normalized_lines) is np.ndarray
    assert len(normalized_lines.shape) == 2
    assert normalized_lines.shape[1] == 2
    n = normalized_lines.shape[0]

    homogeneous_normalized_lines = np.hstack((normalized_lines, np.ones(n).reshape(n, 1)))
    homogeneous_lines = (transformation.T @ homogeneous_normalized_lines.T).T
    lines = homogeneous_lines[:, :2] / homogeneous_lines[:, -1].reshape(n, 1)
    return lines


def compute_A_from_points(normalized_image1_points: np.ndarray, normalized_image2_points: np.ndarray) -> np.ndarray:
    assert normalized_image1_points is not None
    assert normalized_image2_points is not None
    assert type(normalized_image1_points) is np.ndarray
    assert type(normalized_image2_points) is np.ndarray
    assert len(normalized_image1_points.shape) == len(normalized_image2_points.shape) == 2
    assert normalized_image1_points.shape[1] == normalized_image2_points.shape[1] == 2
    assert normalized_image1_points.shape[0] == normalized_image2_points.shape[0]
    n = normalized_image1_points.shape[0]

    A = np.zeros((n, 9))
    for i in range(n):
        x, y = normalized_image1_points[i]
        x_tag, y_tag = normalized_image2_points[i]
        A[i] = np.array([x_tag * x, x_tag * y, x_tag, y_tag * x, y_tag * y, y_tag, x, y, 1])
    return A


def find_F_from_A(A: np.ndarray) -> np.ndarray:
    assert A is not None
    assert type(A) is np.ndarray
    assert len(A.shape) == 2
    assert A.shape[1] == 9

    u, s, vh = np.linalg.svd(A)
    f = vh[-1]
    F = np.array([[f[0], f[1], f[2]],
                  [f[3], f[4], f[5]],
                  [f[6], f[7], f[8]]])
    return F


def find_rank_2_matrix_from_F(F: np.ndarray) -> np.ndarray:
    assert F is not None
    assert type(F) is np.ndarray
    assert F.shape == (3, 3)

    u, s, vh = np.linalg.svd(F)
    new_s = np.array([s[0], s[1], 0])
    new_F = u @ np.diag(new_s) @ vh
    return new_F


def find_fundamental_matrix(normalized_image1_points: np.ndarray, normalized_image2_points: np.ndarray) -> np.ndarray:
    assert len(normalized_image1_points.shape) == len(normalized_image2_points.shape) == 2
    assert normalized_image1_points.shape[1] == normalized_image2_points.shape[1] == 2
    assert normalized_image1_points.shape[0] == normalized_image2_points.shape[0]

    A = compute_A_from_points(normalized_image1_points, normalized_image2_points)
    F = find_F_from_A(A)
    F = find_rank_2_matrix_from_F(F)
    return F


def get_epipolar_lines(image1_points, image2_points, F):
    lines_in_first_image = np.zeros((image2_points.shape[0], 2))
    lines_in_second_image = np.zeros((image1_points.shape[0], 2))
    for i in range(image2_points.shape[0]):
        x, y = image2_points[i]
        line = F.T @ np.array([x, y, 1]).reshape(3, 1)
        lines_in_first_image[i] = line.reshape(3)[:2] / line.reshape(3)[2]

    for i in range(image1_points.shape[0]):
        x, y = image1_points[i]
        line = F @ np.array([x, y, 1]).reshape(3, 1)
        lines_in_second_image[i] = line.reshape(3)[:2] / line.reshape(3)[2]

    return lines_in_first_image, lines_in_second_image


def draw_epipolar_lines(image, lines):
    plt.figure()
    plt.imshow(image)
    x_range = np.arange(0, image.shape[1])
    for i in range(lines.shape[0]):
        a, b = lines[i]
        y_values = (-a * x_range - 1) / b
        ind = np.argwhere((0 <= y_values) & (y_values <= image.shape[0]))
        x_values, y_values = x_range[ind], y_values[ind]
        plt.plot(x_values, y_values, color='blue')
    plt.show()


if __name__ == '__main__':
    points1 = np.array([[552, 189], [438, 163], [358, 310], [512, 301],
                        [434, 337], [163, 430], [437, 411], [479, 310], [412, 357]])
    points2 = np.array([[217, 230], [110, 186], [12, 313], [166, 334],
                        [85, 360], [104, 457], [80, 435], [134, 338], [61, 378]])
    normalized_points1, transformation1 = compute_transformation_and_normalize_points(points1)
    normalized_points2, transformation2 = compute_transformation_and_normalize_points(points2)
    F = find_fundamental_matrix(normalized_points1, normalized_points2)
    normalized_lines_in_first_image, normalized_lines_in_second_image = \
        get_epipolar_lines(normalized_points1, normalized_points2, F)
    lines_in_first_image, lines_in_second_image = \
        from_normalized_lines_to_regular_lines(normalized_lines_in_first_image, transformation1), \
        from_normalized_lines_to_regular_lines(normalized_lines_in_second_image, transformation2)
    image1 = imread("external/oxford1.jpg")
    image2 = imread("external/oxford2.jpg")
    draw_epipolar_lines(image1, lines_in_first_image)
    draw_epipolar_lines(image2, lines_in_second_image)
