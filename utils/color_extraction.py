import numpy as np

from sklearn.cluster import KMeans


def highest_frequency_color(img: np.ndarray, no_black=True) -> np.ndarray:
    """
    Get the highest frequency color from the image.

    Args:
        img (np.ndarray): Image
        no_black (bool): Whether to remove black from results
            (mainly for skin color extraction). Defaults to True.

    Returns:
        np.ndarray: BGR value of the highest frequency color
    """
    unique_pixels, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)

    if no_black:
        index_of_black = np.where(unique_pixels == np.array([0, 0, 0]))[0][0]
        counts = np.delete(counts, index_of_black)
        unique_pixels = unique_pixels[unique_pixels != np.array([0, 0, 0])].reshape(
            -1, 3
        )

    return unique_pixels[np.argmax(counts)]


def k_means_color_clustering(img: np.ndarray, clusters=5, no_black=True) -> np.ndarray:
    """
    Get the clusters of most common colors from image

    Args:
        img (np.ndarray): Image
        clusters (int, optional): Max clusters of colors. Defaults to 5.
        no_black (bool, optional): Whether to remove black from results
            (mainly for skin color extraction). Defaults to True.

    Returns:
        np.ndarray: BGR values of the cluster of n colors
    """

    img = img.reshape(-1, 3)
    cluster = KMeans(n_clusters=clusters)
    result = cluster.fit(img)

    k_means_colors = result.cluster_centers_.astype(np.uint8)
    if no_black:
        k_means_colors = k_means_colors[k_means_colors != np.array([0, 0, 0])].reshape(
            -1, 3
        )

    return k_means_colors


def avg_k_means_high_freq_colors(
    highest_freq_color: np.ndarray, k_means_colors: np.ndarray
) -> list:
    """
    Average the color value results from highest frequency and k-means clustering

    Args:
        highest_freq_color (np.ndarray): BGR value of the highest frequency color
        k_means_colors (np.ndarray): BGR values of the cluster of n colors

    Returns:
        list: Average RGB color values
    """
    b_1, g_1, r_1 = highest_freq_color
    b_2, g_2, r_2 = np.sum(k_means_colors, axis=0)

    num_colors = k_means_colors.shape[0] + 1
    b_avg = np.divide(np.add(b_1, b_2), num_colors)
    g_avg = np.divide(np.add(g_1, g_2), num_colors)
    r_avg = np.divide(np.add(r_1, r_2), num_colors)

    return [r_avg, g_avg, b_avg]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB color code to HEX color code
    """

    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
