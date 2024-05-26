# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import numpy as np
from numba import njit


# --------------------------------------------------------------------------------------------
# DEFINITIONS OF FUNCTIONS
# --------------------------------------------------------------------------------------------


@njit
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):

    """
    Samples from a beta distribution.

    Args:
        size (int): Size of the sample.
        concentration_0 (float): Concentration parameter of the first gamma distribution.
        concentration_1 (float): Concentration parameter of the second gamma distribution.

    Returns:
        numpy.ndarray: Sample from the beta distribution.
    """

    gamma_1_sample = np.array([np.random.gamma(concentration_1) for _ in range(size)])
    gamma_2_sample = np.array([np.random.gamma(concentration_0) for _ in range(size)])

    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


@njit
def clip(value, min_value, max_value):

    """
    Clips a value within a specified range.

    Args:
        value (float): Value to be clipped.
        min_value (float): Minimum value.
        max_value (float): Maximum value.

    Returns:
        float: Clipped value.
    """

    return max(min_value, min(value, max_value))


@njit
def get_box(lambda_value, height, width):

    """
    Generates bounding box coordinates based on lambda value.

    Args:
        lambda_value (float): Lambda value from the Beta distribution.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        tuple: Bounding box coordinates (boundaryx1, boundaryy1, boundaryx2, boundaryy2).
    """

    cut_rat = np.sqrt(1.0 - lambda_value)
    cut_w = np.int32(width * cut_rat)
    cut_h = np.int32(height * cut_rat)
    cut_x = np.random.randint(width)
    cut_y = np.random.randint(height)
    boundaryx1 = clip(cut_x - cut_w // 2, 0, width)
    boundaryy1 = clip(cut_y - cut_h // 2, 0, height)
    boundaryx2 = clip(cut_x + cut_w // 2, 0, width)
    boundaryy2 = clip(cut_y + cut_h // 2, 0, height)

    return boundaryx1, boundaryy1, boundaryx2, boundaryy2


@njit
def cutmix_one_sample(train_ds_one, train_ds_two, alpha, beta):

    """
    Performs CutMix on a single pair of images and labels.

    Args:
        train_ds_one (tuple): Tuple containing image and label from the first dataset.
        train_ds_two (tuple): Tuple containing image and label from the second dataset.
        alpha (float): Hyperparameter controlling the Beta distribution.
        beta (float): Hyperparameter controlling the Beta distribution.

    Returns:
        tuple: Augmented image and corresponding label.
    """

    (image1, label1), (image2, label2) = train_ds_one, train_ds_two
    height, width, _ = image1.shape
    lambda_value = sample_beta_distribution(1, alpha, beta)[0]
    bbx1, bby1, bbx2, bby2 = get_box(lambda_value, height, width)
    image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))
    label = lambda_value * label1 + (1 - lambda_value) * label2

    return image1, label


@njit
def cut_mix(dataset_one, dataset_two, alpha, beta):

    """
    Applies CutMix augmentation to a pair of datasets.

    Args:
        dataset_one (tuple): Tuple containing images and labels from the first dataset.
        dataset_two (tuple): Tuple containing images and labels from the second dataset.
        alpha (float): Hyperparameter controlling the Beta distribution.
        beta (float): Hyperparameter controlling the Beta distribution.

    Returns:
        tuple: Augmented images and corresponding labels.
    """

    (ds_one, labels_one) = dataset_one
    (ds_two, labels_two) = dataset_two
    images = np.empty_like(ds_one)
    labels = np.empty_like(labels_one)

    for i in range(ds_one.shape[0]):

        images[i], labels[i] = cutmix_one_sample((ds_one[i], labels_one[i]), (ds_two[i], labels_two[i]), alpha, beta)
    
    return images, labels