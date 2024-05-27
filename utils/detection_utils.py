import cv2 as cv
import numpy as np

def detect_face(portrait_img: np.ndarray) -> np.ndarray:
    """
    Detect face in the image using Haar-cascade detection from OpenCV.

    Args:
        portrait_img (np.ndarray): Portrait image

    Returns:
        np.ndarray: Cropped portrait image centered around the face
    """
    
    portrait_img = cv.resize(portrait_img, (2048, 2048))
    grayscale_img = cv.cvtColor(portrait_img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    face = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.5, minNeighbors=6, minSize=(40, 40))
    face_cropped_img = None
    for (x, y, w, h) in face:
        face_cropped_img = portrait_img[y:y+h, x:x+w] 

    return face_cropped_img
    
def detect_skin(portrait_img: np.ndarray) -> np.ndarray:
    """
    Detect skin in an image by using a masking of HSV, RGB, and YCbCR Color space.
    
    References:
    https://www.sciencedirect.com/science/article/abs/pii/S0262885620300573?via%3Dihub
    https://arxiv.org/pdf/1708.02694

    Args:
        portrait_img (np.ndarray): Cropped portrait image centered around the face

    Returns:
        np.ndarray: Image with mostly only skin highlighted
    """

    # HSV mask
    hsv_img = cv.cvtColor(portrait_img, cv.COLOR_BGR2HSV)
    hsv_min, hsv_max = np.array([0, 15, 0]), np.array([17, 170, 255])
    hsv_mask = cv.inRange(hsv_img, hsv_min, hsv_max)
    hsv_mask = cv.GaussianBlur(hsv_mask, (3, 3), 0)

    # RGB mask
    b_frame, g_frame, r_frame = [portrait_img[:, :, i] for i in range(3)]
    rgb_mask = np.logical_and.reduce(
        [
            r_frame > 95,
            g_frame > 40,
            b_frame > 20,
            np.abs(r_frame - g_frame) > 15,
            r_frame > g_frame,
            r_frame > b_frame,
        ]
    ).astype(np.uint8)
    rgb_mask = cv.GaussianBlur(rgb_mask, (3, 3), 0)

    # YCbCr mask
    ycbcr_img = cv.cvtColor(portrait_img, cv.COLOR_BGR2YCrCb)
    ycbcr_min, ycbcr_max = np.array([0, 135, 85]), np.array([255, 180, 135])
    ycbcr_mask = cv.inRange(ycbcr_img, ycbcr_min, ycbcr_max)
    ycbcr_mask = cv.GaussianBlur(ycbcr_mask, (3, 3), 0)

    detected_skin_mask = np.logical_and.reduce([hsv_mask, rgb_mask, ycbcr_mask]).astype(
        np.uint8
    )
    detected_skin_img = cv.bitwise_and(
        portrait_img, portrait_img, mask=detected_skin_mask
    )

    return detected_skin_img

def detect_hair(portrait_img: np.ndarray) -> np.ndarray:
    """
    Detect hair from portrait image

    Args:
        portrait_img (np.ndarray): Portrait image

    Returns:
        np.ndarray: The detected hair from portrait image
    """
    
    return portrait_img