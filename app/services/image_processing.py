import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(
    pil_image: Image.Image,
    max_width: int = 1280,
    max_height: int = 1920,
) -> Image.Image:
    """Pre-process an image to maximize OCR accuracy.

    All enhancement runs at the full render resolution to preserve detail
    for deskew angle estimation, non-local-means denoising, and edge-
    preserving sharpen. The resize happens last so that downsampling
    operates on an already-enhanced image.

    Pipeline:
    1. Grayscale
    2. Background normalization (remove shadows/vignetting)
    3. CLAHE contrast enhancement
    4. Non-local means denoise (preserves text strokes)
    5. Sharpen to reinforce text edges
    6. Deskew
    7. Resize to target dimensions (aspect-ratio preserved)
    8. White border padding
    """
    img = np.array(pil_image)

    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # 2. Background normalization at full resolution.
    #    Removes shadows, vignetting, uneven lighting.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=51)
    normalized = cv2.divide(gray, background, scale=255)

    # 3. CLAHE — locally adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 4. Gentle denoise — preserves text strokes and handwriting.
    #    Critical for photocopies, faxes, and older scans.
    denoised = cv2.fastNlMeansDenoising(
        enhanced, h=5, templateWindowSize=7, searchWindowSize=21,
    )

    # 5. Sharpen to reinforce text edges before downscale
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    # 6. Deskew at full resolution for accurate angle estimation
    result = _deskew(sharpened)

    # 7. Resize to fit within the target canvas (aspect-ratio preserved),
    #    then pad with white to reach EXACTLY max_width × max_height.
    #    Consistent output resolution matches DeepSeek-OCR's vision encoder
    #    input size exactly so the model never internally resizes.
    h, w = result.shape
    scale = min(max_width / w, max_height / h)
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        result = cv2.resize(result, (new_w, new_h), interpolation=interp)
    else:
        new_w, new_h = w, h

    # 8. Pad to exact canvas size with white borders (centered).
    pad_w = max_width - new_w
    pad_h = max_height - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    result = cv2.copyMakeBorder(
        result, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=255,
    )

    return Image.fromarray(result)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct small skew angles using minimum-area bounding rectangle."""
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in [-90, 0); normalise to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct small skews
    if abs(angle) > 10 or abs(angle) < 0.3:
        return image

    h, w = image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255,
    )
    return rotated
