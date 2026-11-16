"""Manual screen-photo geometry correction and full-photo reconstruction."""

import cv2
import numpy as np
from PIL import Image


def enhance_photo_pixels(image, channel="亮度", strength=12):
    if not 1 <= strength <= 40:
        raise ValueError("增强强度必须在 1 到 40 之间")
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    channels = {"红色": 0, "绿色": 1, "蓝色": 2}
    if channel == "亮度":
        plane = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    elif channel in channels:
        plane = pixels[:, :, channels[channel]]
    else:
        raise ValueError("不支持的颜色通道")
    baseline = cv2.GaussianBlur(plane, (0, 0), 4)
    detail = plane - baseline
    return Image.fromarray(np.clip(128 + detail * strength, 0, 255).astype(np.uint8))


def validate_photo_geometry(image, corners, screen_size):
    width, height = screen_size
    if not (256 <= width <= 7680 and 128 <= height <= 4320 and width * height <= 16777216):
        raise ValueError("屏幕分辨率超出支持范围（最多 16777216 像素）")
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("必须选择四个有效的屏幕角点")
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= image.width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= image.height).any()
    ):
        raise ValueError("角点必须位于照片范围内")
    if not cv2.isContourConvex(points) or cv2.contourArea(points, oriented=True) < 100:
        raise ValueError("角点必须按左上、右上、右下、左下的顺序选择")
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    return points, target


def rectify_photo(image, corners, screen_size):
    points, target = validate_photo_geometry(image, corners, screen_size)
    transform = cv2.getPerspectiveTransform(points, target)
    pixels = np.asarray(image.convert("RGB"))
    corrected = cv2.warpPerspective(pixels, transform, screen_size, flags=cv2.INTER_LINEAR)
    return Image.fromarray(corrected), transform


def restore_photo(original, corrected, enhanced, transform):
    if corrected.size != enhanced.size:
        raise ValueError("增强图与校正图尺寸不一致")
    difference = np.asarray(enhanced.convert("RGB"), dtype=np.float32) - np.asarray(
        corrected.convert("RGB"), dtype=np.float32
    )
    inverse = np.linalg.inv(transform)
    difference_in_photo = cv2.warpPerspective(
        difference, inverse, original.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    pixels = np.asarray(original.convert("RGB"), dtype=np.float32)
    return Image.fromarray(np.clip(pixels + difference_in_photo, 0, 255).astype(np.uint8))
