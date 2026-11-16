"""Experimental sparse text watermark and pixel-based recovery."""

import binascii

import cv2
import numpy as np
from PIL import Image

WATERMARK = "9845456435684656"
TILE_WIDTH = 256
TILE_HEIGHT = 128
TEXT_LEFT = 16
TEXT_BASELINE = 65
CELL_WIDTH = 13
TRANSPARENT_COLOR = (0, 255, 0)
CARRIER_SEED = 20261112
ACTIVE_FRACTION = 0.25
IDENTIFIER_LENGTH = 16
CHECKSUM_TOP = 88
CHECKSUM_BITS = 16
TEXT_ANGLE = 20


def rotate_tile(image, angle, interpolation=cv2.INTER_NEAREST):
    matrix = cv2.getRotationMatrix2D((TILE_WIDTH / 2, TILE_HEIGHT / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (TILE_WIDTH, TILE_HEIGHT), flags=interpolation)


def text_mask(identifier):
    text = str(identifier)
    if len(text) != IDENTIFIER_LENGTH or not text.isascii() or not text.isdigit():
        raise ValueError("测试编号必须是 16 位数字")
    mask = np.zeros((TILE_HEIGHT, TILE_WIDTH), dtype=np.uint8)
    for index, digit in enumerate(text):
        cv2.putText(
            mask,
            digit,
            (TEXT_LEFT + index * CELL_WIDTH, TEXT_BASELINE),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            255,
            1,
            cv2.LINE_8,
        )
    return rotate_tile(mask, TEXT_ANGLE)


def text_region():
    region = np.zeros((TILE_HEIGHT, TILE_WIDTH), dtype=np.uint8)
    region[TEXT_BASELINE - 28:TEXT_BASELINE + 8, TEXT_LEFT - 8:TEXT_LEFT + CELL_WIDTH * IDENTIFIER_LENGTH + 8] = 1
    return rotate_tile(region, TEXT_ANGLE) > 0


def carrier_pattern():
    generator = np.random.default_rng(CARRIER_SEED)
    shape = (TILE_HEIGHT, TILE_WIDTH)
    carrier = generator.choice(np.array([-1, 1], dtype=np.int8), size=shape)
    active = generator.random(shape) < ACTIVE_FRACTION
    return carrier, active


def checksum_groups():
    generator = np.random.default_rng(CARRIER_SEED + 1)
    return generator.integers(0, CHECKSUM_BITS, size=(TILE_HEIGHT - CHECKSUM_TOP, TILE_WIDTH))


def checksum(identifier):
    return binascii.crc_hqx(identifier.encode("ascii"), 0xFFFF)


def make_overlay(identifier, width, height):
    mask = text_mask(identifier)
    carrier, active = carrier_pattern()
    symbols = carrier.copy()
    checksum_value = checksum(str(identifier))
    bits = np.array([(checksum_value >> (15 - index)) & 1 for index in range(CHECKSUM_BITS)])
    symbols[CHECKSUM_TOP:] *= np.where(bits[checksum_groups()] == 1, -1, 1)
    tile = np.empty((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    tile[:] = TRANSPARENT_COLOR
    tile[active & (symbols > 0)] = (255, 255, 255)
    tile[active & (symbols < 0)] = (0, 0, 0)
    tile[text_region()] = TRANSPARENT_COLOR
    outline = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8)) > 0
    tile[outline & (mask == 0)] = (255, 255, 255)
    tile[mask > 0] = (0, 0, 0)
    repetitions = ((height + TILE_HEIGHT - 1) // TILE_HEIGHT, (width + TILE_WIDTH - 1) // TILE_WIDTH, 1)
    return Image.fromarray(np.tile(tile, repetitions)[:height, :width])


def simulate_overlay(image, identifier, alpha):
    background = image.convert("RGB")
    overlay = make_overlay(identifier, background.width, background.height)
    pixels = np.asarray(overlay)
    active = np.any(pixels != TRANSPARENT_COLOR, axis=2)
    mask = Image.fromarray((active * 255).astype(np.uint8))
    return Image.composite(Image.blend(background, overlay, alpha), background, mask)


def recover(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    if height < TILE_HEIGHT or width < TILE_WIDTH:
        raise ValueError(f"图像至少需要 {TILE_WIDTH} × {TILE_HEIGHT} 像素")

    highpass = gray - cv2.GaussianBlur(gray, (0, 0), 3)
    padded_height = ((height + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    padded_width = ((width + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
    padding = ((0, padded_height - height), (0, padded_width - width))
    padded = np.pad(highpass, padding, constant_values=np.nan)
    shape = (padded_height // TILE_HEIGHT, TILE_HEIGHT, padded_width // TILE_WIDTH, TILE_WIDTH)
    folded = np.nanmedian(padded.reshape(shape), axis=(0, 2))

    carrier, active = carrier_pattern()
    template = (carrier * active).astype(np.float32)
    template[text_region()] = 0
    template[CHECKSUM_TOP:] = 0
    template -= cv2.GaussianBlur(template, (0, 0), 3)
    correlation = np.fft.ifft2(np.fft.fft2(folded) * np.conj(np.fft.fft2(template))).real
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    deviation = float(np.std(correlation))
    score = float((correlation[peak] - np.mean(correlation)) / deviation) if deviation > 1e-9 else 0.0
    aligned = np.roll(folded, (-peak[0], -peak[1]), axis=(0, 1))

    region = text_region()
    detail = aligned - np.median(aligned[region])
    limit = float(np.percentile(np.abs(detail[region]), 99))
    signal = np.where(region, detail * min(80 / limit, 100), 0) if limit >= 0.005 else np.zeros_like(detail)
    signal = np.roll(signal, (peak[0], peak[1]), axis=(0, 1))
    repetitions = ((height + TILE_HEIGHT - 1) // TILE_HEIGHT, (width + TILE_WIDTH - 1) // TILE_WIDTH)
    signal = np.tile(signal, repetitions)[:height, :width]
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    enhanced = Image.fromarray(np.clip(pixels + signal[:, :, None], 0, 255).astype(np.uint8))

    demodulated = aligned * carrier * active
    horizontal = rotate_tile(aligned, -TEXT_ANGLE, cv2.INTER_LINEAR)
    digits = []
    for index in range(IDENTIFIER_LENGTH):
        cell_left = TEXT_LEFT + index * CELL_WIDTH
        cell = horizontal[TEXT_BASELINE - 20:TEXT_BASELINE + 1, cell_left:cell_left + CELL_WIDTH]
        candidates = []
        for digit in "0123456789":
            digit_mask = np.zeros((21, CELL_WIDTH), dtype=np.uint8)
            cv2.putText(digit_mask, digit, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 1, cv2.LINE_8)
            glyph = digit_mask > 0
            outline = cv2.dilate(digit_mask, np.ones((3, 3), dtype=np.uint8)) > 0
            template = np.where(glyph, -1.0, np.where(outline, 1.0, 0.0))
            candidates.append(float(np.sum(cell * template) / max(np.linalg.norm(template), 1)))
        digits.append(str(int(np.argmax(candidates))))
    candidate = "".join(digits)
    groups = checksum_groups()
    checksum_values = demodulated[CHECKSUM_TOP:]
    checksum_active = active[CHECKSUM_TOP:] & ~region[CHECKSUM_TOP:]
    checksum_bits = [
        int(np.sum(checksum_values[(groups == index) & checksum_active]) < 0)
        for index in range(CHECKSUM_BITS)
    ]
    extracted_checksum = sum(bit << (15 - index) for index, bit in enumerate(checksum_bits))
    checksum_ok = extracted_checksum == checksum(candidate)
    detected = score >= 7.0 and checksum_ok
    return {
        "detected": detected,
        "identifier": candidate if detected else None,
        "checksum_ok": checksum_ok,
        "match_score": score,
        "offset": [int(peak[1]), int(peak[0])],
    }, enhanced


def extract(image):
    return recover(image)[0]
