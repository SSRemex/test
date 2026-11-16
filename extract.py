"""Extract a test identifier from an image supplied by the user."""

import argparse
import json

from PIL import Image, ImageOps

from watermark import extract


def main():
    parser = argparse.ArgumentParser(description="从截图或照片中提取实验水印")
    parser.add_argument("image", help="待分析的本地图片路径")
    arguments = parser.parse_args()
    with Image.open(arguments.image) as image:
        result = extract(ImageOps.exif_transpose(image))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["detected"]:
        print("未能可靠解码并通过校验；不能据此认定图片不含水印。")
        return 1
    print("编号通过 CRC 校验；仅供实验，不构成身份或防伪证明。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
