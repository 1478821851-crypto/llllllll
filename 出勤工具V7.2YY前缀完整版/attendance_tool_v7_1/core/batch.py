"""Merge independently recognized screenshots without mixing crop coordinates."""
from attendance_tool_v7_1.core.pipeline import recognize_regions
import cv2
import numpy as np


def load_image(item):
    if 'image' in item:
        return item['image']
    return cv2.imdecode(np.frombuffer(item['data'], np.uint8), cv2.IMREAD_COLOR)


def crop_image(region):
    crop = region['crop']
    return cv2.imdecode(np.frombuffer(crop, np.uint8), cv2.IMREAD_COLOR) if isinstance(crop, bytes) else crop


def recognize_images(images, source, names, status=None, manual_settings=None, recognizer=None, compress_crops=False, ignored_prefixes=()):
    recognizer = recognizer or recognize_regions
    manual_settings = manual_settings or {}
    found, uncertain, texts, regions = set(), [], [], []
    for index, item in enumerate(images, 1):
        if status:
            status.write(f"{source}：第 {index}/{len(images)} 张 · {item['name']}")
        image = load_image(item)
        if image is None:
            raise ValueError(f"图片无法读取：{item['name']}")
        options = {'ignored_prefixes': ignored_prefixes} if source == 'YY' and ignored_prefixes else {}
        result = recognizer(image, source, names, status, manual_settings.get(item['key']), **options)
        image_found, image_uncertain, image_texts, image_regions = result
        found.update(image_found)
        texts.extend(image_texts)
        id_map = {}
        for region in image_regions:
            merged = dict(region)
            merged['id'] = len(regions) + 1
            merged['image_index'] = index
            merged['image_name'] = item['name']
            if compress_crops:
                ok, encoded = cv2.imencode('.png', merged['crop'])
                if not ok:
                    raise ValueError('核对裁图压缩失败。')
                merged['crop'] = encoded.tobytes()
            id_map[region['id']] = merged['id']
            regions.append(merged)
        # A crop may be a view retaining the entire decoded source image.
        region = None
        for row in image_uncertain:
            merged = dict(row)
            merged['区域'] = id_map[row['区域']]
            merged['图片'] = f"第{index}张 · {item['name']}"
            uncertain.append(merged)
        del image, image_regions, result
    return found, uncertain, list(dict.fromkeys(texts)), regions
