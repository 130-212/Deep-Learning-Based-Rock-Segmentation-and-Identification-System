from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import numpy as np
from torchvision import transforms


def getMask(sam, image, iou_threshold, nms_threshold):

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=iou_threshold,
        stability_score_thresh=0.95,
        box_nms_thresh=nms_threshold,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,  # Requires open-cv to run post-processing
    )

    masks = mask_generator.generate(image)

    print('图像被分为{}部分'.format(len(masks)))

    return masks

def getwrongMask(image, i):
    import json
    print('load {}'.format(i))
    with open('{}.json'.format(i), "r") as file:
        anns = json.load(file)
    for ann in anns:
        ann['segmentation'] = [[True if x == 1 else False for x in row] for row in ann['segmentation']]
        ann['segmentation'] = np.array(ann['segmentation'], dtype=bool)
    return anns


def getImgs(image, masks):

    masked_images = []

    for ann in masks:
        m = ann['segmentation']
        masked_image = np.copy(image)
        masked_image[~m] = 255
        masked_images.append(masked_image)

    return masked_images

def recolor_and_display_image(image, masks):
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

    img = image.astype(float) / 255.0  # 将图像转换为浮点数类型
    mask_size = (image.shape[1], image.shape[0])  # 调整后的掩码大小
    for ann in sorted_anns:
        if 'stone' not in ann.keys():
            ann['stone'] = True
        m = ann['segmentation']
        resized_mask = cv2.resize(m.astype(float), mask_size)  # 调整掩码大小
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        expanded_color_mask = np.tile(color_mask, (np.count_nonzero(resized_mask), 1))  # 扩展颜色掩码的形状
        img[resized_mask.astype(bool)] = img[resized_mask.astype(bool)] * 0.7 + expanded_color_mask[:, :3] * 0.3  # 使用混合颜色方法叠加
    img = (img * 255).astype(np.uint8)  # 将图像重新缩放为0到255之间的整数类型
    return img

