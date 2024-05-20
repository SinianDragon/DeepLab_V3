import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import imgviz
import tqdm
import argparse
import shutil


def main(args):
    # 初始化COCO api
    annFile = '{}/annotations/instances_{}.json'.format(args.dataDir, args.split)
    os.makedirs(os.path.join(args.dataDir, 'SegmentationClass', args.data_type), exist_ok=True)
    os.makedirs(os.path.join(args.dataDir, 'JPEGImages', args.data_type), exist_ok=True)
    coco = COCO(annFile)

    # 获取所有图像的ID
    imgIds = coco.getImgIds()

    # 对于每一张图像
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        # 获取图像信息
        img_info = coco.loadImgs(imgId)[0]
        img_name = img_info['file_name']

        # 获取该图像的所有注释ID
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        # 创建一个空的掩码
        mask = np.zeros((img_info['height'], img_info['width']))

        # 对于每一个注释
        if len(annIds) > 0:
            for ann in anns:
                # 获取类别ID
                catId = ann['category_id']
                # 获取该注释的掩码
                ann_mask = coco.annToMask(ann) * catId
                # 将注释的掩码添加到总掩码上
                # np.maximum(mask, ann_mask):mask和ann_mask按元素进行对比，保留较大的值
                # 这样做的好处是避免了重叠物体导致的类别索引值溢出
                # 缺点是导致了大物体上的小物体如果类别索引小于小物体，则不会在mask上标注出来
                mask = np.maximum(mask, ann_mask)

            # 将掩码转换为图像
            mask_img = Image.fromarray(mask.astype(np.uint8), mode="P")

            # 将图像转换为调色板模式
            colormap = imgviz.label_colormap()
            mask_img.putpalette(colormap.flatten())

            # 保存图像和对应的掩码
            img_origin_path = os.path.join(args.dataDir, args.split, img_name)
            img_output_path = os.path.join(args.dataDir, 'test', 'JPEGImages', args.data_type, img_name)
            seg_output_path = os.path.join(args.dataDir, 'test', 'SegmentationClass', args.data_type,
                                           img_name.replace('.jpg', '.png'))
            shutil.copy(img_origin_path, img_output_path)
            mask_img.save(seg_output_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", default="DataSets/COCO2017", type=str,
                        help="input dataset directory")
    parser.add_argument("--split", default="train2017", type=str,
                        help="train2017 or val2017")
    parser.add_argument("--data_type", default="train", type=str,
                        help="train or val")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)