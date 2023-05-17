import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

from ppdet.utils.logger import setup_logger

logger = setup_logger('dataset prepare')



def make_dataset_dir(dataset_root):
    data_root = os.path.join(dataset_root, 'channel_transmission')
    images_dir = os.path.join(data_root, 'images')
    annotations_dir = os.path.join(data_root, 'annotations')
    if not os.path.exists(images_dir):
        logger.info('making {} dir'.format(images_dir))
        os.mkdir(images_dir)
    if not os.path.exists(annotations_dir):
        logger.info('making {} dir'.format(annotations_dir))
        os.mkdir(annotations_dir)
    # vocdevkit_dir = os.path.join(voc_dir, 'VOCdevkit')
    # voc2007 = os.path.join(vocdevkit_dir, 'VOC2007')
    # annotations_dir = os.path.join(voc2007, 'Annotations')
    # jpegimages_dir = os.path.join(voc2007, 'JPEGImages')
    # imageset_dir = os.path.join(voc2007, 'ImageSets')
    # main_dir = os.path.join(imageset_dir, 'Main')
    # if not os.path.exists(voc_dir):
    #     logger.info('making {} dir'.format(voc_dir))
    #     os.mkdir(voc_dir)
    # if not os.path.exists(vocdevkit_dir):
    #     logger.info('making {} dir'.format(vocdevkit_dir))
    #     os.mkdir(vocdevkit_dir)
    # if not os.path.exists(voc2007):
    #     logger.info('making {} dir'.format(voc2007))
    #     os.mkdir(voc2007)
    # if not os.path.exists(annotations_dir):
    #     logger.info('making {} dir'.format(annotations_dir))
    #     os.mkdir(annotations_dir)
    # if not os.path.exists(jpegimages_dir):
    #     logger.info('making {} dir'.format(jpegimages_dir))
    #     os.mkdir(jpegimages_dir)
    # if not os.path.exists(imageset_dir):
    #     logger.info('making {} dir'.format(imageset_dir))
    #     os.mkdir(imageset_dir)
    # if not os.path.exists(main_dir):
    #     logger.info('making {} dir'.format(main_dir))
    #     os.mkdir(main_dir)

def make_coco(voc_root):
    coco_dir = os.path.join(voc_root, 'coco')
    annotations_dir = os.path.join(coco_dir, 'annotations')
    images_dir = os.path.join(coco_dir, 'images')
    if not os.path.exists(coco_dir):
        logger.info('making {} dir'.format(coco_dir))
        os.mkdir(coco_dir)
    if not os.path.exists(annotations_dir):
        logger.info('making {} dir'.format(annotations_dir))
        os.mkdir(annotations_dir)
    if not os.path.exists(images_dir):
        logger.info('making {} dir'.format(images_dir))
        os.mkdir(images_dir)


def split_dataset(dataset_root, count=False, make_coco=False):
    train_dir = os.path.join(dataset_root, 'train')
    file_list = os.listdir(train_dir)
    imgs_list = []
    xmls_list = []
    for file_name in file_list:
        if '.jpg' in file_name:
            imgs_list.append(file_name)
            xml_name = file_name.replace('.jpg', '.xml')
            xmls_list.append(xml_name)

    if count:
        pass

    img_train, img_val, xml_train, xml_val = train_test_split(imgs_list, xmls_list, test_size=0.1)
    logger.info('spilt train dataset ------ number:{}'.format(len(img_train)))
    logger.info('spilt val dataset ------ number:{}'.format(len(img_val)))
    # main_dir = os.path.join(voc_root, 'voc','VOCdevkit', 'VOC2007', 'ImageSets', 'Main')
    data_root = os.path.join(dataset_root, 'channel_transmission')
    train_txt_path = os.path.join(data_root, 'train.txt')
    val_txt_path = os.path.join(data_root, 'valid.txt')


    with open(train_txt_path, 'w+') as f:
        for img_name, xml_name in zip(img_train, xml_train):
            img_name = os.path.join('./images', img_name)
            xml_name = os.path.join('./annotations', xml_name)
            f.writelines(img_name + ' ' + xml_name + '\n')

    with open(val_txt_path, 'w+') as f:
        for img_name, xml_name in zip(img_val, xml_val):
            img_name = os.path.join('./images', img_name)
            xml_name = os.path.join('./annotations', xml_name)
            f.writelines(img_name + ' ' + xml_name + '\n')

    # annotations_dir = os.path.join(voc_root, 'voc/VOCdevkit/VOC2007/Annotations')
    # jpegimages_dir = os.path.join(voc_root, 'voc/VOCdevkit/VOC2007/JPEGImages')
    for img_name, xml_name in zip(imgs_list, xmls_list):
        img_path = os.path.join(train_dir, img_name)
        img_copy_to = os.path.join(data_root, 'images',img_name)
        xml_path = os.path.join(train_dir, xml_name)
        xml_copy_to = os.path.join(data_root, 'annotations', xml_name)
        shutil.copy(img_path, img_copy_to)
        shutil.copy(xml_path, xml_copy_to)
    #     if make_coco:
    #         coco_images_path = os.path.join(voc_root, 'coco/images')
    #         img_copy_to = os.path.join(coco_images_path, img_name)
    #         shutil.copy(img_path, img_copy_to)
    logger.info('VOC dataset has been done!')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=str,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--count",
        action='store_true',
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--make_coco",
        action='store_true',
        help="Directory for images to perform inference on.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    make_dataset_dir(args.dataset_root)
    # if args.make_coco:
    #     make_coco(args.voc_root)
    split_dataset(args.dataset_root, args.count, args.make_coco)




