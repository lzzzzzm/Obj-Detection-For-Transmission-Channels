import os
import shutil
import argparse
import cv2 as cv
import xml.dom.minidom as xmldom

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

    for img_name, xml_name in zip(imgs_list, xmls_list):
        xml_path = os.path.join(train_dir, xml_name)
        xml_file = xmldom.parse(xml_path)
        eles = xml_file.documentElement
        eles.getElementsByTagName("filename")[0].firstChild.data = img_name
        with open(xml_path, 'w', encoding='utf-8') as f:
            xml_file.writexml(f, encoding='utf-8')
    logger.info('xml filename correct!')

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

    for img_name, xml_name in zip(imgs_list, xmls_list):
        img_path = os.path.join(train_dir, img_name)
        img_copy_to = os.path.join(data_root, 'images',img_name)
        xml_path = os.path.join(train_dir, xml_name)
        xml_copy_to = os.path.join(data_root, 'annotations', xml_name)
        img = cv.imread(img_path)
        xml_file = xmldom.parse(xml_path)
        eles = xml_file.documentElement
        bndbox = eles.getElementsByTagName("object")[0].getElementsByTagName('bndbox')[0]
        size = eles.getElementsByTagName("size")[0]
        width = size.getElementsByTagName("width")[0].childNodes[0].data
        height = size.getElementsByTagName("height")[0].childNodes[0].data
        xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)
        if xmin > xmax:
            logger.info('invalid xml file:{} xmin:{} while xmax:{}'.format(img_path, xmin, xmax))
            t = xmax
            xmax = xmin
            xmin = t
        if ymin > ymax:
            logger.info('invalid xml file:{} ymin:{} while ymax:{}'.format(img_path, ymin, ymax))
            t = ymax
            ymax = ymin
            ymin = t
        w = img.shape[1]
        h = img.shape[0]
        resize_ratio_w = w/1024
        resize_ratio_h = h/1024
        xmin = int(xmin/resize_ratio_w)
        xmax = int(xmax/resize_ratio_w)
        ymin = int(ymin/resize_ratio_h)
        ymax = int(ymax/resize_ratio_h)
        if img.shape[0] > 1024 and img.shape[1] > 1024:
            img = cv.resize(img, (1024, 1024))
            bndbox.getElementsByTagName('xmin')[0].childNodes[0].data = xmin
            bndbox.getElementsByTagName('ymin')[0].childNodes[0].data = ymin
            bndbox.getElementsByTagName('xmax')[0].childNodes[0].data = xmax
            bndbox.getElementsByTagName('ymax')[0].childNodes[0].data = ymax
            size.getElementsByTagName("width")[0].childNodes[0].data = 1024
            size.getElementsByTagName("height")[0].childNodes[0].data = 1024
            with open(xml_path, 'w', encoding='utf-8') as f:
                xml_file.writexml(f, encoding='utf-8')
            # cv.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0))
            # cv.imshow('img', img)
            # cv.waitKey()
        cv.imwrite(img_copy_to, img)
        # shutil.copy(img_path, img_copy_to)
        shutil.copy(xml_path, xml_copy_to)
    logger.info('VOC dataset has been done!')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default='../dataset',
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
    split_dataset(args.dataset_root, args.count, args.make_coco)




