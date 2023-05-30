import os
import shutil
import argparse
import numpy as np
import cv2 as cv
import xml.dom.minidom as xmldom
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from ppdet.utils.logger import setup_logger

logger = setup_logger('dataset prepare')

class Resize():
    def __init__(self, target_size, keep_ratio, interp=cv.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_area(self, area, scale):
        im_scale_x, im_scale_y = scale
        return area * im_scale_x * im_scale_y


    def __call__(self, im, bbox):
        """ Resize the image numpy.
        """
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))

        # apply image
        if len(im.shape) == 3:
            im_shape = im.shape
        else:
            im_shape = im[0].shape

        if self.keep_ratio:
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = int(im_scale * float(im_shape[0]) + 0.5)
            resize_w = int(im_scale * float(im_shape[1]) + 0.5)

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        if len(im.shape) == 3:
            im = self.apply_image(im, [im_scale_x, im_scale_y])
            # im = im.astype(np.float32)
        else:
            resized_images = []
            for one_im in im:
                applied_im = self.apply_image(one_im, [im_scale_x, im_scale_y])
                resized_images.append(applied_im)

            im = np.array(resized_images)

        bbox = self.apply_bbox(bbox,
                            [im_scale_x, im_scale_y],
                            [resize_w, resize_h])
        return im, bbox

def make_dataset_dir(dataset_root):
    data_root = os.path.join(dataset_root, 'channel_transmission')
    train_images_dir = os.path.join(data_root, 'train_images')
    val_images_dir = os.path.join(data_root, 'val_images')
    annotations_dir = os.path.join(data_root, 'annotations')
    if not os.path.exists(train_images_dir):
        logger.info('making {} dir'.format(train_images_dir))
        os.mkdir(train_images_dir)
    if not os.path.exists(val_images_dir):
        logger.info('making {} dir'.format(val_images_dir))
        os.mkdir(val_images_dir)
    if not os.path.exists(annotations_dir):
        logger.info('making {} dir'.format(annotations_dir))
        os.mkdir(annotations_dir)


def split_dataset(dataset_root):
    resize_op = Resize(target_size=[800, 1333], keep_ratio=True)

    train_dir = os.path.join(dataset_root, 'train')
    file_list = os.listdir(train_dir)
    imgs_list = []
    xmls_list = []
    for file_name in file_list:
        if '.jpg' in file_name:
            imgs_list.append(file_name)
            xml_name = file_name.replace('.jpg', '.xml')
            xmls_list.append(xml_name)

    category_list = {'nest':[], 'kite':[], 'balloon':[], 'trash':[]}
    for img_name, xml_name in zip(imgs_list, xmls_list):
        xml_path = os.path.join(train_dir, xml_name)
        xml_file = xmldom.parse(xml_path)
        eles = xml_file.documentElement
        eles.getElementsByTagName("filename")[0].firstChild.data = img_name
        object = eles.getElementsByTagName('object')[0]
        category = object.getElementsByTagName('name')[0].childNodes[0].data
        category_list[category].append(xml_file)
        with open(xml_path, 'w', encoding='utf-8') as f:
            xml_file.writexml(f, encoding='utf-8')
    logger.info('xml filename correct!')


    logger.info('nest category number : {}'.format(len(category_list['nest'])))
    logger.info('kite category number : {}'.format(len(category_list['kite'])))
    logger.info('balloon category number : {}'.format(len(category_list['balloon'])))
    logger.info('trash category number : {}'.format(len(category_list['trash'])))
    val_category_num = {'nest':int(len(category_list['nest']) * 0.1),
                        'kite':int(len(category_list['kite']) * 0.1),
                        'balloon':int(len(category_list['kite']) * 0.1),
                        'trash':int(len(category_list['trash']) * 0.1)}
    img_train, img_val, xml_train, xml_val = [], [], [], []
    for img_name, xml_name in zip(imgs_list, xmls_list):
        xml_path = os.path.join(train_dir, xml_name)
        xml_file = xmldom.parse(xml_path)
        eles = xml_file.documentElement
        object = eles.getElementsByTagName('object')[0]
        category = object.getElementsByTagName('name')[0].childNodes[0].data
        if val_category_num[category]:
            val_category_num[category] = val_category_num[category] - 1
            img_val.append(img_name)
            xml_val.append(xml_name)
        else:
            img_train.append(img_name)
            xml_train.append(xml_name)

    # img_train, img_val, xml_train, xml_val = train_test_split(imgs_list, xmls_list, test_size=0.1)
    logger.info('spilt train dataset ------ number:{}'.format(len(img_train)))
    logger.info('spilt val dataset ------ number:{}'.format(len(img_val)))
    # # main_dir = os.path.join(voc_root, 'voc','VOCdevkit', 'VOC2007', 'ImageSets', 'Main')
    data_root = os.path.join(dataset_root, 'channel_transmission')
    train_txt_path = os.path.join(data_root, 'train.txt')
    val_txt_path = os.path.join(data_root, 'val.txt')

    with open(train_txt_path, 'w+') as f:
        for img_name, xml_name in zip(img_train, xml_train):
            img_name = os.path.join('./train_images', img_name)
            xml_name = os.path.join('./annotations', xml_name)
            f.writelines(img_name + ' ' + xml_name + '\n')

    with open(val_txt_path, 'w+') as f:
        for img_name, xml_name in zip(img_val, xml_val):
            img_name = os.path.join('./val_images', img_name)
            xml_name = os.path.join('./annotations', xml_name)
            f.writelines(img_name + ' ' + xml_name + '\n')

    for img_name, xml_name in tqdm(zip(img_train, xml_train), total=len(img_train)):
        img_path = os.path.join(train_dir, img_name)
        img_copy_to = os.path.join(data_root, 'train_images', img_name)
        xml_path = os.path.join(train_dir, xml_name)
        xml_copy_to = os.path.join(data_root, 'annotations', xml_name)
        img = cv.imread(img_path)
        xml_file = xmldom.parse(xml_path)
        eles = xml_file.documentElement
        object = eles.getElementsByTagName("object")
        for obj in object:
            bndbox = obj.getElementsByTagName('bndbox')[0]
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
            bndbox.getElementsByTagName('xmin')[0].childNodes[0].data = xmin
            bndbox.getElementsByTagName('ymin')[0].childNodes[0].data = ymin
            bndbox.getElementsByTagName('xmax')[0].childNodes[0].data = xmax
            bndbox.getElementsByTagName('ymax')[0].childNodes[0].data = ymax

        with open(xml_copy_to, 'w', encoding='utf-8') as f:
            xml_file.writexml(f, encoding='utf-8')
        cv.imwrite(img_copy_to, img)

    for img_name, xml_name in tqdm(zip(img_val, xml_val), total=len(img_val)):
        img_path = os.path.join(train_dir, img_name)
        img_copy_to = os.path.join(data_root, 'val_images', img_name)
        xml_path = os.path.join(train_dir, xml_name)
        xml_copy_to = os.path.join(data_root, 'annotations', xml_name)
        img = cv.imread(img_path)
        xml_file = xmldom.parse(xml_path)
        eles = xml_file.documentElement
        object = eles.getElementsByTagName("object")
        for obj in object:
            bndbox = obj.getElementsByTagName('bndbox')[0]
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
            bndbox.getElementsByTagName('xmin')[0].childNodes[0].data = xmin
            bndbox.getElementsByTagName('ymin')[0].childNodes[0].data = ymin
            bndbox.getElementsByTagName('xmax')[0].childNodes[0].data = xmax
            bndbox.getElementsByTagName('ymax')[0].childNodes[0].data = ymax

        with open(xml_copy_to, 'w', encoding='utf-8') as f:
            xml_file.writexml(f, encoding='utf-8')
        cv.imwrite(img_copy_to, img)
    logger.info('VOC dataset has been done!')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default='../dataset',
        help="Directory for images to perform inference on.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    make_dataset_dir(args.dataset_root)
    split_dataset(args.dataset_root)
