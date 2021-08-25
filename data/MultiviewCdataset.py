#TODO: parse fname, annotaion, calib
#TODO: generate bins and bin_ranges
#TODO: calculate ClassAverage
#TODO: dataloader: generate image_crop, proj_matrix, theta_ray, label, detection_class
import json, os, sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy.lib.type_check import imag
from torchvision import transforms
sys.path.append(os.getcwd())
from data.ClassAverage import ClassAverage
from data.util import draw_3DBBox, vis_colors, vis_styles, compute_3d_bbox, corners8_to_rect4, read_image, get_worldcoord_for_imagecoord
from data.util import get_K, get_P, get_Rz, inverse

MULTIVIEWC_BBOX_LABEL_NAMES = ['Cow']

class MultiviewC_dataset(object):
    def __init__(self, root = r'F:\ANU\ENGN8602\Data\MultiviewC_dataset', # PATH of MultiviewC dataset
                       json_root=r'annotations', 
                       img_root =r'images', 
                       calib_root=r'calibrations', 
                       cam_range=range(1, 8),
                       img_size=(1280, 720),
                       save_img_ann=False,
                       read_img_ann = True,
                       split=0.9,
                       mode='train',
                       bins = 2, 
                       overlap = 0.1,
                ) -> None:
        super().__init__()
        """
            json_root: annotation path
            img_root: image path
            calib_root: calibration path
            cam_range: default value： range(1, 8), represent the camera ID
            mode: determine the data format `2D` or `3D`
            img_size: image size (W, H)
        """
        assert mode in ['train', 'test'], 'train mode must be `train` or `test`.'
        self.img_size = img_size
        self.json_root = os.path.join(root, json_root)
        self.img_root = os.path.join(root, img_root)
        self.calib_root = os.path.join(root, calib_root)
        self.cam_range = cam_range
        self.label_names = MULTIVIEWC_BBOX_LABEL_NAMES
        
        calib_fnames = [os.path.join(self.calib_root, 'Camera{}\\parameters.json'.format(cam_id)) for cam_id in self.cam_range]
        self.K_calib_list = list()
        self.P_calib_list = list()
        self.Rz_calib_list = list()
        for i in range(len(calib_fnames)):
            self.K_calib_list.append(get_K(calib_fnames[i]))
            self.P_calib_list.append(get_P(calib_fnames[i]))
            self.Rz_calib_list.append(get_Rz(calib_fnames[i]))

        if not read_img_ann:
            self.images_list, self.annotations_list = self.generate()
        else:
            self.images_list, self.annotations_list = self.read_in_dick()
        if save_img_ann:
            self.write_in_dick()
        self.ids = range(len(self.images_list))

        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = np.pi * 2 / bins
        for i in range(1, bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        self.bin_ranges = []
        for i in range(0, bins):
            self.bin_ranges.append( ((i * self.interval - overlap) % (np.pi*2),\
                                    (i * self.interval + self.interval + overlap) % (np.pi*2)) )
        self.averages = ClassAverage(classes=MULTIVIEWC_BBOX_LABEL_NAMES)
        self.object_list = self.get_object(self.ids)
        self.cur_id = None # used to read current image
        if mode == 'train':
            self.object_list = self.object_list[:int(split * len(self.object_list))]
        else:
            self.object_list = self.object_list[int(split * len(self.object_list)):]

    def get_object(self, ids):
        """ calculate dimension average and return the object list
            objects format: [[img_id, cow_id],
                             [ , ] , 
                             ....]
        """
        objects = list()
        for id in ids:
            annotation = self.annotations_list[id]
            for cow_id, ann in enumerate(annotation):
                dimension = np.array(ann['dimension'], dtype=np.float32)
                # only one class `Cow`
                self.averages.add_item('Cow', dimension)
                objects.append((id, cow_id))
        self.averages.dump_to_file()
        return objects

    def write_in_dick(self, spath=r'./data/annotations.json'):
        with open(spath, 'w') as f:
            save_dict = dict()
            for img, ann in zip(self.images_list, self.annotations_list):
                save_dict[img] = ann
            json.dump(save_dict, f, indent=4)
        print('Complete saving.')
    
    def read_in_dick(self, spath=r'./data/annotations.json'):
        with open(spath, 'r') as f:
            save_dict = json.loads(f.read())
        images_list = list()
        annotations_list = list()
        for img, ann in save_dict.items():
            images_list.append(img)
            annotations_list.append(ann)
        print('Complete reading.')
        return images_list, annotations_list

    def generate(self):
        num_json_file = len(os.listdir(self.json_root))
        images_list = list()
        annotations_list = list()
        for index in range(num_json_file):
            json_fname = self.json_root + '\\{:04d}.json'.format(index)
            image_fnames = [ os.path.join(self.img_root, 'C{}\\{:04d}.png'.format(cam_id, index))for cam_id in self.cam_range]
            with open(json_fname, 'r') as f:
                annotations = json.load(f)
            for cam_id, annotation in enumerate(annotations.values()):
                P = self.P_calib_list[cam_id]
                ann_list = list()
                for ann in annotation:
                    visible = ann['visible']
                    if not visible:
                        continue
                    ann['dimension'] = get_worldcoord_for_imagecoord(ann['dimension'])
                    ann['location'] = get_worldcoord_for_imagecoord(ann['location'])
                    location = ann['location']
                    rotation = np.deg2rad(ann['rotation'])
                    dimension = ann['dimension']
                  
                    corner2d = compute_3d_bbox(dimension, rotation, location, P)
                    [xmin, ymin, xmax, ymax] = corners8_to_rect4(corner2d)
                    xmin = np.clip(xmin, 0, self.img_size[0]).astype(float)
                    xmax = np.clip(xmax, 0, self.img_size[0]).astype(float)
                    ymin = np.clip(ymin, 0, self.img_size[1]).astype(float)
                    ymax = np.clip(ymax, 0, self.img_size[1]).astype(float)
                    ann['bbox2d'] = [ymin, xmin, ymax, xmax]
                    ann_list.append(ann)

                annotations_list.append(ann_list)
                images_list.append(image_fnames[cam_id])
        return images_list, annotations_list
                
    def get_example(self, index):
        img_id = self.object_list[index][0]
        cow_id = self.object_list[index][1]
        cam_id = img_id % len(self.K_calib_list)

        K_matrix = self.K_calib_list[cam_id]
        P_matrix = self.P_calib_list[cam_id]
        Rz = self.Rz_calib_list[cam_id]
        fname = self.images_list[img_id]
        label = self.annotations_list[img_id][cow_id]

        if self.cur_id != img_id:
            self.cur_id = img_id
            # self.cur_img = cv2.imread(fname)
            self.cur_img = np.array(Image.open(fname).convert('RGB'))
        
        obj = DetectedObject(self.cur_img, 'Cow', label['bbox2d'], K_matrix, label)
        label = self.format_label(label, obj.theta_ray, K_matrix, P_matrix, Rz)
        return obj.img, label 

    __getitem__ = get_example

    def format_label(self, label, theta_ray, K_matrix, P_matrix, Rz):
        """
        # theta_w_global [Orient]: cattle's global orientation in world coordinate
        #
        # theta_ref_global: cattle's global orientation in reference camera coordinate
        #
        # theta_c_global : cattle's global orientation in specific camera coordinate  
        #
        # theta_local [Alpha]: cattle's local orientation in specific camera coordinate 
        #
        # theta_ray: the angle between the ray from cammera center to objects' center 
        #            and the y axis of camera.  (angle of camera coordinate)
        #
        # Rz: the rotation angle of camera on Z-axis of the world coordinate
        #
        # [FORMULA]
        # theta_ref_global = theta_w_global + 90
        # theta_c_global = theta_ref_global - R_z
        # theta_c_global = theta_local + theta_ray
        """
        
        # Action = label['action']
        Location = label['location']
        Rotation = label['rotation']
        theta_w_global = np.deg2rad(Rotation)
        Dimension = label['dimension']
        Bbox2d = np.array(label['bbox2d']).reshape((2, 2))
        # calculate the offset of the dimension
        Dimension -= self.averages.get_item('Cow')
        # define orientation and confidence
        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)
        # calculate theta_local 
        if Rz < 0:
            Rz += 360
        theta_c_global = (theta_w_global + np.deg2rad(90)) - np.deg2rad(Rz)
        theta_local = theta_c_global - theta_ray
        
        # alpha: [-pi, pi], shift to angle: [0, 2*pi]
        angle = theta_local + np.pi

        bin_idxs = self.get_bin(angle)
        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]
            Orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
                'Class' : label['CowID'],
                'Box_2d' : Bbox2d,
                'Location' : Location,
                'Dimension' : Dimension,
                'Alpha' : theta_local, 
                'theta_ray' : theta_ray,
                'Orient_w' : theta_w_global,
                'Orient_c' : theta_c_global,
                'Rz' : Rz,
                'Orientation' : Orientation,
                'Confidence' : Confidence,
                'K_matrix' : K_matrix,
                'P_matrix' : P_matrix
                }
        return label 

    def __len__(self):
        return len(self.object_list)

    def get_example_batch(self, img_index):
        temp_object_list = np.array(self.object_list)
        mask = temp_object_list[:, 0] == int(img_index)
        batch_list = temp_object_list[mask]
        cam_id = img_index % len(self.K_calib_list)
        K_matrix = self.K_calib_list[cam_id]
        P_matrix = self.P_calib_list[cam_id]
        Rz = self.Rz_calib_list[cam_id]

        labels = list()
        for (img_id, cow_id) in batch_list:
            fname = self.images_list[img_id]
            label = self.annotations_list[img_id][cow_id]

            if self.cur_id != img_id:
                self.cur_id = img_id
                self.cur_img = np.array(Image.open(fname).convert('RGB'))
            
            obj = DetectedObject(self.cur_img, 'Cow', label['bbox2d'], K_matrix, label)
            labels.append(self.format_label(label, obj.theta_ray, K_matrix, P_matrix, Rz))

        return labels, self.cur_img


    def get_bin(self, angle):
        bin_idxs = []
        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max
        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)
        return bin_idxs


class DetectedObject(object):
    def __init__(self, img, detection_class, bbox_2d, K_matrix, label=None) -> None:
        """ 
        Args:
            img: [H,W,C] bgr image (read by opencv2)
            bbox: [[ymin, xmin], [ymax, xmax]] 2D bounding box in image
            detection_class: default class cow
            proj_matrix / K_matrix: project_matrix or intrinsic matrix? #TODO:double check!
            label: annotation information
        """
        super().__init__()
        if np.array(bbox_2d).shape != (2, 2): 
            # ensure the shape of bbox2d is 2 by 2 and its data type is int
            bbox_2d = np.array(bbox_2d).reshape(2, 2).astype(np.int32)
        self.theta_ray = self.cal_theta_ray(img, bbox_2d, K_matrix)
        self.img = self.format_image(img, bbox_2d)
        self.label = label
        self.detection_class = detection_class
    
    def cal_theta_ray(self, img, bbox, proj_matrix):
        """
            Args:
                img: [H,W,C] bgr image (read by opencv2)
                bbox: [[ymin, xmin], [ymax, xmax]] 2D bounding box in image
                proj_matrix: project_matrix or intrinsic matrix? #TODO:double check!

        """
        fx = proj_matrix[0][0]
        width = img.shape[1]
        center = (bbox[0][1] + bbox[1][1]) / 2
        dx = center - width / 2
        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        theta_ray = np.arctan( dx / fx)
        theta_ray *= mult
        return theta_ray
    
    def format_image(self, img, bbox, H = 224, W = 224):
        """
            Args:
                img: [H,W,C] bgr image (read by opencv2)
                bbox: [[ymin, xmin], [ymax, xmax]] 2D bounding box in image
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            normalize
        ])

        # crop image
        pt1 = bbox[0] #[ymin, xmin]
        pt2 = bbox[1] #[ymax, xmax]
        crop = img[pt1[0]:pt2[0]+1, pt1[1]:pt2[1]+1]
        iH, iW = crop.shape[:2]
        scale = min(H / iH, W / iW)
        nw = int(iW * scale)
        nh = int(iH * scale)
        crop = Image.fromarray(crop).resize((nw, nh), Image.BICUBIC)
        dy = (H - nh) // 2
        dx = (W - nw) // 2
        newcrop = Image.new('RGB', size=(W, H), color=(128, 128, 128))
        newcrop.paste(crop, (dx, dy))
        
        # recolor, reformat
        batch = process(np.array(newcrop))

        return batch

if __name__ == '__main__':
    from utils.config import opt
    from torch.utils import data
    # MC = MultiviewC_dataset(mode='test', save_img_ann=False, read_img_ann=True)
    dataset = MultiviewC_dataset( root = opt.root, split = opt.split, bins = opt.bins, mode='train')
    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    print(len(dataset))
    print(len(dataloader))