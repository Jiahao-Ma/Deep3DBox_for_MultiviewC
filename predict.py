from nets.faster_rcnn_vgg16 import FasterRCNNVGG16
from utils.config import opt
import torch
from data.dataset import Dataset
from data.MultiviewCdataset import DetectedObject
from data.util import inverse_normalize, generate_bins, compute_3d_bbox, draw_3DBBox
from library.math import cal_location
from utils import array_tool as at
from nets.model import Deep3DBox
import matplotlib.pyplot as plt
import numpy as np

def predict(idx):
    print('load MultiviewC dataset.')
    dataset = Dataset(opt)

    # load faster_rcnn and Deep3DBox model
    faster_rcnn = FasterRCNNVGG16().cuda()
    print('faster_rcnn construct completed.')
    state_dict_faster_rcnn = torch.load(opt.faster_rcnn_path)
    print('load checkpoint from %s '%opt.faster_rcnn_path)
    faster_rcnn.load_state_dict(state_dict_faster_rcnn['model'])

    deep3dbox = Deep3DBox(opt.bins).cuda()
    print('Deep3DBox constructed completed.')
    state_dict_deep3dbox = torch.load(opt.model_path)
    deep3dbox.load_state_dict(state_dict_deep3dbox['model'])
    print('load checkpoint from %s '%opt.model_path)

    # load tested image
    img, labels, _ = dataset[idx]
    img = torch.Tensor(img)[None]
    img = img.cuda().float()
    ori_img = inverse_normalize(at.tonumpy(img[0]))
    K_matrix = labels[0]['K_matrix']
    P_matrix = labels[0]['P_matrix']
    Rz = labels[0]['Rz']
    angle_bins = generate_bins(2)

    # 2D detection
    _bboxes, _labels, _scores = faster_rcnn.predict([ori_img], visualize=True)
    _bboxes = _bboxes[0]
    _scores = _scores[0]

    # 3D detection 
    ori_img = np.transpose(ori_img, axes=(1, 2, 0)).astype(np.uint8)
    _orients = list()
    # _confs = list()
    _dims = list()
    _theta_rays = list()
    deep3dbox.eval()
    for bbox in _bboxes:
        obj = DetectedObject(ori_img, 'Cow', bbox, K_matrix)
        with torch.no_grad():
            crop_img = obj.img[None].cuda()
            [orient, conf, dim] = deep3dbox(crop_img)
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :] + dataset.db.averages.get_item('Cow')
            orient = orient.cpu().data.numpy()[0, :, :]
            
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi # shift the angle [0, 2*pi] to the angle [-pi, pi]

            _orients.append(alpha)
            # _confs.append(conf)
            _dims.append(dim)
            _theta_rays.append(obj.theta_ray)

    # visualization
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.imshow(ori_img)
    ax.axis('off')
    for i in range(len(_bboxes)):
        # visualize 2D
        bbox = _bboxes[i]
        score = _scores[i]
        bbox = [int(b) for b in bbox]
        width = bbox[3] - bbox[1]
        height = bbox[2] - bbox[0]
        rect = plt.Rectangle([bbox[1], bbox[0]], width, height, fill=False, linewidth=3)
        ax.add_patch(rect)
        ax.text(bbox[1], bbox[0]-5, 'Cow {:.2f}'.format(score), style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
        
        # visualize 3D
        alpha = _orients[i]
        theta_ray = _theta_rays[i]
        """
        # [FORMULA]
        # theta_ref_global = theta_w_global + 90
        # theta_c_global = theta_ref_global - R_z
        # theta_c_global = theta_local + theta_ray
        """
        theta_w_global = alpha + theta_ray + Rz - 90
        dimension = _dims[i]
        bbox = np.array(bbox).reshape((2,2))
        best_loc, _ = cal_location(dimension, P_matrix, bbox, alpha, theta_w_global)
        proj_3dbox = compute_3d_bbox(dimension, theta_w_global, best_loc, P_matrix)
        ax.scatter(proj_3dbox[:,0], proj_3dbox[:, 1], c='yellow', s = 5)
        ax = draw_3DBBox(ax, proj_3dbox)

    plt.show()

if __name__ == '__main__':
    predict(5)