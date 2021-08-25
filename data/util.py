import numpy as np
from PIL import Image
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
import math, json
from utils import array_tool as at

def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def vis_colors(classes):
    hsv_tuples = [(x / len(classes), 1., 1.) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list( map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors) )
    return hsv_tuples, colors

def vis_styles(facecolor='red', alpha=0.5, size=10, color='black'):
    box = {"facecolor": facecolor, "alpha": alpha}
    styles = {"size": size, "color": color, "bbox": box}
    return styles

def get_worldgrid_from_worldcoord(worldcoord, len_of_each_grid = 100):
    coord_x, coord_y, coord_z = worldcoord
    coord_x = coord_x * len_of_each_grid
    coord_y = coord_y * len_of_each_grid
    coord_z = coord_z * len_of_each_grid
    return np.array([coord_x, coord_y, coord_z], dtype=np.float32)


def get_worldcoord_for_imagecoord(worldcoord, len_of_each_grid = 100):
    if isinstance(worldcoord, list):
        worldcoord = [c / len_of_each_grid for c in worldcoord]
        return worldcoord
    worldcoord = worldcoord / len_of_each_grid
    return worldcoord
    
def corners8_to_rect4(corners8):
    xmin = np.min(corners8[:, 0])
    ymin = np.min(corners8[:, 1])
    xmax = np.max(corners8[:, 0])
    ymax = np.max(corners8[:, 1])
    return [xmin, ymin, xmax, ymax]

def rotz(t):
    """ Rotation about z-axis """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project_to_image(corner_3d, calib):
    """ Project corner_3d `nx4 points` in camera rect coord to image2 plane
        Args:
            corner_3d: nx4 `numpy.ndarray`
            calib: camera projection matrix
        Returns:
            corner_2d: nx2 `numpy.ndarray` 2d points in image2
    """
    corner_2d = np.dot(calib, corner_3d.T)
    corner_2d[0, :] = corner_2d[0, :] / corner_2d[2, :]
    corner_2d[1, :] = corner_2d[1, :] / corner_2d[2, :]
    corner_2d = np.array(corner_2d, dtype=np.int)
    return corner_2d[0:2, :].T


def compute_3d_bbox(dimension, rotation, location, calib):
    h, w, l = dimension[0], dimension[1], dimension[2]
    x = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]
    z = [0, 0, 0, 0, h, h, h, h]
    # NOTICE! rotation: -np.pi ~ np.pi ! instead of -180 ~ 180
    rotMat = rotz(rotation)
    corner_3d = np.vstack([x, y, z])
    corner_3d = np.dot(rotMat, corner_3d)
    bottom_center = np.tile(location, (corner_3d.shape[1], 1)).T
    corner_3d = corner_3d + bottom_center
    corner_3d_homo = np.vstack([corner_3d, np.ones((1, corner_3d.shape[1]))])
    corner_2d = project_to_image(corner_3d_homo.T, calib)
    return corner_2d

def draw_3DBBox(ax, corners, edgecolor=(0, 1, 0), linewidth=1):
    if len(corners) != 8:
        return ax
    assert corners.shape[1] == 2, 'corners` shape should be [8, 2]'
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        ax.plot((corners[i, 0], corners[j, 0]), (corners[i, 1], corners[j, 1]), color=edgecolor, linewidth=linewidth)
        i, j = k + 4, (k + 1) % 4 + 4
        ax.plot((corners[i, 0], corners[j, 0]), (corners[i, 1], corners[j, 1]), color=edgecolor, linewidth=linewidth)
        i, j = k, k + 4
        ax.plot((corners[i, 0], corners[j, 0]), (corners[i, 1], corners[j, 1]), color=edgecolor, linewidth=linewidth)
    return ax

def get_P(calib_path):
    with open(calib_path, 'r') as f:
        params = json.loads(f.read())
    return params['P']

def get_K(calib_path):
    with open(calib_path, 'r') as f:
        params = json.loads(f.read())
    return params['K']

def get_Rz(calib_path):
    with open(calib_path, 'r') as f:
        params = json.loads(f.read())
    return params['R_z']

def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def inverse(img):
    # get the original image
    # steps: (1) convert to nump (2) inverse normalize (3)transpose and change data type
    img = at.tonumpy(img)
    img = inverse_normalize(img)
    return np.transpose(img, axes=[1, 2, 0]).astype(np.uint8)

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins