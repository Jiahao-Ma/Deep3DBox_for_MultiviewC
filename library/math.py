import os, sys
sys.path.append(os.getcwd())
from data.util import rotz, get_worldcoord_for_imagecoord, get_worldgrid_from_worldcoord
import numpy as np
from data.util import compute_3d_bbox, corners8_to_rect4, draw_3DBBox
def cal_location(dimension, proj_matrix, bbox_2d, alpha, orient):
    """
        Args:
            dimension: `list` contains height, width, length
            proj_matrix: intrinsic matrix P
            bbox_2d: `np.ndarray` size: [2,2]; data format: [[ymin, xmin], [ymax, xmax]]
            alpha: local orientation
            theta_ray: the angle between ray from object's center to camera and the x axis of camera
        
        [FORMULA]   theta_global = theta_local + theta_ray
                    theta_global: global orientation (variable `orient`)
                    theta_local: local orientation (variable `alpha`)

        Coordinate of MultiviewC Dataset
                z
                |   x
                |  /
                | /
              0 |/__________ y

    """

    R  = rotz(orient)

    # format 2d corners
    ymin = bbox_2d[0][0]
    xmin = bbox_2d[0][1]
    ymax = bbox_2d[1][0]
    xmax = bbox_2d[1][1]

    box_corners = [xmin, ymin, xmax, ymax]

    constraints = []

    height, width, length = dimension
    dx = length / 2
    dy = width / 2
    dz = height #/ 2
    CORNERS = {
        1: [dx, -dy, 0],
        2: [dx, dy, 0],
        3: [-dx, dy, 0],
        4: [-dx, -dy, 0],
        5: [dx, -dy, dz],
        6: [dx, dy, dz],
        7: [-dx, dy, dz],
        8: [-dx, -dy, dz],
    }

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []
    constraints = []
    
    # keep alpha in -180 ~ 180
    if alpha < np.deg2rad(-180):
        alpha += 2 * np.pi
    elif alpha > np.deg2rad(180):
        alpha -= 2 * np.pi
    # define left_constraints and right_constraints according to 4 cases
    # case1 85 <= alpah <= 95
    if alpha >  np.deg2rad(85) and alpha < np.deg2rad(95):
        left_constraints += [CORNERS[4], CORNERS[8]]
        right_constraints += [CORNERS[3], CORNERS[7]]
    # case2 -95 <= alpah <= -85
    elif alpha < np.deg2rad(-85) and alpha > np.deg2rad(-95):
        left_constraints += [CORNERS[2], CORNERS[6]]
        right_constraints += [CORNERS[1], CORNERS[5]]

    # case3 -90 <= alpah <= 90
    if alpha > np.deg2rad(-90) and alpha < np.deg2rad(0):
        left_constraints += [CORNERS[2], CORNERS[6]]
        right_constraints += [CORNERS[4], CORNERS[8]]
    elif alpha > np.deg2rad(0) and alpha < np.deg2rad(90):
        left_constraints += [CORNERS[1], CORNERS[5]]
        right_constraints += [CORNERS[3], CORNERS[7]]
    # case4 -180 <= alpah <= -90 / 90 <= alpah <= 180
    elif alpha > np.deg2rad(-180) and alpha < np.deg2rad(-90):
        left_constraints += [CORNERS[3], CORNERS[7]]
        right_constraints += [CORNERS[1], CORNERS[5]]
    elif alpha > np.deg2rad(90) and alpha < np.deg2rad(180):
        left_constraints += [CORNERS[4], CORNERS[8]]
        right_constraints += [CORNERS[2], CORNERS[6]]

    # define top_constraints and bottom_constraints
    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i * dx, j * dy, dz])
            bottom_constraints.append([i * dx, j * dy, 0])

    # 64 constraints
    for left in left_constraints:
        for right in right_constraints:
            for top in top_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    
    # filter out the one with repeats
    constraints = list(filter(lambda x: len(x) == len(set( tuple(i) for i in x )), constraints) )

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
 
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3], dtype=np.float)
        b = np.zeros([4,1])

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3,3] = RX.reshape(3)

            M = np.dot(proj_matrix, M)

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=True)

        # found a better estimation
        if error < best_error:
            best_loc = loc
            best_error = error
            best_X = X_array

    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc, best_X

if __name__ == "__main__":
    from data.dataset import MultiviewC_dataset, inverse
    from data.util import project_to_image
    import matplotlib.pyplot as plt
    MC = MultiviewC_dataset(bbox_mode='2D', train_mode='train')
    for i in range(0, 7):
        labels, image = MC.get_example_batch(i)
        error = 0
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111)
        ax.imshow(image)

        for label in labels:
            dimension = label['Dimension'] + MC.averages.get_item('Cow')
            P_matrix = label['P_matrix']
            alpha = label['Alpha']
            theta_ray = label['theta_ray']
            orient = label['Orient_w']
            theta_c_global = label['Orient_c']
            location = label['Location']
            bbox_2d = label['Box_2d']
            best_loc, best_X = cal_location(get_worldcoord_for_imagecoord(dimension), P_matrix, bbox_2d, alpha, orient)
            error += sum(np.array(best_loc) - np.array(get_worldcoord_for_imagecoord(location)))
            projected_bbox_2d = compute_3d_bbox(dimension, orient, location, P_matrix) #get_worldgrid_from_worldcoord(best_loc)
            [ymin, xmin, ymax, xmax] = bbox_2d.reshape(-1)
            width = int(xmax - xmin)
            height = int(ymax - ymin)
            rect = plt.Rectangle([xmin, ymin], width, height, fill=None, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.scatter(projected_bbox_2d[:,0], projected_bbox_2d[:, 1], c='yellow', s = 5)
            ax = draw_3DBBox(ax, projected_bbox_2d)
        print('location mean error:{:.2f}'.format(error / len(labels)))
        ax.axis('off')
        plt.show()

