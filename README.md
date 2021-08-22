# Deep3DBBox_for_MultiviewC
 Pytorch implementation for Deep3DBox on [MultiviewC dataset](https://github.com/Robert-Mar/MultiviewC).
 
## Visualization 
![alt text](https://github.com/Robert-Mar/Deep3DBox_for_MultiviewC/blob/main/results/C0.png "Visualization of Camera1")
![alt text](https://github.com/Robert-Mar/Deep3DBox_for_MultiviewC/blob/main/results/C6.png "Visualization of Camera6")

## Angle Adjustment
Different from [Kitti](http://www.cvlibs.net/datasets/kitti/) dataset, MultiviewC contains seven cameras located at four corners of field.There are several angle information thate need to be explained.

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

## Reference
* 3D Bounding Box Estimation Using Deep Learning and Geometry. [Paper](https://arxiv.org/abs/1612.00496).
* PyTorch implementation for this paper. [Link](https://github.com/skhadem/3D-BoundingBox).
