import torch
import numpy as np

import _init_path
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import tensorboard_logger as tb_log
from dataset import KittiDataset
import argparse
import importlib
from pointnet2_msg import Pointnet2MSG as pointnet2_msg

import kitti_utils

'''
import mayavi.mlab as mlab

# pts_mode='sphere'
def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=0.3,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=False,
    pc_label=False,
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    # ind = (pc[:,2]< -1.65)
    # pc = pc[ind]
    pts_mode = "point"
    print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    if color is None:
        color = pc[:, 2]
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        color = pc[:, 2]

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=pts_color,
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
    )

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )

    # draw fov (todo: update to real sensor spec.)
    fov = np.array(
        [[20.0, 20.0, 0.0, 0.0], [20.0, -20.0, 0.0, 0.0]], dtype=np.float64  # 45 degree
    )

    mlab.plot3d(
        [0, fov[0, 0]],
        [0, fov[0, 1]],
        [0, fov[0, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [0, fov[1, 0]],
        [0, fov[1, 1]],
        [0, fov[1, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )

    # draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d(
        [x1, x1],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x2, x2],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y1, y1],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y2, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )

    # mlab.orientation_axes()
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )
    return fig
'''



def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        #log_print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        #log_print("==> Done")
    else:
        raise FileNotFoundError

    return epoch


def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

def getdata(pts_lidar):

    '''
        pts_lidar: (N, 4)
    '''
    
    # get valid point (projected points should be in image)
    pts_rect = pts_lidar[:, 0:3]
    npoints = 16384
    
    calib = kitti_utils.Calibration('mycode/data/calib_03.txt')
    img_shape = (1242, 375, 3)
    
    # get valid point (projected points should be in image)
    pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
    #pts_intensity = pts_lidar[:, 3]

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    pts_valid_flag = get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
    
    pts_rect = pts_rect[pts_valid_flag][:, 0:3]
    #pts_intensity = pts_intensity[pts_valid_flag]
    
    if npoints < len(pts_rect):
        pts_depth = pts_rect[:, 2]
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        near_idxs_choice = np.random.choice(near_idxs, npoints - len(far_idxs_choice), replace=False)

        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
            if len(far_idxs_choice) > 0 else near_idxs_choice
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(pts_rect), dtype=np.int32)
        if npoints > len(pts_rect):
            extra_choice = np.random.choice(choice, npoints - len(pts_rect), replace=False)
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)

    ret_pts_rect = pts_rect[choice, :]
    # = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
    
    #pts_features = [ret_pts_intensity.reshape(-1, 1)]
    #ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]


    return choice, ret_pts_rect

'''
class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth
'''

def load_res(filename):
    res = np.fromfile(filename, dtype=np.float32, sep=' ')
    return res.reshape(-1,4).view()


if __name__== "__main__":
    
    FG_THRESH = 0.3
    
    # load model
    MODEL = importlib.import_module("pointnet2_msg")  # import network module
    model = MODEL.get_model(input_channels=0)
    #model = pointnet2_msg(input_channels=0)
      
    # load ckpt    
    ckpt = load_checkpoint(model, "mycode/checkpoint_epoch_100.pth")
    model.cuda()
    print("===> load checkpoint done")
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    pc = np.fromfile('mycode/data/train_03.bin', dtype=np.float32).reshape(-1,4)
    choice, pc_for_det = getdata(pc)
    pc_for_det = torch.from_numpy(pc_for_det.reshape(1,-1,3)).cuda(non_blocking=True).float()
        
    with torch.no_grad():
        pred_cls = model(pc_for_det)
    pred_class = (torch.sigmoid(pred_cls) > FG_THRESH).cpu().numpy()[0]  
    print("===> detection done")
    
    
    pts_rect = pc[choice, 0:3]
    res = np.concatenate((pts_rect, pred_class), axis = 1)
    print("res shape : ", res.shape)
    np.savetxt('res_03.txt', res)
    #draw_lidar(res)
    
    print(res)
        
    print("glk")