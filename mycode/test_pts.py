

import numpy as np
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


def load_res_data(filename):
    data_list = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            #data = np.array([float(s) for s in line.split(" ")])
            data_list.append( np.fromstring(line, dtype=float, sep=' ') )
    return np.stack(data_list, axis = 0)


# load detecting results with 
pts_res = load_res_data('res_03.txt')
#draw_lidar(pts_res)
choice = np.where(pts_res[:,3]>0)
pts_car = pts_res[choice, :].squeeze()
#pts_car = np.multiply(pts_res, pts_res[:,3].reshape(-1,1))
#draw_lidar(pts_car)



file_dir = "train_03.bin"
pc = np.fromfile(file_dir, dtype=np.float32).reshape(-1, 4) # load velodyne data
#x_flag = pc[:,0]>0
#pc = pc[x_flag, :]
#fig = draw_lidar(pc)


'''

mlab.close()

'''

'''

# Create the data.
from numpy import pi, sin, cos, mgrid
dphi, dtheta = pi/250.0, pi/250.0
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
x = r*sin(phi)*cos(theta)
y = r*cos(phi)
z = r*sin(phi)*sin(theta)

# View it.
from mayavi import mlab
s = mlab.mesh(x, y, z)
mlab.show()


'''


