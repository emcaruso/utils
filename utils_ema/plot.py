import matplotlib.pyplot as plt
from .geometry_pose import *
from .geometry_euler import *
from .geometry_sphere import *

class plotter():

    @staticmethod
    def init_figure( ndim = 3, limit=1, title = "plot", figsize=(10,10) ):
        plotter.ndim = ndim
        if ndim==3:
            plotter.fig = plt.figure(figsize=figsize)
            plotter.ax = plotter.fig.add_subplot(111, projection='3d')
            plotter.ax.set_xlim([-limit, limit])
            plotter.ax.set_ylim([-limit, limit])
            plotter.ax.set_zlim([-limit, limit])
            plotter.ax.set_xlabel('X')
            plotter.ax.set_ylabel('Y')
            plotter.ax.set_zlabel('Z')
            plotter.ax.set_title(title)
            plotter.ax.set_box_aspect([1.0, 1.0, 1.0])
            # plotter.ax.set_aspect('equal')

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def plot_sphere(sphere, color='b', transparency=0.6):
        if plotter.ndim != 3:
            plotter.init_figure(ndim=3)

        l = sphere.frame.location()
        l = l.view( -1, l.shape[-1] )
        for i in range(l.shape[0]):
            theta, phi = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
            x = sphere.radius[i] * np.sin(phi) * np.cos(theta)
            y = sphere.radius[i] * np.sin(phi) * np.sin(theta)
            z = sphere.radius[i] * np.cos(phi)
            x += float(l[i,0])
            y += float(l[i,1])
            z += float(l[i,2])
            plotter.ax.plot_surface(x, y, z, color=color, alpha=transparency)

    @staticmethod
    def plot_points(points, color='r', label='Points', marker='o'):
        if plotter.ndim != points.shape[-1]:
            plotter.init_figure(ndim=points.shape[-1])

        if plotter.ndim == 3:
            points_np = points.numpy()
            x_points = points_np[..., 0]
            y_points = points_np[..., 1]
            z_points = points_np[..., 2]
            plotter.ax.scatter(x_points, y_points, z_points, c=color, marker=marker, label=label)

    @staticmethod
    def plot_line(start, end, color='m', linewidth=0.5,label='Lines'):
        if plotter.ndim != start.shape[-1]:
            plotter.init_figure(ndim=start.shape[-1])

        if plotter.ndim == 3:
            start = start.numpy()
            end = end.numpy()
            plotter.ax.plot3D([start[...,0], end[...,0]], [start[...,1], end[...,1]], [start[...,2], end[...,2]], linewidth=linewidth, linestyle='-', color=color, label='Line Segments')
            


    @staticmethod
    def plot_cam(camera, size=0.1):
        assert(plotter.ndim == 3)
        o = camera.frame.location()
        c00 = o+camera.pix2dir(torch.LongTensor([0,0]))*size
        c01 = o+camera.pix2dir(torch.LongTensor([0,camera.resolution[1]]))*size
        c10 = o+camera.pix2dir(torch.LongTensor([camera.resolution[0],0]))*size
        c11 = o+camera.pix2dir(torch.LongTensor([camera.resolution[0],camera.resolution[1]]))*size
        plotter.plot_line(o,c00)
        plotter.plot_line(o,c10)
        plotter.plot_line(o,c01)
        plotter.plot_line(o,c11)
        plotter.plot_line(c00,c01)
        plotter.plot_line(c00,c10, linewidth=1.5, color='darkmagenta')
        plotter.plot_line(c11,c01)
        plotter.plot_line(c11,c10)
        plotter.plot_points(o, color='darkmagenta')


    @staticmethod
    def plot_ray(origin, dir, color='g', label='Rays'):
        if plotter.ndim != origin.shape[-1]:
            plotter.init_figure(ndim=origin.shape[-1])

        if plotter.ndim == 3:
            origin = origin.numpy()
            dir = dir.numpy()
            plotter.ax.quiver(origin[..., 0], origin[..., 1], origin[..., 2], dir[..., 0], dir[..., 1], dir[..., 2], color=color, label='Rays')

    @staticmethod
    def plot_frame(frame):
        x_axis_end = torch.FloatTensor((1, 0, 0))
        y_axis_end = torch.FloatTensor((0, 1, 0))
        z_axis_end = torch.FloatTensor((0, 0, 1))
        x_axis_end_rot = torch.matmul(frame.rotation(), x_axis_end)
        y_axis_end_rot = torch.matmul(frame.rotation(), y_axis_end)
        z_axis_end_rot = torch.matmul(frame.rotation(), z_axis_end)
        a_x = frame.location()+x_axis_end_rot
        a_y = frame.location()+y_axis_end_rot
        a_z = frame.location()+z_axis_end_rot
        plotter.ax.plot([frame.location()[0], a_x[0]], [frame.location()[1], a_x[1]], [frame.location()[2], a_x[2]], 'r-', label='X-axis')
        plotter.ax.plot([frame.location()[0], a_y[0]], [frame.location()[1], a_y[1]], [frame.location()[2], a_y[2]], 'g-', label='Y-axis')
        plotter.ax.plot([frame.location()[0], a_z[0]], [frame.location()[1], a_z[1]], [frame.location()[2], a_z[2]], 'b-', label='Z-axis')




if __name__ == "__main__":
    p = plotter()
    p.init_figure()

    # # plot 2 spheres
    # f = frame( torch.stack((torch.eye(4, dtype=torch.float32),torch.eye(4, dtype=torch.float32))) )
    # f.set_location( torch.FloatTensor([[0.5,0.2,0], [0,0,0.6]]))
    # r = torch.FloatTensor( [[0.5],[0.1]])
    # s = sphere( f, r)
    # p.plot_sphere(s)

    # plot points
    points = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.5, 0.4, 0.5],
        [0.3, 0.5, 0.4],
        [0.5, 0.6, 0.2],
        [0.5, 0.3, 0.1]
    ])
    p.plot_points(-points)

    p.show()
