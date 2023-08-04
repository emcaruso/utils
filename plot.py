import matplotlib.pyplot as plt
from geometry_pose import *
from geometry_euler import *
from geometry_sphere import *

class plotter():

    @staticmethod
    def init_figure( ndim = 3, limit=1, title = "plot" ):
        plotter.ndim = ndim
        if ndim==3:
            plotter.fig = plt.figure()
            plotter.ax = plotter.fig.add_subplot(111, projection='3d')
            plotter.ax.set_xlim([-limit, limit])
            plotter.ax.set_ylim([-limit, limit])
            plotter.ax.set_zlim([-limit, limit])
            plotter.ax.set_xlabel('X')
            plotter.ax.set_ylabel('Y')
            plotter.ax.set_zlabel('Z')
            plotter.ax.set_title(title)
            plotter.ax.set_aspect('equal')

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def plot_sphere(sphere, transparency=0.6):
        if plotter.ndim != 3:
            plotter.init_figure(ndim=3)

        l = sphere.frame.location()
        l = l.view( -1, l.shape[-1] )
        for i in range(l.shape[0]):
            theta, phi = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
            x = sphere.radius[i] * np.sin(phi) * np.cos(theta)
            y = sphere.radius[i] * np.sin(phi) * np.sin(theta)
            z = sphere.radius[i] * np.cos(phi)
            x = x + float(l[i,0])
            y = y + float(l[i,1])
            z = z + float(l[i,2])
            plotter.ax.plot_surface(x, y, z, color='b', alpha=transparency)

if __name__ == "__main__":
    p = plotter()
    p.init_figure()

    # plot 2 spheres
    f = frame( torch.stack((torch.eye(4, dtype=torch.float32),torch.eye(4, dtype=torch.float32))) )
    f.set_location( torch.FloatTensor([[0.5,0.2,0], [0,0,0.6]]))
    r = torch.FloatTensor( [[0.5],[0.1]])
    s = sphere( f, r)
    p.plot_sphere(s)
    p.show()

