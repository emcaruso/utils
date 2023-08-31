import plotly.graph_objects as go

try:
    from .geometry_pose import *
    from .geometry_euler import *
    from .geometry_sphere import *
    from .torch_utils import *
except:
    from geometry_pose import *
    from geometry_euler import *
    from geometry_sphere import *
    from torch_utils import *


class plotter():

    points = []
    surfaces = []
    lines = []
    colorscale='Viridis',  # Choose a colorscale
    arrows = []
    meshes = []
    layout: go.Layout = None 

    @classmethod
    def init_figure(cls, range=[-1,1], title='3D plot'):
        # Create a layout
        cls.layout = go.Layout(
            title=title,
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                xaxis_range=range,
                yaxis_range=range,
                zaxis_range=range,
                aspectmode='cube'  # Set aspect ratio to cube

            )
        )

    @classmethod
    def plot_points(cls, points, opacity=1, color='red', name=''):
        if torch.is_tensor(points): points=points.numpy()
        points = go.Scatter3d(
            x = points[..., 0],
            y = points[..., 1],
            z = points[..., 2],
            opacity=opacity,  # Set the transparency level (0 to 1, where 0 is fully transparent and 1 is fully opaque)
            mode = 'markers',
            name = name,
            marker=dict( color=color )
        )   
        cls.points.append(points)

    @classmethod
    def show(cls):
        # get data
        data = cls.points+cls.surfaces+cls.lines+cls.arrows+cls.meshes
        fig = go.Figure(data, layout=cls.layout)
        fig.show()


    @classmethod
    def plot_sphere(cls, sphere, opacity=0.7, colorscale='Viridis', name='' ):
        l = sphere.frame.location()
        l = l.view( -1, l.shape[-1] )
        for i in range(l.shape[0]):
            theta, phi = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
            sphere.radius = repeat_tensor_to_match_shape(sphere.radius, phi.shape)
            x = sphere.radius[i] * np.sin(phi) * np.cos(theta)
            y = sphere.radius[i] * np.sin(phi) * np.sin(theta)
            z = sphere.radius[i] * np.cos(phi)
            x += float(l[i,0])
            y += float(l[i,1])
            z += float(l[i,2])
            surface = go.Surface(
                x = x,
                y = y,
                z = z,
                opacity=opacity,  # Set the transparency level (0 to 1, where 0 is fully transparent and 1 is fully opaque)
                colorscale=colorscale,  # Choose a colorscale
                showscale=False,  # Hide the color scale bar
                name = name,
            )   
            cls.surfaces.append(surface)

    @classmethod
    def plot_line(cls,start, end, opacity=1, color='blue', width=3, name='' ):
        assert(len(start)==len(end))
        if torch.is_tensor(start): start=start.numpy()
        if torch.is_tensor(end): end=end.numpy()
        start = start.reshape( -1, start.shape[-1] )
        end = end.reshape( -1, end.shape[-1] )
        for i in range(start.shape[0]):
            x = [start[i,0], end[i,0]]
            y = [start[i,1], end[i,1]]
            z = [start[i,2], end[i,2]]
            line = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',  # Use 'lines' mode for line segments
                line=dict(color=color, width=width),  # Set line color and width
                name='Line Segments'
            )
            cls.lines.append(line)
            

    @classmethod
    def plot_cam(cls,camera, size=0.1):
        o = camera.frame.location()
        c00 = o+camera.pix2dir(torch.LongTensor([0,0]))*size
        c01 = o+camera.pix2dir(torch.LongTensor([0,camera.intr.resolution[1]]))*size
        c10 = o+camera.pix2dir(torch.LongTensor([camera.intr.resolution[0],0]))*size
        c11 = o+camera.pix2dir(torch.LongTensor([camera.intr.resolution[0],camera.intr.resolution[1]]))*size
        cls.plot_line(o,c00,color='magenta')
        cls.plot_line(o,c10,color='magenta')
        cls.plot_line(o,c01,color='magenta')
        cls.plot_line(o,c11,color='magenta')
        cls.plot_line(c00,c01,color='magenta')
        cls.plot_line(c00,c10, width=5, color='darkmagenta')
        cls.plot_line(c11,c01,color='magenta')
        cls.plot_line(c11,c10,color='magenta')
        cls.plot_points(o, color='darkmagenta')


    @classmethod
    def plot_ray(cls,origin, dir, color='c', label='Rays'):
        cls.plot_line(origin, origin+dir, color='cyan')

    @classmethod
    def plot_frame(cls,frame):
        x_axis_end = torch.FloatTensor((1, 0, 0))
        y_axis_end = torch.FloatTensor((0, 1, 0))
        z_axis_end = torch.FloatTensor((0, 0, 1))
        x_axis_end_rot = torch.matmul(frame.rotation(), x_axis_end)
        y_axis_end_rot = torch.matmul(frame.rotation(), y_axis_end)
        z_axis_end_rot = torch.matmul(frame.rotation(), z_axis_end)
        a_x = frame.location()+x_axis_end_rot
        a_y = frame.location()+y_axis_end_rot
        a_z = frame.location()+z_axis_end_rot
        cls.plot_line(frame.location(),a_x, color='red')
        cls.plot_line(frame.location(),a_y, color='green')
        cls.plot_line(frame.location(),a_z, color='blue')




if __name__ == "__main__":
    plotter.init_figure()

    # plot a sphere
    f = Frame( torch.eye(4, dtype=torch.float32))
    f.set_location( torch.FloatTensor([0,0,0.5]))
    r = torch.FloatTensor( [0.5])
    s = sphere( f, r)
    plotter.plot_sphere(s)

    # plot lines
    start = torch.tensor([
        [0.7, 0.0, 0.1],
        [0.5, 0.1, 0.1],
        ])
    end = torch.tensor([
        [0.2, 0.2, 0.3],
        [0.0, 0.9, 0.3],
        ])
    plotter.plot_line(start,end)

    # plot points
    points = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.5, 0.4, 0.5],
        [0.3, 0.5, 0.4],
        [0.5, 0.6, 0.2],
        [0.5, 0.3, 0.1]
    ])
    plotter.plot_points(-points)

    plotter.show()
