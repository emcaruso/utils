import plotly.graph_objects as go
import wandb

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

    frames = []
    data_static = []
    max_corner:float = 0

    @classmethod
    def reset(cls):
        max_corner=0
        cls.frames.clear()
        cls.data_static.clear()
        # cls.points.clear()
        # cls.surfaces.clear()
        # cls.lines.clear()
        # cls.arrows.clear()
        # cls.meshes.clear()

    @classmethod
    def init_figure(cls, max_corner=[-1,1], sliders=False, title='3D plot'):

        frames=[]

        # Create a list of frames for animation
        for i in range(len(cls.frames)):
            frame = go.Frame(data=cls.frames[i], name=str(i))
            frames.append(frame)

        # Create the animation slider steps
        slider_steps = []
        for i in range(len(frames)):
            step = {
                'args': [
                    [str(i)],  # Frame name to display
                    {'frame': {'duration': 1000, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}
                ],
                'label': str(i),
                'method': 'animate'
            }
            slider_steps.append(step)

        # Create the figure with frames and slider
        f = []
        if len(cls.frames)!=0:
            f = cls.frames[0]
        fig = go.Figure(data=f+cls.data_static, frames=frames)
        fig.update_layout(
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16},
                    'prefix': 'Frame:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'steps': slider_steps
            }],
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                xaxis_range=max_corner,
                yaxis_range=max_corner,
                zaxis_range=max_corner,
                aspectmode="cube"
            ),
            title='3D Scatter Plot Animation'
        )
        return fig


    @classmethod
    def plot_points(cls, points, size=1, opacity=1, color='red', frame=None):
        if torch.is_tensor(points): points=points.detach().cpu().numpy()
        cls.max_corner=max(np.max(np.abs(points)), cls.max_corner)
        points = go.Scatter3d(
            x = points[..., 0],
            y = points[..., 1],
            z = points[..., 2],
            opacity=opacity,  # Set the transparency level (0 to 1, where 0 is fully transparent and 1 is fully opaque)
            mode = 'markers',
            marker=dict( color=color, size=size )
        )   
        cls.append_data(points, frame)

    @classmethod
    def append_data(cls, data, frame=None):
        if frame is not None:
            d = frame-len(cls.frames)+1
            if d>0: cls.frames.extend( [[]*d] )
            cls.frames[frame].append(data)
        else:
            cls.data_static.append(data)

    @classmethod
    def save(cls, path):
        fig = cls.init_figure(max_corner=[-cls.max_corner,cls.max_corner], sliders=True)
        fig.write_html(path)

    @classmethod
    def wandb_log(cls, name="plotly" ):
        fig = cls.init_figure(max_corner=[-cls.max_corner,cls.max_corner], sliders=True)
        path_to_plotly_html = "./plotly_figure.html"
        table = wandb.Table(columns=["plotly_figure"])
        fig.write_html(path_to_plotly_html, auto_play=False)
        table.add_data(wandb.Html(path_to_plotly_html))
        wandb.log({name: table})


    @classmethod
    def show(cls):
        assert(cls.frames is not None)
        fig = cls.init_figure(max_corner=[-cls.max_corner,cls.max_corner], sliders=True)
        fig.show()
        # get data
        # frames = []
        # for i,data in enumerate(cls.frames):
        #     frames.append(go.Frame(data=data, name=str(i)))
        # fig = go.Figure(data=cls.frames[0], frames=frames, layout=cls.layout)
        # fig.show()

    @classmethod
    def plot_sphere(cls, sphere, opacity=0.7, colorscale='Viridis', frame=None):
        l = sphere.pose.location()
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
            cls.plot_surface(np.concatenate( (x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)), axis=-1 ), frame)

    @classmethod
    def plot_surface(cls, surface, opacity=0.7, colorscale='Viridis', frame=None):
        if torch.is_tensor(surface): surface=surface.numpy()
        cls.max_corner=max(np.max(np.abs(surface)), cls.max_corner)
        x = surface[...,0]
        y = surface[...,1]
        z = surface[...,2]
        surface = go.Surface(
            x = x,
            y = y,
            z = z,
            opacity=opacity,  # Set the transparency level (0 to 1, where 0 is fully transparent and 1 is fully opaque)
            colorscale=colorscale,  # Choose a colorscale
            showscale=False,  # Hide the color scale bar
        )   
        cls.append_data(surface, frame)

    @classmethod
    def plot_line(cls,start, end, opacity=1, color='blue', width=3, frame=None ):
        assert(len(start)==len(end))
        if torch.is_tensor(start): start=start.detach().cpu().numpy()
        if torch.is_tensor(end): end=end.detach().cpu().numpy()
        start = start.reshape( -1, start.shape[-1] )
        end = end.reshape( -1, end.shape[-1] )
        cls.max_corner=max(np.max(np.abs(start)), cls.max_corner)
        cls.max_corner=max(np.max(np.abs(end)), cls.max_corner)
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
            )
            cls.append_data(line,frame)
            
    @classmethod
    def plot_aabb(cls, corners, opacity=.4, color='lightcyan', frame=None ):
        if torch.is_tensor(corners): corners=corners.numpy()
        x,y,z = corners.T
        data = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=opacity,
            color=color,
            # flatshading = True
        )
        cls.append_data(data, frame)



    @classmethod
    def plot_cam(cls,camera, size=0.1, frame=None):
        device = camera.device
        camera.to("cpu")
        o = camera.pose.location()
        c00 = o+camera.pix2dir(torch.LongTensor([0,0]))*size
        c01 = o+camera.pix2dir(torch.LongTensor([0,camera.intr.resolution[1]]))*size
        c10 = o+camera.pix2dir(torch.LongTensor([camera.intr.resolution[0],0]))*size
        c11 = o+camera.pix2dir(torch.LongTensor([camera.intr.resolution[0],camera.intr.resolution[1]]))*size
        cls.plot_line(o,c00,color='magenta', frame=frame)
        cls.plot_line(o,c10,color='magenta', frame=frame)
        cls.plot_line(o,c01,color='magenta', frame=frame)
        cls.plot_line(o,c11,color='magenta', frame=frame)
        cls.plot_line(c00,c01,color='magenta', frame=frame)
        cls.plot_line(c00,c10, width=6, color='darkmagenta', frame=frame)
        cls.plot_line(c11,c01,color='magenta', frame=frame)
        cls.plot_line(c11,c10,color='magenta', frame=frame)
        cls.plot_points(o, color='darkmagenta', frame=frame)
        camera.to(device)

    @classmethod
    def plot_dir(cls,dir, color='cyan', label='Rays', frame=None):
        cls.plot_ray(torch.zeros_like(dir), dir, color=color, label=label, frame=frame)

    @classmethod
    def plot_ray(cls,origin, dir, length=1, color='cyan', label='Rays', frame=None):
        if len(origin.shape)==len(dir.shape)-1:
            origin = origin.unsqueeze(0).repeat(dir.shape[0],1)
        cls.plot_line(origin, origin+(dir*length), color=color, frame=frame)
        cls.plot_points(origin, color='red', frame=frame)

    @classmethod
    def plot_pose(cls,pose, size=1, frame=None):
        x_axis_end = torch.FloatTensor((size, 0, 0))
        y_axis_end = torch.FloatTensor((0, size, 0))
        z_axis_end = torch.FloatTensor((0, 0, size))
        x_axis_end_rot = torch.matmul(pose.rotation().type(torch.float64), x_axis_end.type(torch.float64))
        y_axis_end_rot = torch.matmul(pose.rotation().type(torch.float64), y_axis_end.type(torch.float64))
        z_axis_end_rot = torch.matmul(pose.rotation().type(torch.float64), z_axis_end.type(torch.float64))
        a_x = pose.location().reshape([-1,3])+x_axis_end_rot
        a_y = pose.location().reshape([-1,3])+y_axis_end_rot
        a_z = pose.location().reshape([-1,3])+z_axis_end_rot
        cls.plot_line(pose.location().reshape([-1,3]),a_x, color='red', frame=frame)
        cls.plot_line(pose.location().reshape([-1,3]),a_y, color='green', frame=frame)
        cls.plot_line(pose.location().reshape([-1,3]),a_z, color='blue', frame=frame)

    @classmethod
    def plot_frame(cls,pose, size=1, frame=None):
        cls.plot_pose(pose, size=size, frame=frame)


    @classmethod
    def plot_mesh(cls, vertices, indices, opacity=1, color='lightblue', frame=None ):
        vertices = vertices.to("cpu")
        indices = indices.to("cpu")
        if torch.is_tensor(vertices): vertices=vertices.numpy()
        if torch.is_tensor(indices): indices=indices.numpy()
        cls.max_corner=max(np.max(np.abs(vertices)), cls.max_corner)
        mesh = go.Mesh3d(
            x=vertices[..., 0],
            y=vertices[..., 1],
            z=vertices[..., 2],
            i=indices[..., 0],
            j=indices[..., 1],
            k=indices[..., 2],
            opacity=opacity,
            color=color
        )

        cls.append_data(mesh,frame)

    @classmethod
    def plot_object(cls, object, opacity=1, color='lightblue', frame=None):
        v = object.mesh.get_transformed_vertices(object.pose)
        # v = object.mesh.vertices
        plotter.plot_mesh(v,object.mesh.indices , frame=frame)

    # @classmethod
    # def plot_sgs(cls, sgs):
    #     plot_sphere(sphere(Pose()))




if __name__ == "__main__":

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
