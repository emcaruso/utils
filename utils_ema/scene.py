from utils_ema.plot import plotter

class Scene():

    def __init__(self, n_frames, cams=None, mesh=None):
        self.dynamic = (n_frames>1)
        self.cams = cams
        self.n_frames = n_frames

    def set_point_lights(self, point_lights):
        assert( hasattr(point_lights,"__iter__") )
        self.point_lights = point_lights

    def get_cams_in_frame(self, frame):
        return self.cams[frame]

    def get_cams(self):
        if self.cams is None: return None
        if self.dynamic:
            return self.cams
        else:
            return self.cams[0]
        return self.cams[frame]

    def get_cam(self, cam_idx, **kwargs):
        frame = 0
        if self.dynamic:
            assert( "frame" in kwargs.keys() )
            frame = kwargs["frame"]

        return self.cams[frame][cam_idx]

    def plot_cams(self):
        if self.cams is None: return None
        if self.dynamic:
            pass
        else:
            for cam in self.get_cams():
                plotter.plot_cam(cam)

    def plot(self, point_light_size=5):
        self.plot_cams()

        if hasattr(self,"point_lights"):
            for pl in self.point_lights:
                plotter.plot_points(pl.position, size=point_light_size, color='orange')

        plotter.show()


        

