from utils_ema.plot import plotter

class Scene():

    def __init__(self, cams=None, objects=None, lights=None, device='cpu'):
        self.set_cams(cams)
        self.set_lights(lights)
        self.set_objects(objects)
        self.device = device

    # set attributes
    def __set_attrs( self, attr_name: str, attr ):
        if attr is not None:
            assert( hasattr(attr,"__iter__") )
            assert( hasattr(attr[0],"__iter__") )
        setattr(self, attr_name, attr)

    def set_cams(self, cams):
        self.__set_attrs("cams", cams)

    def set_lights(self, lights):
        self.__set_attrs("lights", lights)

    def set_objects(self, objects):
        self.__set_attrs("objects", objects)

    # get attribute at frame t
    def __get_attr_in_frame(self, attr_name, frame):
        attr = getattr(self, attr_name)
        assert( hasattr(attr,"__iter__") )
        return attr[frame]

    def get_cams_in_frame(self, frame):
        return self.__get_attr_in_frame("cams", frame)

    def get_lights_in_frame(self, frame):
        return self.__get_attr_in_frame("lights", frame)

    def get_objects_in_frame(self, frame):
        return self.__get_attr_in_frame("objects", frame)

    # get attributes
    def __get_attrs(self, attr_name):
        attrs = getattr(self, attr_name)
        if attrs is None: return None
        if len(attrs)>1:
            return attrs
        else:
            return attrs[0]

    def get_cams(self):
        return self.__get_attrs("cams")

    def get_lights(self):
        return self.__get_attrs("lights")

    def get_objects(self):
        return self.__get_attrs("objects")


    # get i-th attribute at frame t
    def __get_attr(self, attr_name, idx, frame):
        attrs = getattr(self, attr_name)
        return attrs[frame][idx]

    def get_cam(self, idx, frame=0):
        return self.__get_attr( "cams", idx, frame)

    def get_light(self, idx, frame=0):
        return self.__get_attr( "lights", idx, frame)

    def get_object(self, idx, frame=0):
        return self.__get_attr( "objects", idx, frame)


    # plot attributes
    def __plot_attrs(self, attr_name, kwargs={}):
        attrs = self.__get_attrs(attr_name)
        if attrs is None: return False

        plot_fns = { "cams": plotter.plot_cam, "lights": plotter.plot_points, "objects": plotter.plot_object}
        plot_fn = plot_fns[attr_name]

        if len(attrs)>1:
            for frame, ats in enumerate(attrs):
                for a in ats:
                    plot_fn(a, frame=frame, **kwargs)
        else:
            for a in self.__get_attrs(attr_name):
                plot_fn(a, **kwargs)
        return True

    def plot_cams(self):
        self.__plot_attrs("cams")

    def plot_lights(self, point_light_size=5):
        kwargs = { "point_light_size":point_light_size }
        self.__plot_attrs("lights", kwargs)

    def plot_objects(self):
        self.__plot_attrs("objects")

    def plot_scene(self, point_light_size=5):
        self.plot_cams()
        self.plot_lights()
        self.plot_objects()

    def show_scene(self):
        self.plot_scene()
        plotter.show()



        

