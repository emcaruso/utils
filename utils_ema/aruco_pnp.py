import cv2
import math
import numpy as np
from cv2.aruco import CharucoBoard
try:
    from .image import Image
    from .aruco import ArucoDetector
    from .geometry_pose import Pose
    from .geometry_euler import eul
except:
    from image import Image
    from aruco import ArucoDetector
    from geometry_pose import Pose
    from geometry_euler import eul

class ArucoPnP():

    def __init__(self, cfg ):
        self.cfg = cfg
        self.set_detector()
        self.set_points()

    def tune_basl_exposure_for_detection(self, et= 20000, steps = 20, mean_range = [50, 200], mean_thresh = 10):
        try:
            from .basler_utils import frame_extractor
        except:
            from basler_utils import frame_extractor
        fe = frame_extractor()
        fe.start_cams(exposure_time=et)
        s = int((mean_range[1] - mean_range[0])/steps)
        mean_vals = range(mean_range[0], mean_range[1], s)
        print(mean_vals)
        for mean_val in mean_vals:
            print("mean val: ",mean_val)
            ets = fe.tune_exposure(exposure_time=et, K_et=250, key="mean", val_target=mean_val, val_thresh=mean_thresh, start=False)
            images = fe.grab_multiple_cams()
            n = 0
            for image in images:
                marker_corners, marker_idxs, _ = self.detector.detect_arucos(image)
                n += marker_corners.shape[0]
            print("n corners: ",n)


    def set_detector(self):

        #get dict
        # print(self.cfg)
        attr = f"DICT_{self.cfg.aruco_cfg.dict_n}X{self.cfg.aruco_cfg.dict_n}_{self.cfg.aruco_cfg.dict_size}"
        d = getattr(cv2.aruco, attr)
        aruco_dict = cv2.aruco.getPredefinedDictionary(d)

        #get range
        range_idxs = self.cfg.aruco_cfg.range_idxs

        self.detector = ArucoDetector(aruco_dict, range_idxs=range_idxs, detector_params=None)

    def set_points(self):
        self.ids = np.array(self.cfg.struct.ids)
        positions = np.array(self.cfg.struct.positions)
        convention = self.cfg.struct.convention
        eulers = np.array(self.cfg.struct.eulers)
        eulers = eulers*(math.pi/180)# da gradi a radianti
        eulers = eul(e=eulers, convention=convention)

        self.corners = np.zeros( (len(self.ids), 4, 3), dtype=np.float32 )

        R = eulers.eul2rot().numpy()

        # corners = np.array( [[[-0.5,-0.5,0],
        #                       [-0.5, 0.5,0],
        #                       [ 0.5, 0.5,0],
        #                       [ 0.5,-0.5,0]]] )

        corners = np.array( [[
                              [-0.5, 0.5,0],
                              [ 0.5, 0.5,0],
                              [ 0.5,-0.5,0],
                              [-0.5,-0.5,0],
                            ]] )


        corners = np.repeat(corners, repeats=positions.shape[0], axis=0)

        # rotate corners
        for i in range(corners.shape[0]):
            c = corners[i,:,:].reshape([-1,3])
            r = R[i,:,:]
            res = (r@c.T).T
            corners[i,:,:] = res

        positions = np.expand_dims(positions, axis=1)
        positions = np.repeat(positions, repeats=4, axis=1)
        self.corners = positions+corners
        # self.corners *= self.cfg.struct.marker_length
        self.corners *= self.cfg.struct.scale

        # sort wrt indices
        sorted_indices = np.argsort(self.ids)
        self.corners = self.corners[sorted_indices,:,:]
        self.ids = self.ids[sorted_indices]
        # print(self.ids)

    def debug_corners( self, image ):

        marker_corners, marker_idxs, _ = self.detector.detect_arucos(image)

        if len(marker_corners)==0:
            print(f"{len(marker_corners)} detected markers, less than thresh ({self.cfg.n_aruco_thresh})")
            return False

        # corners = self.corners[marker_idxs[:,0],...]
        indices_dict = {value: index for index, value in enumerate(self.ids)}
        idxs = np.array([indices_dict[a] for a in marker_idxs[:,0]])
        corners = self.corners[idxs,...]

        corners = corners.reshape([-1,3])
        marker_corners = marker_corners.reshape([-1,2])
        
        # debug
        for i in range(len(corners)):
            print(corners[i,:])
            curr = np.expand_dims(marker_corners[i,:].astype(np.int32), axis=0)
            img_new = image.draw_circles(curr, radius=2, color=(255,0,0), thickness=2)
            img_new.show(wk=0)

        return True



    def estimate_pose( self, image, intr, vecs=False):

        marker_corners, marker_idxs, _ = self.detector.detect_arucos(image)

        if len(marker_corners)==0:
            print(f"{len(marker_corners)} detected markers, less than thresh ({self.cfg.n_aruco_thresh})")
            if vecs:
                return None, None, None
            else:
                return None


        # corners = self.corners[marker_idxs[:,0],...]
        indices_dict = {value: index for index, value in enumerate(self.ids)}
        idxs = np.array([indices_dict[a] for a in marker_idxs[:,0]])
        corners = self.corners[idxs,...]

        corners = corners.reshape([-1,3])
        
        # # debug
        # for i in range(len(corners)):
        #     print(corners[i,:])
        #     curr = np.expand_dims(marker_corners[i,:].astype(np.int32), axis=0)
        #     img_new = image.draw_circles(curr, radius=2, color=(255,0,0), thickness=2)
        #     img_new.show(wk=0)

        camera_matrix = intr.K_pix.numpy()
        dist_coeffs = intr.D.numpy()

        marker_corners = marker_corners.reshape([-1,2])
        marker_corners = np.flip(marker_corners, axis=-1)
        ret, rvec, tvec = cv2.solvePnP(corners, marker_corners, camera_matrix, dist_coeffs)

        if vecs:
            return ret, rvec, tvec
        
        pose = Pose.cvvecs2pose(rvec, tvec)
        return pose


    def draw_pose(self, image, intr):
        img_new = image.numpy().copy()

        ret, rvec, tvec = self.estimate_pose(image, intr, vecs=True)

        if ret is None:
            return Image(img_new)


        cv2.drawFrameAxes(img_new, intr.K_pix.numpy(), intr.D.numpy(), rvec, tvec, length=0.1)

        marker_corners, _, _ = self.detector.detect_arucos(image)
        marker_corners = marker_corners.reshape([-1,2])
        img_new = Image(img_new).draw_circles(marker_corners, radius=3, color=(0,0,255), thickness=3)

        return img_new

    def show_pose(self, image, intr, wk=0, img_name="unk"):
        img = self.draw_pose(image=image, intr=intr)
        key = img.show(img_name=img_name, wk=wk)
        return key

    def show_poses(self, images, cams, wk=0, img_name="image"):
        key = 0
        for i in range(len(images)):
            if i<len(images)-1: w = 1
            else: w = wk
            key = self.show_pose(images[i], cams[i].intr, wk=w, img_name=img_name+"_"+str(i))
        return key


    def draw_arucos(self, image):
        return self.detector.draw_arucos(image)

    def draw_arucos_multi(self, images):
        return self.detector.draw_arucos_multi(images)
                      
