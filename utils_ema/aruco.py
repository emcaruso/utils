import cv2
import numpy as np
from cv2.aruco import CharucoBoard
try:
    from .image import Image
except:
    from image import Image

class ArucoGenerator():
    def __init__(self, aruco_dict):
        self.aruco_dict = aruco_dict

    def generate_aruco_image(self, id, marker_size=200):
        image = cv2.aruco.generateImageMarker(self.aruco_dict, id, marker_size)
        return Image(image)

    def generate_aruco_images(self, first_id, num_markers, marker_size=200):
        imgs = []
        for i in range(first_id,first_id+num_markers,1):
            img = self.generate_aruco_image(i, marker_size)
            imgs.append(img)
        return imgs

    def generate_aruco_stripe(self, first_id, num_markers, spacing, marker_size=200):
        imgs = self.generate_aruco_images(first_id, num_markers, marker_size)
        string = str(first_id)+"_"+str(first_id+num_markers)

        # Calculate the total width of the canvas
        canvas_width = num_markers * marker_size + (num_markers - 1) * spacing + 2*spacing
        canvas_height = marker_size + 2*spacing
        canvas = np.full((canvas_height, canvas_width), 255, np.uint8)

        for i, img in enumerate(imgs):
            x_position = i * (marker_size + spacing) + spacing
            canvas[spacing:-spacing, x_position:x_position + marker_size] = img.img

        return canvas, string

    def generate_aruco_stripes(self, first_id, num_markers, spacing, num_stripes, marker_size=200):
        n_mark = int(num_markers / num_stripes)
        assert( (num_markers % num_stripes) == 0)

        stripes = []
        strings = []
        for i in range(num_stripes):
            f_id = first_id + i * n_mark
            stripe, string = self.generate_aruco_stripe(f_id, n_mark, spacing, marker_size)
            stripes.append(stripe)
            strings.append(string)

        return stripes, string

    def generate_aruco_grid(self, first_id, spacing, X, Y, border_pixs, marker_size=200):
        num_stripes = Y
        num_markers = X*Y
        stripes, _ = self.generate_aruco_stripes(first_id, num_markers, spacing, num_stripes, marker_size=marker_size)
        canvas = np.full((Y*(marker_size+2*spacing+border_pixs)+border_pixs, (X*(marker_size+spacing)+spacing)+2*border_pixs), 0, np.uint8)
        string = str(first_id)+"_"+str(first_id+(X*Y))
        for y in range(Y):
            y_ = y*(marker_size+2*spacing+border_pixs)+border_pixs
            stripe = stripes[y]
            canvas[y_:y_+(marker_size+2*spacing),border_pixs:-border_pixs] = stripe
        return canvas, string

    def generate_aruco_grids(self, first_id, spacing, X, Y, border_pixs, num_grids, marker_size=200):
        grids = []
        strings = []
        for i in range(num_grids):
            f_id = first_id+i*X*Y
            grid, string = self.generate_aruco_grid(f_id, spacing, X, Y, border_pixs, marker_size)
            grids.append(grid)
            strings.append(string)
        return grids, strings


class ArucoDetector():

    # def __init__(self, board_params):
    def __init__(self, aruco_dict, range_idxs = None, detector_params = None, color=(255,0,255)):
        self.aruco_dict = aruco_dict
        self.range_idxs = range_idxs
        self.set_detector(detector_params)
        self.color = color

    @staticmethod
    def get_aruco_dict(n=6, size=1000):
        attr = f"DICT_{n}X{n}_{size}"
        d = getattr(cv2.aruco, attr)
        res = cv2.aruco.getPredefinedDictionary(d)
        return res, attr

    def set_detector(self, detector_params=None):
        if detector_params is None:
            detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary=self.aruco_dict, detectorParams=detector_params)

    def detect_arucos(self, image):
        img = image.numpy()
        marker_corners, marker_ids, marker_rejected = self.detector.detectMarkers(image=img)

        if marker_ids is not None:
            marker_corners = np.array(marker_corners)
            marker_ids = np.array(marker_ids)

            # range up
            indices = np.nonzero(marker_ids[:,0] <= self.range_idxs[1])[0]
            marker_corners = marker_corners[indices]
            marker_ids = marker_ids[indices]

            # range down
            indices = np.nonzero(marker_ids[:,0] >= self.range_idxs[0])[0]
            marker_corners = marker_corners[indices]
            marker_ids = marker_ids[indices]

        marker_corners = np.flip(marker_corners, axis=-1)

        return marker_corners, marker_ids, marker_rejected

    def show_arucos(self, image, wk=0):
        img = self.draw_arucos(image)
        img.show(wk=wk, img_name="arucos")

    def draw_arucos(self, image, rejected=False):
        marker_corners, marker_ids, marker_rejected = self.detect_arucos(image)
        marker_corners = np.flip(marker_corners, axis=-1)
        img = image.numpy().copy()
        img = cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        if rejected:
            img = cv2.aruco.drawDetectedMarkers(img, marker_rejected, borderColor=self.color)
        return Image(img)

    def draw_arucos_multi(self, images, rejected=False):
        assert( hasattr(images, "__iter__"))
        imgs_new = []
        for image in images:
            img = self.draw_arucos(image, rejected=rejected)
            imgs_new.append(img) 
        return imgs_new

if __name__=="__main__":
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    ah = ArucoGenerator(aruco_dictionary)
    marker_size = 200
    spacing = int(marker_size*0.1)
    num_markers = 180
    num_stripes = 30
    border_pixs = max(int(marker_size*0.02),1)
    X = 6
    Y = 4
    num_grids = 4

    # stripe = ah.generate_aruco_stripe(100, num_markers, spacing, marker_size)
    # cv2.imwrite("./arucos/aruco_sequence.png", stripe)

    # stripes, string = ah.generate_aruco_stripes(100, num_markers, spacing, marker_size=marker_size, num_stripes = num_stripes)
    # for i, stripe in enumerate(stripes):
    #     cv2.imwrite(f"./arucos/{string}.png", grid)

    # grid, string = ah.generate_aruco_grid(100, spacing, X=X, Y=Y, border_pixs=border_pixs, marker_size=marker_size)
    # cv2.imwrite(f"./arucos/{string}.png", grid)

    grids, strings = ah.generate_aruco_grids(100, spacing, X=X, Y=Y, num_grids = num_grids, border_pixs=border_pixs, marker_size=marker_size)
    for i, grid in enumerate(grids):
        cv2.imwrite(f"./arucos/{strings[i]}.png", grid)
