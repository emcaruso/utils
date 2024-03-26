import cv2
import numpy as np
try:
    from .image import Image
    from .aruco import ArucoDetector
    from .geometry_pose import Pose
except:
    from image import Image
    from aruco import ArucoDetector
    from geometry_pose import Pose

class Charuco(ArucoDetector):

    def __init__(self, board_params):
        self.load_charuco_boards(board_params)

    def load_charuco_boards(self, board_params):
        """
        board_params is a dict with: 
        n_board: int, number_x_square: int, number_y_square: int, length_square_real: float,
        length_marker_real: float, aruco_dictionary (e.g. cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000) )

        """

        charuco_data = {}

        # control on number_board
        if "number_board" not in list(board_params.keys()):
            charuco_data["number_board"] = 1
        charuco_data["number_board"] = board_params["number_board"]

        # # aruco dictionary
        # charuco_data["aruco_dictionary"] = board_params["aruco_dictionary"]

        # boards
        charuco_data["boards"] = []
        for i in range(charuco_data["number_board"]):
            charuco_data["boards"].append( { "charuco_board": cv2.aruco.CharucoBoard((board_params["number_x_square"],
                                                                                     board_params["number_y_square"]),
                                                                                     board_params["length_square_real"],
                                                                                     board_params["length_marker_real"],
                                                                                     board_params["aruco_dictionary"]),
                                            "number_x_square": board_params["number_x_square"],
                                            "number_y_square": board_params["number_y_square"],
                                            "length_square_real": board_params["length_square_real"],
                                            "length_marker_real": board_params["length_marker_real"],
                                            "aruco_dictionary": board_params["aruco_dictionary"],
                                             "n_markers": int(board_params["number_x_square"]*board_params["number_y_square"]/2),
                                             "n_corners": int((board_params["number_x_square"]-2) * (board_params["number_y_square"]-2)),
                                             "aruco_dictionary": board_params["aruco_dictionary"] } )


        self.charuco_data = charuco_data


    def detect_charuco_corners(self, image, board):
        aruco_dict = board["aruco_dictionary"]
        board = board["charuco_board"]

        img = image.numpy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
        charuco_corners = None
        charuco_ids = None
        if marker_corners:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)

        return charuco_corners, charuco_ids, marker_corners, marker_ids

    def detect_charuco_corners_multi(self, image):

        img = image.numpy()

        charuco_corners_all = []
        charuco_corners_ids = []
        charuco_markers_cor = []
        charuco_markers_ids = []

        offs_markers = 0
        offs_corners = 0

        for b in self.charuco_data["boards"]:
            aruco_dict = b["aruco_dictionary"]
            board = b["charuco_board"]

            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
            charuco_markers_cor.append(marker_corners)
            charuco_markers_ids.append(marker_ids)
            if marker_corners:
                ids_curr = marker_ids-offs_markers
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, ids_curr, img, board)
                if retval:
                    charuco_corners_all.append(charuco_corners)
                    charuco_corners_ids.append(charuco_ids+offs_corners)
            offs_markers += b["n_markers"]
            offs_corners += b["n_corners"]

        if len(charuco_corners_all) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        charuco_corners_all = np.vstack(charuco_corners_all)
        charuco_corners_ids = np.vstack(charuco_corners_ids)
        charuco_markers_cor = np.vstack(charuco_markers_cor)
        charuco_markers_ids = np.vstack(charuco_markers_ids)

        return charuco_corners_all, charuco_corners_ids, charuco_markers_cor, charuco_markers_ids

    def draw_charuco(self, image, corners=True, markers=True, borderColor=(0,255,255) ):
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detect_charuco_corners_multi(image)

        if charuco_corners is None:
            return image.clone()

        img = image.numpy().copy()
        if corners:
            img = cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        if markers:
            img = cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids, borderColor)
        return Image(img)

    def get_charuco_poses(self, image, cam_params, vectors=False):

        results = []
        for b in self.charuco_data["boards"]:
            # aruco_dict = b["aruco_dictionary"]
            # charuco_board = b["charuco_board"]
            # results.append( self.get_charuco_pose(image, cam_params, charuco_board, vectors) )
            results.append( self.get_charuco_pose(image, cam_params, b, vectors) )

        return results

    def get_charuco_pose(self, image, cam_params, board, vectors=False):
        charuco_corners, charuco_ids, _, _ = self.detect_charuco_corners(image, board)
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board["charuco_board"], cam_params['camera_matrix'], cam_params['distortion_vector'], np.empty(1), np.empty(1))

        if vectors:
            return rvec, tvec

        return Pose.cvvecs2pose(rvec, tvec)

    def get_charuco_pose_multi(self, images, calib_params, charuco_board, vectors=False):
        poses = []
        for i, image in enumerate(images):
            cam_params = list(calib_params.values())[i]
            pose = self.get_charuco_pose(image, cam_params, charuco_board, vectors=vectors)
            poses.append(pose)
        return poses

    def get_charuco_poses_multi(self, images, calib_params, vectors=False):
        poses = []
        for i, image in enumerate(images):
            cam_params = list(calib_params.values())[i]
            pose = self.get_charuco_poses(image, cam_params, vectors=vectors)
            poses.append(pose)
        return poses



    def draw_charuco_pose(self, image, cam_params, board, center = False):
        img = image.numpy().copy()

        # aruco_dict = b["aruco_dictionary"]
        # board = b["charuco_board"]
        charuco_corners, charuco_ids, _, _ = self.detect_charuco_corners(image, board)
        _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board["charuco_board"], cam_params['camera_matrix'], cam_params['distortion_vector'], np.empty(1), np.empty(1))

        if not center:
            img = cv2.drawFrameAxes( img, cam_params['camera_matrix'], cam_params['distortion_vector'], rvec, tvec, 0.1	)
            return Image(img)
        else:
            rmat, _ = cv2.Rodrigues(rvec)
            cam_T_boardcorn = np.vstack((np.column_stack((rmat, tvec)), [0, 0, 0, 1]))
            # show center
            boardcorn_center = np.float32([ board["length_square_real"]*board["number_x_square"]/2,
                                 board["length_square_real"]*board["number_y_square"]/2,0])
            boardcorn_T_boardcen = np.eye(4)
            boardcorn_T_boardcen[:3,-1] = boardcorn_center
            boardcorn_T_boardcen[1:3,:3] *= -1 
            cam0_T_boardcen = cam_T_boardcorn.dot(boardcorn_T_boardcen)
            R_t = cam0_T_boardcen[:3,:3].transpose()
            t = cam0_T_boardcen[:3,-1]

            # # check board pose
            # board_pose_img = cv2.drawFrameAxes( image_list[0], cam_params['camera_matrix'], cam_params['distortion_vector'], rvec, tvec, 0.1	)
            img = cv2.drawFrameAxes( img, cam_params['camera_matrix'], cam_params['distortion_vector'], cam0_T_boardcen[:3,:3], cam0_T_boardcen[:3,3], 0.1	)
            return Image(img)

    def draw_charuco_poses(self, images, calib_params, board, center=False):
        new_imgs = []
        for i, image in enumerate(images):
            cam_params = list(calib_params.values())[i]
            new_imgs.append(self.draw_charuco_pose(image, cam_params, board, center=center))
        return new_imgs

    def show_charuco_pose(self, image, cam_params, board, center=False, wk=0, img_name="unk"):
        new_img = self.draw_charuco_pose(image, cam_params, board, center=center)
        new_img.show(wk=wk, img_name=img_name)

    def show_charuco_poses(self, images, calib_params, board, center=False, wk=0, img_name="unk"):
        new_imgs = self.draw_charuco_poses(images, calib_params, board, center=center)
        Image.show_multiple_images(new_imgs, wk=0, name=img_name)



if __name__=="__main__":
    ah = ArucoHandler()

    cam_params = {'camera_matrix': np.array([[2.34906201e+03, 0.00000000e+00, 9.15301059e+02],
                                                 [0.00000000e+00, 2.36165568e+03, 5.33705308e+02],
                                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                  'distortion_vector': np.array([[-3.27091949e-01, -7.04355296e-02,  3.51485911e-04,
                                                        9.00112209e-04,  1.53837936e+00]]),
                    'camera_pose_matrix': np.array([[1., 0., 0., 0.],
                                                        [0., 1., 0., 0.],
                                                        [0., 0., 1., 0.],
                                                        [0., 0., 0., 1.]]),
                  'img_width': 1920.0, 'img_height': 1200.0}

    board_params = { "number_x_square": 7,
                      "number_y_square": 5,
                      "length_square_real": 0.055,
                      "length_marker_real": 0.055*(0.03/0.04),
                      "number_board": 3,
                      "aruco_dictionary": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)}

    ah.load_charuco_boards(board_params)
    ah.load_arucos([cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)])
    img = Image(path="./0000.png")
    # img_d = ah.draw_charuco(img)
    # print(ah.charuco_data["boards"][0])
    img_d = ah.draw_charuco_pose(img, ah.charuco_data["boards"][0], cam_params, center = True)
    # img_d = ah.draw_arucos(img, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
    img_d.show()
    
