from pypylon import pylon
import numpy as np
import torch
from pypylon import genicam
import sys
import cv2

from utils_ema.image import *
from utils_ema.general import get_monitor
import pprint

m = get_monitor()


class frame_extractor:

    sensor_sizes = {
        "a2A1920-165g5c": {"sensor_width_mm": 6.6, "sensor_height_mm": 4.1},
        "a2A4504-5gcBAS": {"sensor_width_mm": 12.34, "sensor_height_mm": 12.34},
    }

    def __init__(
        self,
        sRGB=True,
        pixel_format="BayerRG8",
        exposure_time=20000,
        signal_period=100000,
        strategy="UpcomingImage",
        crop=False,
    ):

        self.load_devices()

        self.strategy = strategy
        self.sRGB = "sRgb" if sRGB else "Off"
        self.pixel_format = pixel_format
        self.min_exp_time = 20
        self.max_exp_time = 200000
        self.exposure_time = exposure_time
        self.signal_period = signal_period
        self.crop = crop
        self.count_max = 10
        self.converter = None
        self.init()

    def init(self):
        if self.pixel_format == "BayerRG8":
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed

    def load_devices(self):
        self.tlf = pylon.TlFactory.GetInstance()
        self.devices = self.tlf.EnumerateDevices(
            [
                pylon.DeviceInfo(),
            ]
        )
        self.n_devices = len(self.devices)
        print(f"PYLON: {self.n_devices} devices detected")
        if self.n_devices == 0:
            raise Exception("No devices detected!")

        self.num_cameras = len(self.devices)
        self.cam_array = pylon.InstantCameraArray(
            min(len(self.devices), self.num_cameras)
        )
        for i, cam in enumerate(self.cam_array):
            cam.Attach(self.tlf.CreateDevice(self.devices[i]))
            print("Using device ", cam.GetDeviceInfo().GetModelName())

    def print_devices_info(self):
        infos = self.get_devices_info()
        pprint.pp(infos)

    def get_devices_info(self):
        infos = []
        for i, device in enumerate(self.devices):
            vendor_name = None
            if device.IsVendorNameAvailable():
                vendor_name = device.GetVendorName()
            model_name = None
            if device.IsModelNameAvailable():
                model_name = device.GetModelName()
            sensor_size = None
            if model_name in self.sensor_sizes:
                sensor_size = self.sensor_sizes[model_name]
            infos.append(
                {
                    "id": i,
                    "vendor_name": vendor_name,
                    "model_name": model_name,
                    "sensor_size": sensor_size,
                }
            )
        return infos

    def check_sensor_data_availability(self):
        infos = self.get_devices_info()
        err_str = ""
        for i, info in enumerate(infos):
            if info["model_name"] is None:
                err_str += f"model name not available for camera {i}\n"
            elif info["sensor_size"] is None:
                err_str += f"sensor size not available for camera {i}, add sensor sizes for model {info['model_name']}\n"
        if err_str != "":
            raise ValueError(err_str)

    def restart_cams(self):
        self.stop_multiple_cams()
        p = self.last_cams_params
        self.start_cams(p[0], p[1], p[2])

    def start_cams(
        self,
        num_cameras: int = None,
        signal_period: int = None,
        exposure_time: int = None,
    ):

        if self.cam_array.IsGrabbing():
            return False

        self.last_cams_params = (num_cameras, signal_period, exposure_time)

        # get signal_period and exposure_time
        if signal_period is None:
            signal_period = self.signal_period
        if exposure_time is None:
            exposure_time = self.exposure_time

        self.cam_array.Open()
        for idx, cam in enumerate(self.cam_array):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            print(f"set context {idx} for camera {camera_serial}")
            cam.SetCameraContext(idx)

        # set exposure time
        for camera in self.cam_array:
            camera.BslColorSpace.Value = self.sRGB
            camera.PixelFormat.Value = self.pixel_format
            self.change_exposure(camera, exposure_time)
            self.set_crop(camera)

        # set hardware trigger
        # for camera in self.cam_array:
        self.set_trigger(self.cam_array, signal_period)

        # self.cam_array.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # fast
        # self.cam_array.StartGrabbing(pylon.GrabStrategy_LatestImages) # fast
        # self.cam_array.StartGrabbing(pylon.GrabStrategy_OneByOne)  # fast
        # self.cam_array.StartGrabbing(pylon.GrabStrategy_UpcomingImage)  # slow
        self.cam_array.StartGrabbing(getattr(pylon, "GrabStrategy_" + self.strategy))

        return True

    def set_crop(self, camera):
        if self.crop:
            camera.BslMultipleROIRowsEnable.Value = True
            camera.BslMultipleROIColumnsEnable.Value = True
            self.crop_offs_x = camera.BslMultipleROIRowOffset.Value
            self.crop_offs_y = camera.BslMultipleROIColumnOffset.Value
            self.crop_size_x = camera.BslMultipleROIRowSize.Value
            self.crop_size_y = camera.BslMultipleROIColumnSize.Value
        else:
            camera.BslMultipleROIRowsEnable.Value = False
            camera.BslMultipleROIColumnsEnable.Value = False
            camera.Height.Value = camera.SensorHeight.Value
            camera.Width.Value = camera.SensorWidth.Value
        #

    def get_exposure(self, camera=None):
        if camera is None:
            exposure_times = []
            for cam in self.cam_array:
                exposure_times.append(cam.ExposureTime.GetValue())
            return exposure_times
        else:
            return camera.ExposureTime.GetValue()

    def change_exposure(self, camera=None, exposure_time: int = 20000):
        if camera is None:
            for cam in self.cam_array:
                cam.ExposureTime.SetValue(exposure_time)
        else:
            camera.ExposureTime.SetValue(exposure_time)

    def tune_exposure(
        self,
        exposure_time=20000,
        K_et=100,
        key="max",
        val_target=245,
        val_thresh=5,
        show=True,
        start=True,
    ):

        if start:
            self.start_cams(exposure_time=exposure_time)

        # Tune exposure time
        bool_list = [False for i in range(self.num_cameras)]
        ets = [0 for i in range(self.num_cameras)]

        while True:
            images = self.grab_multiple_cams()
            if show:
                Image.show_multiple_images(images, wk=1)
            for i, image in enumerate(images):

                if bool_list[i]:
                    continue

                if key == "max":
                    _, val = image.get_pix_max_intensity()
                elif key == "min":
                    _, val = image.get_pix_min_intensity()
                elif key == "mean":
                    val = image.get_intensity_mean()

                # change exposure time
                cam = self.cam_array[i]
                et = cam.ExposureTime.GetValue()

                r = val - val_target
                et_new = int(et - K_et * r)
                # et_new = max(et_new, self.min_exp_time)
                # et_new = min(et_new, self.max_exp_time)
                # self.cfg.tuning.exposure_time = et_new

                if (
                    abs(r) < val_thresh
                    or et_new <= self.min_exp_time
                    or et_new >= self.max_exp_time
                ):
                    bool_list[i] = True
                    ets[i] = et_new
                else:
                    self.change_exposure(cam, et_new)

            if all(bool_list):
                print("calibrated exposure times: ", ets)
                return ets

    def set_trigger(self, camera=None, signal_period=250000):
        if isinstance(camera, pylon.InstantCamera):
            camera = [camera]
        for cam in camera:
            cam.BslPeriodicSignalPeriod.Value = signal_period
            cam.BslPeriodicSignalDelay.Value = 0
            cam.TriggerSelector.Value = "FrameStart"
            cam.TriggerMode.Value = "On"
            cam.TriggerSource.Value = "PeriodicSignal1"
            # cam.TriggerActivation.Value = "RisingEdge"
            # cam.TriggerDelayAbs.Value = 0

    def grab_multiple_cams(self, timeout: int = 5000, dtype=torch.float32):

        assert self.cam_array.IsGrabbing()

        img_nr = -1
        count = 0
        images = [None] * self.num_cameras

        while True:
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = self.cam_array.RetrieveResult(
                timeout, pylon.TimeoutHandling_ThrowException
            )

            i = grabResult.ImageNumber
            if i != img_nr:
                img_nr = i
                images.clear()
                images = [None] * self.num_cameras

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                cam_id = grabResult.GetCameraContext()
                # Access the image data.
                if self.converter is not None:
                    img_ = self.converter.Convert(grabResult).GetArray()

                else:
                    img_ = grabResult.GetArray()
                grabResult.Release()
                if self.crop:
                    img = np.zeros(
                        (
                            self.cam_array[cam_id].SensorHeight.Value,
                            self.cam_array[cam_id].SensorWidth.Value,
                            img_.shape[2],
                        ),
                        dtype=np.uint8,
                    )
                    img[
                        self.crop_offs_x : self.crop_offs_x + self.crop_size_x,
                        self.crop_offs_y : self.crop_offs_y + self.crop_size_y,
                        :,
                    ] = img_
                else:
                    img = img_
                images[cam_id] = Image(img=torch.from_numpy(img), dtype=dtype)
                if not any(im is None for im in images):
                    break
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)

            if count > self.count_max:
                self.restart_cams()
                print("Failed sync, restart")
                count = 0
                # break

            count += 1

        return images

    def show_cams(self, wk=0, undistort=None, cams=None):
        while True:
            images = self.grab_multiple_cams()
            v = Image.show_multiple_images(images, wk, undistort=undistort, cams=cams)
            if not v:
                break

    def collect_frames_multiple(
        self,
        manual: bool = True,
        max_frames: int = 50,
        show: bool = True,
        func_show=[],
        drop_rate: int = 2,
    ):
        collection = []
        while True:
            rawframes = self.grab_multiple_cams()

            if show:
                imgs_show = [r for r in rawframes]
                for func in func_show:
                    imgs_show = func[0](imgs_show, *func[1])
                for cam_id, img_show in enumerate(imgs_show):
                    resized = cv2.resize(
                        img_show.img.numpy(),
                        (
                            int(img_show.img.shape[1] / drop_rate),
                            int(img_show.img.shape[0] / drop_rate),
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    winname = "Cam_" + str(cam_id).zfill(3)
                    cv2.namedWindow(winname)  # Create a named window
                    cv2.moveWindow(
                        winname,
                        int(((cam_id % 2) == 1) * (m.width / 2)),
                        int((cam_id > 1) * (m.height / 2)),
                    )
                    cv2.imshow(winname, resized)
            key = cv2.waitKey(1)
            space_pressed = key == ord(" ")
            if not manual or space_pressed:
                if space_pressed:
                    print("Space pressed")
                print("Frame collected")
                collection.append(rawframes)
                if len(collection) >= max_frames:
                    print(f"Max n frames collected ({max_frames})")
                    break

            elif key == ord("q"):
                print("Q pressed")
                print("Sequence grabbed")
                cv2.destroyAllWindows()
                break

        return collection

    def stop_multiple_cams(self):
        self.cam_array.Close()

if __name__ == "__main__":
    frame_extr = frame_extractor()
    frame_extr.print_devices_info()

    # # single frame
    # frame_extr.start_single_cam()
    # img = frame_extr.grab_single_cam()
