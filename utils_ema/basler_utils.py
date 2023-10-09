from pypylon import pylon 
from pypylon import genicam
import sys
import cv2
import numpy as np
import torch
try:
    from .image import *
except:
    from image import *


#Interact to start
def ask_user_ready():
    input("Press Enter when you're ready: ")

class frame_extractor:
    def __init__(self):
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.load_devices()

    def load_devices(self):
        self.tlf = pylon.TlFactory.GetInstance()
        self.devices = self.tlf.EnumerateDevices([pylon.DeviceInfo(),])
        self.n_devices = len(self.devices)
        print(f"PYLON: {self.n_devices} devices detected")
        if self.n_devices == 0:
            print("No devices detected!")
            exit(1)

    def start_single_cam(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        # self.camera.Open()
        # self.camera.MaxNumBuffer = 10
        # self.camera.StartGrabbingMax(1000000)

    def start_cams(self, NUM_CAMERAS:int = None, signal_period = 200000, exposure_time = 10000):

        # get devices
        if NUM_CAMERAS is None: NUM_CAMERAS = len(self.devices)

        self.num_cameras = NUM_CAMERAS
        self.cam_array = pylon.InstantCameraArray(min(len(self.devices), self.num_cameras))
        for i, cam in enumerate(self.cam_array):
            cam.Attach(self.tlf.CreateDevice(self.devices[i]))
            print("Using device ", cam.GetDeviceInfo().GetModelName())
        self.cam_array.Open()
        for idx, cam in enumerate(self.cam_array):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            print(f"set context {idx} for camera {camera_serial}")
            cam.SetCameraContext(idx)

        # set exposure time
        for camera in self.cam_array:
            camera.ExposureTime.SetValue(exposure_time)

        #set hardware trigger 
        for camera in self.cam_array:
           camera.BslPeriodicSignalPeriod=200000
           camera.BslPeriodicSignalDelay=0
           camera.TriggerSelector="FrameStart"
           camera.TriggerMode="On"
           camera.TriggerSource="PeriodicSignal1"

        self.cam_array.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    
    def grab_multiple_cams(self):
        assert(self.cam_array.IsGrabbing())

        images = [None] * self.num_cameras
        while True:
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = self.cam_array.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Access the image data.
                cam_id = grabResult.GetCameraContext()
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                images[cam_id] = Image(img=img)
                if any( im is None for im in images):
                    continue
                else:
                    break
        return images

    def collect_frames_multiple(self, manual=True, max_frames = 50, show=True, func_show=[], drop_rate=2):
        collection = []
        while True:
            rawframes = self.grab_multiple_cams()

            if show:
                imgs_show = [ r for r in rawframes ]
                for func in func_show:
                    imgs_show = func[0](imgs_show, *func[1])
                for cam_id,img_show in enumerate(imgs_show):
                    resized = cv2.resize(img_show.img.numpy(), (int(img_show.img.shape[1]/drop_rate), int(img_show.img.shape[0]/drop_rate)), interpolation= cv2.INTER_LINEAR)
                    cv2.imshow("Cam_"+str(cam_id), resized)
            key = cv2.waitKey(1)
            space_pressed = key == ord(' ')
            if not manual or space_pressed:
                if space_pressed: print("Space pressed")
                print("Frame collected")
                collection.append(rawframes)
                if len(collection)>=max_frames:
                    print(f"Max n frames collected ({max_frames})")
                    break

            elif key == ord('q'):
                print("Q pressed")
                print("Sequence grabbed")
                cv2.destroyAllWindows()
                break

        return collection


        
    def stop_single_cam(self):
        self.camera.Close()

    def stop_multiple_cams(self):
        self.cam_array.Close()

    def grab_single_cam(self):
        assert(self.camera.IsGrabbing())

        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image gimg=rabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            image = self.converter.Convert(grabResult)
            img = Image(img=image.GetArray())
            grabResult.Release()
            # cv2.imshow( "ao", img )
            # cv2.waitKey(0)
            return img
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)


if __name__=="__main__":
    frame_extr = frame_extractor()

    # # single frame
    # frame_extr.start_single_cam()
    # img = frame_extr.grab_single_cam()
