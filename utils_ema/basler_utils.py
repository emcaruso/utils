from pypylon import pylon 
from pypylon import genicam
import sys
import cv2

#Interact to start
def ask_user_ready():
    input("Press Enter when you're ready: ")

class frame_extractor:
    def __init__(self):
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def start_single_cam(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        # self.camera.Open()
        # self.camera.MaxNumBuffer = 10
        # self.camera.StartGrabbingMax(1000000)

    def start_cams(self, NUM_CAMERAS, signal_period = 200000):

        # get devices
        self.num_cameras = NUM_CAMERAS
        tlf = pylon.TlFactory.GetInstance()
        di = pylon.DeviceInfo()
        devices = tlf.EnumerateDevices([di,])
        self.cam_array = pylon.InstantCameraArray(min(len(devices), NUM_CAMERAS))
        for i, cam in enumerate(self.cam_array):
            cam.Attach(tlf.CreateDevice(devices[i]))
            print("Using device ", cam.GetDeviceInfo().GetModelName())
        self.cam_array.Open()
        for idx, cam in enumerate(self.cam_array):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            print(f"set context {idx} for camera {camera_serial}")
            cam.SetCameraContext(idx)

        #set hardware trigger 
        for camera in self.cam_array:
           camera.BslPeriodicSignalPeriod=200000
           camera.BslPeriodicSignalDelay=0
           camera.TriggerSelector="FrameStart"
           camera.TriggerMode="On"
           camera.TriggerSource="PeriodicSignal1"

    def collect_synch_frames(self, frames_to_grab=10):

        img_list = [ [], [] ]
        time_list = [ [], [] ]
        frame_counts = [0]*self.num_cameras

        self.cam_array.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while True:
            with self.cam_array.RetrieveResult(5000) as res:
                if res.GrabSucceeded():
                    img_nr = res.ImageNumber
                    cam_id = res.GetCameraContext()
                    frame_counts[cam_id] = img_nr
                    image = self.converter.Convert(res)
                    img = image.GetArray()
                    time_stamp = res.TimeStamp
                    res.Release()
                    img_list[cam_id].append(img)
                    if len(time_list[cam_id])==0:
                        time_list[cam_id].append(time_stamp)
                        time_list[cam_id].append(0)
                    else:
                        time_list[cam_id].append(time_stamp-time_list[cam_id][0])
                    print(f"cam #{cam_id}  image #{img_nr}, time: {time_list[cam_id][-1]}")
                    # check if all cameras have reached 100 images
                    if min(frame_counts) >= frames_to_grab:
                        print(f"all cameras have acquired {frames_to_grab} frames")
                        break
        return img_list, time_list

        
    def stop_single_cam(self):
        self.camera.Close()


    def grab_single_cam(self):
        assert(self.camera.IsGrabbing())

        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            image = self.converter.Convert(grabResult)
            img = image.GetArray()
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

    # multi_frames
    frame_extr.start_cams(2)
    img_list , time_list = frame_extr.collect_synch_frames(10)
    for times, imgs in zip(time_list, img_list):

        # for i,time in enumerate(times):
        #     print(f"{i}, {time}")

        for img in imgs:
            cv2.imshow("",img)
            cv2.waitKey(0)







