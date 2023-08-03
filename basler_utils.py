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
        self.camera.Open()
        self.camera.MaxNumBuffer = 10
        self.camera.StartGrabbingMax(1000000)

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


#Extract synchronized video
def extract_multiple_sync_videos( storing_dir ):
    ask_user_ready()
    # TODO


if __name__=="__main__":
    frame_extr = frame_extractor()
    frame_extr.start_single_cam()
    frame_extr.grab_single_cam()
