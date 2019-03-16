import time
import traceback

import cv2 as cv

from picamera.array import PiRGBArray
from picamera import PiCamera

# Min matches to look for homography
#MIN_MATCH_COUNT = 10
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (224, 224)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(224, 224))
time.sleep(0.1)

def main():
  #img1 = cv.imread('banana.jpg')
  #img1 = cv.resize(img1, (0,0), fx=0.5, fy=0.5)
  #orb = cv.ORB_create(
  #  nfeatures=5000, edgeThreshold=20, patchSize=20, scaleFactor=1.3, nlevels=20)
  #kp1, des1 = orb.detectAndCompute(img1, None)

  frame_count = 0


  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
      # grab the raw NumPy array representing the image, then initialize the timestamp
      # and occupied/unoccupied text
      image = frame.array
      #cv.imshow("Frame", image)
      key = cv.waitKey(1) & 0xFF

      # clear the stream in preparation for the next frame
      rawCapture.truncate(0)

      # if the `q` key was pressed, break from the loop
      if key == ord("q"):
          break

      get_match_image(image)

  # When everything done, release the capture
  cap.release()
  cv.destroyAllWindows()

def get_match_image(img):
    #img_path = 'banana.jpg'
    #img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])


if __name__ == "__main__":
  main()

