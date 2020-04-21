# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load up a picture of steve! keep unchanged to keep the alpha channel
steve_face = cv2.imread("steve_face_clear.png", cv2.IMREAD_UNCHANGED)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
	    (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # lets put steve's face on top of this one
        # first, pull out the region of interest
        ROI = frame[startY:endY, startX:endX, :]

        # skip if any of the dimensions of the box are <=1
        if min(ROI.shape[:3]) <= 1:
            continue

        # Resize the image of steve to fit the ROI
        steve = cv2.resize(steve_face, (ROI.shape[1],ROI.shape[0]),
                interpolation = cv2.INTER_LINEAR)
        # merge the two, replacing steve's transparent background
        # with the values in ROI
        mask = steve[:,:,3]/255.0 # 3 is the transparency channel
        merge = 0*steve[:,:,:3]
        for i in range(3):
            merge[:,:,i] = steve[:,:,i]*mask + ROI[:,:,i]*(1-mask)
        # replace the frame roi with the merged result
        frame[startY:endY, startX:endX, :] = merge

    # show the output frame
    #cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # save the output to the streaming folder
    cv2.imwrite("/home/pi/2.S679-pi/streaming_images/streaming_img.jpg", frame)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
