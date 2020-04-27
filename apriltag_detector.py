# import the necessary packages
# from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from dt_apriltags import Detector

def outlinedText(frame, text, coordinates):
	cv2.putText(frame, text, coordinates,
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 8)
	cv2.putText(frame, text, coordinates,
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	return frame

def plotAxes(frame, t, R, scale):

	K = np.array([[fx, 0, cx],
				[0, fy, cy],
				[0, 0, 1]])

	rotV, _ = cv2.Rodrigues(R)

	points = np.float32([[-scale, 0, 0], [0, -scale, 0], [0, 0, -scale], [0, 0, 0]]).reshape(-1, 3)

	axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))

	cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 3)
	cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
	cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 3)
	return frame

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the apriltags detector for use
# at_detector = Detector(searchpath=['apriltags'],
#                        families='tag36h11',
#                        nthreads=1,
#                        quad_decimate=1.0,
#                        quad_sigma=0.0,
#                        refine_edges=1,
#                        decode_sharpening=0.25,
#                        debug=0)

# at_detector = Detector(families='tag16h5') # initialize detector with our tag family
at_detector = Detector(families='tagStandard41h12') # initialize detector with our tag family

focal_length = 3.04 # mm
sensor_res = (3280, 2464) # pixels
sensor_area = (3.68, 2.76) # mm

fx = focal_length*sensor_res[0]/sensor_area[0]
fy = focal_length*sensor_res[1]/sensor_area[1]
cx = sensor_res[0]/2
cy = sensor_res[1]/2

picam_params = (fx, fy, cx, cy)
tag_size = 6/3.2808 # 6in to meters

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# (assuming you have a variable "frame" with the current camera frame in it)
	frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #converts to grayscale

	tags = at_detector.detect(frameGray, estimate_tag_pose=True, camera_params=picam_params, tag_size=tag_size)
	# detector requires a grayscale frame; do not estimate pose for now

	for tag in tags:
		# print the detected tag id and floating point coordinates of its center
		# print("%s: %6.1f,%6.1f" % (tag.tag_id, tag.center[0], tag.center[1]))

		# draw a point at the center of the apriltag
		center = tuple(tag.center.astype(int).ravel())
		# cv2.circle(frame, center, 5, (0, 255, 0), -1)

		# draw box around apriltag using corners
		for i in range(4):
			pt0 = tuple(tag.corners[i-1].astype(int).ravel())
			pt1 = tuple(tag.corners[i].astype(int).ravel())
			cv2.line(frame, pt0, pt1, (255, 0, 255), 5)

		# plot axes with rotation matrix at center
		plotAxes(frame, tag.pose_t, tag.pose_R, 0.5)

		# print tag id at center or first corner
		text_start = tuple(tag.corners[0].astype(int).ravel()) # first corner
		# text_start = center # center
		outlinedText(frame, str(tag.tag_id), text_start)

		# print rotation information, (kinda ugly)
		# outlinedText(frame, str(tag.pose_R), (0, 20))

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# save the output to the streaming folder
	cv2.imwrite("/home/pi/2.S679-pi/streaming_images/streaming_img.jpg", frame)

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
