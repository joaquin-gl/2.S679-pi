#!/bin/bash

trap "kill 0" EXIT

echo "Starting Face Detection Script"
source /usr/local/bin/virtualenvwrapper.sh
export VIRTUALENVWRAPPER_ENV_BIN_DIR=bin
workon cv
cd /home/pi/2.S679-pi/deep-learning-face-detection
python steve_face_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel > /dev/null 2>&1 &

echo "Warming Up Camera Queue"
sleep 4

echo "Starting Video Stream"
cd /home/pi/2.S679-pi/mjpg-streamer/mjpg-streamer-experimental
export LD_LIBRARY_PATH=.
mjpg_streamer -o "output_http.so -w ./www" -i "input_file.so -f /home/pi/2.S679-pi/streaming_images -r stream_img.jpg -d 0" > /dev/null 2>&1 &

echo "Everything Up and Running!"

wait
