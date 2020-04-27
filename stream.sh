#!/bin/bash

trap "kill 0" EXIT

echo "Starting Video Stream"
cd /home/pi/2.S679-pi/mjpg-streamer/mjpg-streamer-experimental
export LD_LIBRARY_PATH=.
mjpg_streamer -o "output_http.so -w ./www" -i "input_file.so -f /home/pi/2.S679-pi/streaming_images -r stream_img.jpg -d 0" > /dev/null 2>&1 &

wait
