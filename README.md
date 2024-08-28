# LaneLine Detection

This Python script performs lane detection using OpenCV. It captures video from a camera, detects lane lines in real-time, and displays the processed video with detected lanes highlighted.

## Features

- **Region of Interest Masking**: Filters out regions of the image that are not relevant to lane detection.
- **Lane Line Detection**: Uses the Hough Line Transform to detect lines in the image.
- **Line Averaging**: Averages the slopes and intercepts of detected lines to produce a more accurate representation of lane lines.
- **Real-time Processing**: Processes video frames from a camera feed in real-time.

## Requirements

Make sure you have the following packages installed:

```bash
pip3 install opencv-python numpy
