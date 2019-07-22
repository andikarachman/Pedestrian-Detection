# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


def argsParser():

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
		help="path to input video")
    ap.add_argument("-o", "--output", required=True,
		help="path to output video")
    args = vars(ap.parse_args())

    return args


def run_pedestrian_detection(args):

    # define our input 
	INPUT = cv2.VideoCapture(args["input"])

    # define our output path
	OUTPUT_PATH = args["output"]

    # initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	
	print("[INFO] running pedestrian detection...")
	
	# create writer variable to store the output frame to disk
	writer = None
	
	while True:
		# read the next frame from the file
		(ret, frame) = INPUT.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not ret:
			break

        # detect people in the image
		(rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Applies non-max supression from imutils package 
        # to kick-off overlapped boxes
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
		for (xA, yA, xB, yB) in result:
		    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        # check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

		# write the output frame to disk
		writer.write(frame)
    
    # release the file pointers
	print("[INFO] cleaning up...")
	writer.release()
	INPUT.release()


def main():
    args = argsParser()
    run_pedestrian_detection(args)

# Run pedestrian detection
if __name__ == '__main__':
    main()

                                
