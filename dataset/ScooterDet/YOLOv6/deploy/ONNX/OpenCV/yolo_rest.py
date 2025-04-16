import cv2
import numpy as np
import os
import argparse


# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5			# cls score
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45		# obj confidence

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)








if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='models/yolov6n.onnx', help="Input your onnx model.")
	parser.add_argument('--img', default='sample.jpg', help="Path to your input image.")
	parser.add_argument('--classesFile', default='coco.names', help="Path to your classesFile.")
	args = parser.parse_args()

	# Load class names.
	model_path, img_path, classesFile = args.model, args.img, args.classesFile
	window_name = os.path.splitext(os.path.basename(model_path))[0]
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	frame = cv2.imread(img_path)
	input = frame.copy()

	# Give the weight files to the model and load the network using them.
	net = cv2.dnn.readNet(model_path)

	# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the
	# timings for each of the layers(in layersTimes)
	# Process image.
	cycles = 300
	total_time = 0
	for i in range(cycles):
		detections = pre_process(input.copy(), net)
		img = post_process(frame.copy(), detections)
		t, _ = net.getPerfProfile()
		total_time += t
		print(f'Cycle [{i + 1}]:\t{t * 1000.0 / cv2.getTickFrequency():.2f}\tms')

	avg_time = total_time / cycles
	label = 'Average Inference time: %.2f ms' % (avg_time * 1000.0 / cv2.getTickFrequency())
	print(f'Model: {window_name}\n{label}')
	cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
	cv2.imshow(window_name, img)
	cv2.waitKey(0)