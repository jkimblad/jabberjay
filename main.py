#!/usr/bin/env python3

import cv2
import numpy as np

def read_brush(size):
	img = cv2.imread("./brushes/1.jpg", cv2.IMREAD_GRAYSCALE)

	# img = cv2.imread("./brushes/4-removebg-preview.png")
	dim = (size, size)
	img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)

	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img


def show_painting(img):
	# Normalize from 0-255 to 0-1 which openCV likes =)
	img = img / 255.0
	cv2.imshow("image", img)
	cv2.waitKey()


def paint(img, brush, pos, brightness):
	if img.ndim != brush.ndim:
		raise Exception("Mismatch in img and brush dimensions")

	brush = np.multiply(brush, brightness)

	roi = img[pos[0]:pos[0] + brush.shape[0], pos[1]:pos[1] + brush.shape[1]]
	roi = cv2.add(roi, brush)
	roi = np.clip(roi, 0.0, 255.0)
	img[pos[0]:pos[0] + brush.shape[0], pos[1]:pos[1] + brush.shape[1]] = roi.astype(np.uint8)
	print(img)


	return img

def main():
	# create painting
	width = 500
	height = 500
	img = np.zeros([width, height])

	# load brush
	brush_size = 50
	brush = read_brush(brush_size)
	print(brush.shape)

	# Draw brushstrokes
	for x in range(100):

		# get random pos
		pos = (
			np.random.randint(width - brush_size, size=1)[0],
			np.random.randint(height - brush_size, size=1)[0]
		)

		# print(pos)
		# stroke brush on painting
		img = paint(img, brush, pos, np.random.rand(1)[0])


	# show painting 
	show_painting(img)



if __name__ == '__main__':
	main()

