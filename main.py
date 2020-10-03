import cv2
import numpy as np

def read_brush(size):
	img = cv2.imread("./brushes/1.jpg")
	# img = cv2.imread("./brushes/4-removebg-preview.png")
	dim = (size, size)
	img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def paint(img, brush, pos):
	if img.ndim != brush.ndim:
		raise Exception("Mismatch in img and brush dimensions")

	roi = img[pos[0]:pos[0] + brush.shape[0], pos[1]:pos[1] + brush.shape[1]].astype(np.uint8)
	roi = cv2.add(roi, brush)
	roi = np.clip(roi, 0, 255)
	img[pos[0]:pos[0] + brush.shape[0], pos[1]:pos[1] + brush.shape[1]] = roi

	# x_stroke = pos[0]:pos[0] + brush.width
	# y_stroke = pos[1]:pos[1] + brush.height
	# img[pos[0]:pos[0] + brush.shape[0], pos[1]:pos[1] + brush.shape[1]] = brush
	# i = 0
	# j = 0
	# for x_img in xrange(pos[0], pos[0] + brush.shape[0]):
	# 	i += 1
	# 	for y_img in xrange(pos[1], pos[1] + brush.shape[1]):
	# 		j += 1
	# 	j = 0


	return img

def main():
	# create painting
	width = 500
	height = 500
	img = np.zeros([width, height, 3])

	# load brush
	brush_size = 50
	brush = read_brush(brush_size)
	print(brush.shape)

	for x in range(100):

		# get random pos
		pos = (
			np.random.randint(width - brush_size, size=1)[0],
			np.random.randint(height - brush_size, size=1)[0]
		)

		# print(pos)
		# stroke brush on painting
		img = paint(img, brush, pos)


	# show painting 
	cv2.imshow("image", img)
	cv2.waitKey()


if __name__ == '__main__':
	main()