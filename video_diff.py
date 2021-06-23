# USAGE
# >python video_diff.py 4a.avi 4b.avi

# import the necessary packages
from skimage.measure import compare_ssim
import imutils
import cv2

from sys import argv

print("videos\\"+str(argv[1]))
print("videos\\"+str(argv[2]))

vs1 = cv2.VideoCapture("videos\\"+str(argv[1]))
vs2 = cv2.VideoCapture("videos\\"+str(argv[2]))

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed1, imageA) = vs1.read()
	(grabbed2, imageB) = vs2.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed1 or not grabbed2:
		break

	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))

	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# compute the bounding box of the contour and then draw the
		# bounding box on both input images to represent where the two
		# images differ
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# show the output images
	cv2.imshow("Original", imageA)
	cv2.imshow("Modified", imageB)
	#cv2.imshow("Diff", diff)
	#cv2.imshow("Thresh", thresh)

	if cv2.waitKey(1) & 0xFF == ord('q'):
                break
