# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import scipy

# curl -X POST -F image=@/home/zaverichintan/Chintan/projects/cv_api/cell_detector/6.bmp 'http://localhost:8000/cell_detection/detect/' ; echo ""


# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}

	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])

		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)

			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

			# load the image and convert
			image = _grab_image(url=url)


		################################################################## logic goes ###############################################################################
		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image

		import cv2
		from skimage.restoration import unwrap_phase
		import numpy as np
		from skimage.filters import threshold_otsu
		from matplotlib import pyplot as plt

		globvar_avg = 0

		image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# reading image
		img = image_bw
		# print "image", img.shape
		ref_path = os.path.abspath(os.path.dirname(__file__)) + "/r2.bmp"

		print ref_path
		ref = cv2.imread(ref_path, 0)

		dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
		dft_shift = np.fft.fftshift(dft)

		# for only FFT output
		f = np.fft.fft2(img)
		fshift = np.fft.fftshift(f)
		magnitude_spectrum = 20 * np.log(np.abs(fshift))

		rows, cols = img.shape
		crow = 172
		ccol = 320

		# create a mask first, center square is 1, remaining all zeros
		mask = np.zeros((rows, cols, 1), np.uint8)

		cv2.circle(mask, (ccol, crow), 30, 1, thickness=-1)

		# apply mask and inverse DFT
		fshift = dft_shift * mask
		f_ishift = np.fft.ifftshift(fshift)
		img_back = cv2.idft(f_ishift)

		fshift_mag, fshift_phase_var = cv2.cartToPolar(fshift[:, :, 0], fshift[:, :, 1])
		img_back_mag, img_phase_var = cv2.cartToPolar(img_back[:, :, 0], img_back[:, :, 1])

		# show images
		plt.subplot(221), plt.imshow(img, cmap='gray')
		plt.title('Input Image'), plt.xticks([]), plt.yticks([])

		plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
		plt.title('FFT image'), plt.xticks([]), plt.yticks([])

		plt.subplot(223), plt.imshow(fshift_mag, cmap='gray')
		plt.title('Output image'), plt.xticks([]), plt.yticks([])

		plt.subplot(224), plt.imshow(img_phase_var, cmap='gray')
		plt.title('Phase image'), plt.xticks([]), plt.yticks([])

		# plt.show()

		# for refernece
		dft = cv2.dft(np.float32(ref), flags=cv2.DFT_COMPLEX_OUTPUT)
		dft_shift = np.fft.fftshift(dft)

		# for only FFT output
		# performing fft on the image
		f = np.fft.fft2(ref)
		# performing shift
		fshift = np.fft.fftshift(f)
		magnitude_spectrum = 20 * np.log(np.abs(fshift))

		crow = 172
		ccol = 320

		# create a mask first, center square is 1, remaining all zeros
		mask = np.zeros((rows, cols, 1), np.uint8)
		cv2.circle(mask, (ccol, crow), 30, 1, thickness=-1)

		# apply mask and inverse DFT
		fshift = dft_shift * mask

		f_ishift = np.fft.ifftshift(fshift)
		ref_back = cv2.idft(f_ishift)

		fshift_mag, fshift_phase_var = cv2.cartToPolar(fshift[:, :, 0], fshift[:, :, 1])
		ref_back_mag, ref_phase_var = cv2.cartToPolar(ref_back[:, :, 0], ref_back[:, :, 1])

		# show images
		plt.subplot(221), plt.imshow(ref, cmap='gray')
		plt.title('Input Image'), plt.xticks([]), plt.yticks([])

		plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
		plt.title('FFT image'), plt.xticks([]), plt.yticks([])

		plt.subplot(223), plt.imshow(fshift_mag, cmap='gray')
		plt.title('Output image'), plt.xticks([]), plt.yticks([])

		plt.subplot(224), plt.imshow(ref_phase_var, cmap='gray')
		plt.title('Phase image'), plt.xticks([]), plt.yticks([])

		# plt.show()

		diff = []
		diff1 = []
		diff2 = []
		temp = 0
		temp1 = 0

		diff = np.subtract(ref_phase_var, img_phase_var)  # for phase1 - phase2
		diff1 = np.subtract(ref_phase_var, img_phase_var)  # for phase1 - phase2
		diff2 = np.subtract(img_phase_var, ref_phase_var)  # for phase2 - phase1

		counter = 0

		for i in range(0, len(diff)):
			counter = counter + 1
			for j in range(0, len(diff[0])):
				if (diff[i][j] < 0):
					diff[i][j] = diff[i][j] + 6.28

		for i in range(0, len(diff1)):
			counter = counter + 1
			for j in range(0, len(diff1[0])):
				if (diff1[i][j] < 0):
					diff1[i][j] = diff1[i][j] + 6.28

		for i in range(0, len(diff2)):
			counter = counter + 1
			for j in range(0, len(diff2[0])):
				if (diff2[i][j] < 0):
					diff2[i][j] = diff2[i][j] + 6.28

		diff = unwrap_phase(diff)  # unwrapping

		diff1 = unwrap_phase(diff1)  # unwrapping
		diff2 = unwrap_phase(diff2)  # unwrapping

		min_diff1, min_diff2 = diff1.min(), diff2.min()

		min_diff1, min_diff2 = np.average(diff1), np.average(diff2)

		print "min1", min_diff1, "min2", min_diff2

		diff1 = np.subtract(diff1, min_diff1)
		diff2 = np.subtract(diff2, min_diff2)

		# threshold

		plt.close("all")

		a = 1.2
		b = 1.2
		# local average

		local_avg = np.average(diff) * a

		local_avg1 = np.average(diff1) * a
		local_avg2 = np.average(diff2) * b

		print "avg 1", local_avg1
		print "avg 2", local_avg2

		for i in range(0, len(diff)):
			for j in range(0, len(diff[0])):
				if (diff[i][j] < local_avg):
					diff[i][j] = 0

		for i in range(0, len(diff1)):
			for j in range(0, len(diff1[0])):
				if (diff1[i][j] < local_avg1):
					diff1[i][j] = local_avg1

		for i in range(0, len(diff2)):
			for j in range(0, len(diff2[0])):
				if (diff2[i][j] < local_avg2):
					diff2[i][j] = local_avg2

		# For boundary marking we can mark the boundaries and check the radius for each circle and check if it lies in ^.2 to 8.2 um
		# magnification 25 x 1 pixel = 3.2 * 3.2 um
		# 60 * 80 um  is our field of view

		diff3 = diff1 + diff2
		min_diff3 = diff3.min()

		# min_diff3 = np.average(diff3)

		print "min3 ", min_diff3
		diff3 = np.subtract(diff3, min_diff3)

		c = 2.1

		local_avg3 = np.average(diff3) * c
		print "avg 3 ", local_avg3

		for i in range(0, len(diff3)):
			for j in range(0, len(diff3[0])):
				if (diff3[i][j] < local_avg3):
					diff3[i][j] = 0

		plt.subplot(211), plt.imshow(diff1)
		plt.subplot(212), plt.imshow(diff2)

		res = np.float32(diff3)
		max_diff = np.max(diff3)
		scaling_factor = np.float32(255 / max_diff)
		res = res * scaling_factor

		# cv2.imwrite('images/adv/phasemaps/diff_with_unwrapp-' + image_num + '.png', diff3)
		cv2.imwrite('diff_with_unwrapp.png', res)

		# plt.pcolormesh(diff)
		# plt.colorbar()
		plt.title('Phase Difference'), plt.xticks([]), plt.yticks([])
		# plt.show()

		plt.subplot(111), plt.imshow(diff3, cmap='gray')
		plt.title('Phase Difference'), plt.xticks([]), plt.yticks([])
		# plt.show()

		# finding shapes



		import cv2
		import matplotlib.pyplot as plt
		import numpy as np
		from skimage.color import label2rgb
		from skimage.filters import threshold_otsu
		from skimage.measure import label
		from skimage.measure import regionprops
		from skimage.morphology import closing, square
		from skimage.segmentation import clear_border
		import scipy

		image = cv2.imread('diff_with_unwrapp.png', 0)
		image_originial = image
		# image = res
		rows, cols = image.shape
		mask = np.zeros((rows, cols), np.uint8)
		# apply threshold
		thresh = threshold_otsu(image)
		print "thresh", thresh
		# thresh = 75
		bw = closing(image > thresh, square(3))

		# remove artifacts connected to image border
		cleared = bw.copy()
		clear_border(cleared)

		# label image regions
		label_image = label(cleared)
		borders = np.logical_xor(bw, cleared)
		label_image[borders] = -1
		# image_label_overlay = label2rgb(label_image, image=image)

		counter = 0

		rectangle = False
		contour_list = []

		# identifying regions
		for region in regionprops(label_image):
			# include big images
			if (region.area > 1500 and region.area < 5550):
				y1, x1, y2, x2 = region.bbox

				# removal of cells on border
				if (x1 < 10):
					print "corner"
					continue
				if (y1 < 10):
					print "corner"
					continue
				if (x2 > image.shape[1] - 5):
					print "corner"
					continue
				if (y2 > image.shape[0] - 5):
					print "corner"
					continue

				# divide the rectangle into two subparts
				sidex, sidey = x2 - x1, y2 - y1

				if (sidey > sidex * 1.5 and sidey < sidex * 2.1):
					rectangle = True
					print "longitudnal Rectangle"
					print("Region", counter + 1, "Sides", sidey, sidex)

					counter = counter + 2
					sidey = sidey / 2
					contour_list.append([[x1, y1], [x2, y1 + sidey]])  # rectangle 1
					contour_list.append([[x1, y1 + sidey], [x2, y2]])  # rectangle 2

				elif (sidex > sidey * 1.5 and sidex < sidey * 2.1):
					rectangle = True
					print "lateral Rectangle"
					print("Region", counter + 1, "Sides", sidey, sidex)
					counter = counter + 2
					sidex = sidex / 2
					contour_list.append([x1, y1, x1 + sidex, y2])  # rectangle 1
					contour_list.append([[x1 + sidex, y1], [x2, y2]])  # rectangle 2

				elif (sidey > sidex * 0.8 and sidey < sidex * 1.1):
					print("Region", counter + 1, "Sides", sidey, sidex)
					counter = counter + 1
					contour_list.append([[x1, y1], [x2, y2]])  # square

		# plt.show()

		for i in range(0, len(contour_list)):
			a = (contour_list[i][0][0], contour_list[i][0][1])
			b = (contour_list[i][1][0], contour_list[i][1][1])
			print a, b

			cv2.rectangle(image_originial, a, b, (255, 255, 255), 2)
		counter = 0


		############ Feature extraction##################
		# wavelength
		lam = 0.635e-6
		# Refractive index
		Dn = 1  # or 0.1
		####################################################### doubt here


		# MAgnification
		Mag = 40
		# Area of pixel
		Apixel = 3.2e-6 * 3.2e-6 / Mag * Mag

		no_of_pixels = 0
		phase = 0
		factor = lam / (2 * np.pi * Dn);

		rows, cols = image.shape
		mask = np.zeros((rows, cols, 1), np.uint8)

		import json
		parameters = []

		for i in range(0, len(contour_list)):
			print i


			max_thickness = 0
			a = (contour_list[i][0][0], contour_list[i][0][1])  # x,y
			b = (contour_list[i][1][0], contour_list[i][1][1])

			b_min, b_max = min(a[0], b[0]), max(a[0], b[0])  # y1, y2
			a_min, a_max = min(a[1], b[1]), max(a[1], b[1])  # x1, x2

			for j in range(a_min, a_max):
				for k in range(b_min, b_max):
					if (image[j][k] > 0):
						# print j,k,"value: ",image[j][k]
						no_of_pixels = no_of_pixels + 1
						phase += image[j][k]
						mask[j][k] = image[j][k]

			volume = phase * factor * Apixel
			area_projected = no_of_pixels * Apixel
			dia = 2 * np.sqrt(area_projected / np.pi)

			skewness = scipy.stats.skew(image[a_min: b_min, a_max: b_max])
			kurtosis = scipy.stats.kurtosis(image[a_min: b_min, a_max: b_max])
			std_thickness = np.std(image[a_min: b_min, a_max: b_max], dtype=np.float64, ddof=2)

			max_thickness = np.max(image[a_min:a_max, b_min:b_max])
			print max_thickness, "  ", i

			mean_thickness = float(np.average(image[a_min:a_max, b_min:b_max]))
			gradient = np.gradient(image[a_min:a_max, b_min:b_max])

			# curved_surface_area =
			if (np.isnan(std_thickness)):
				std_thickness = 0

			a = 0
			if (no_of_pixels > 5000):
				# print "ROI", "True", "Region: " , i, "No of pixels:", no_of_pixels, "Volume", volume, "Area", area_projected, "Diameter", dia
				# print "Skew", skewness , "kurt", kurtosis,
				a = a + max_thickness

				parameters.append(
					{"Cell Number": i, "No of pixels:": no_of_pixels, "Volume": volume, "Area": area_projected,
					 "Diameter": dia, "Standard Dev thickness": std_thickness, "Mean thickness": mean_thickness,
					 "Max thickness": a})

		import time

		img_path_name = time.strftime("%Y%m%d-%H%M%S")
		FA_PATH = os.path.abspath(os.path.dirname(__file__)) + "/rect/"+img_path_name+".bmp"

		# print FA_PATH
		# cv2.imshow("t", image_label_overlay)
		# cv2.waitKey(0)
		image = image * 150

		image_originial = cv2.cvtColor(image_originial, cv2.COLOR_GRAY2RGB)
		cv2.imwrite(FA_PATH, image_originial)
		data.update({"url": "/rect/"+img_path_name+".bmp","success": True,"parameters":parameters})



	# return a JSON response
	return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)

	# otherwise, the image does Not reside on disk
	else:
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()

		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()

		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image