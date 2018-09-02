import cv2
import numpy as np

cap = cv2.VideoCapture('http://10.147.61.23:4747/video')

width = cap.get(3)
height = cap.get(4)


kernel = np.ones((2,2), np.uint8)/4
kernel2 = np.ones((2,6),np.uint8)/12
# kernel2 = np.ones((4,4),np.uint8)/16

while True:
	ret, frame = cap.read()

	frame2 = frame.copy()
	cv2.rectangle(frame2,(150,20),(490,460),(0,0,255),2)
	cv2.imshow('rec',frame2)

	if cv2.waitKey(1) & 0xFF == ord('c'):
		img = frame.copy()
		

		mask = np.zeros(img.shape[:2],np.uint8)
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)
		rect = (150,20,340,440)
		cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		img2 = img*mask2[:,:,np.newaxis]

		img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		img3 = cv2.medianBlur(img3,5)
		img3 = cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY,11,2)
		img3 = cv2.erode(img3,kernel,iterations=1)

		# img4 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		# img4 = cv2.GaussianBlur(img4,(5,5),0)
		# ret4,img4 = cv2.threshold(img4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# img4 = cv2.erode(img4,kernel,iterations=1)
		

		img_inv = cv2.bitwise_not(img3)
		img_inv = cv2.erode(img_inv,kernel2,iterations = 2)
		img_inv = cv2.dilate(img_inv,kernel2,iterations = 2)

		img_inv2, contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# cv2.imwrite('pic1_video_feed.jpg',frame2)
		# cv2.imwrite('pic2_captured_image.jpg',img)
		# cv2.imwrite('pic3_foreground.jpg',img2)
		# cv2.imwrite('pic4_threshold_mask.jpg',img3)
		# cv2.imwrite('pic5_transformed.jpg',img_inv)

		count = 0
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			aspect_ratio = float(w)/h
			area = cv2.contourArea(cnt)
			perimeter = cv2.arcLength(cnt,True)
			if (aspect_ratio > 6) & (area > 500) & (perimeter > 250):
				print(x,'-',y,'  ',area,'  ',aspect_ratio,'  ',perimeter)
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
				count += 1

		print(count-1)

		# cv2.imwrite('pic6_result.jpg',img)

		cv2.imshow('adaptive',img3)
		cv2.imshow('fin',img_inv)
		cv2.imshow('marked',img)

	elif cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()