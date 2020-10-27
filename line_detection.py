import cv2
import numpy as np
import time

######################################################################################################
####################<<< MODULE that we made >>>#######################################################



import blob_param_siljun # 파라미터값이 설정되어있는 py파일

d1=0
d2=0
t1=0
t2=0

######################################################################################################
###############################<<< Functions >>>######################################################





# def parking_match(keypoints,imgCamGray,orb,bf,desTrain): ### 주차 표지판 인식


# 	print(keypoints)
# 	for i in keypoints:
# 		if i[1]-90<0:
			
# 			roi=imgCamGray[0:i[1]+90,i[0]-90:i[0]+90]
# 			cv2.imshow('sasa',roi)

# 		else:
# 			roi=imgCamGray[i[1]-70:i[1]+70,i[0]-70:i[0]+70]
# 			cv2.imshow('sasa',roi)
	
# 		imgCamGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# 		kpCam = orb.detect(imgCamGray,None)
# 		kpCam, desCam = orb.compute(imgCamGray, kpCam)
# 		matches = bf.match(desCam,desTrain)
# 		dist = [m.distance for m in matches]
# 		dist.sort()		


# 		wow=[]

# 		for d in dist:
# 			if d<38:
# 				wow.append(d)
# 			else:
# 				break

# 		return len(wow)
	

	







def find_color(frame,lower,upper,stage):  ### 신호/주차 표지판에서 컬러 인식

	detector=blob_param_siljun.setting(stage)

		
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) ### process rgb_image to hsv_image 
	mask_red=cv2.inRange(hsv,lower,upper)


	reversmask=255-mask_red ### Detect blobs
	keypoints = detector.detect(reversmask)
	
	if len(keypoints)>0 and stage==1:
		point=[]
		cv2.imshow('zzz',reversmask)
		for i in keypoints:
			point.append(i.pt)
		
		return point

	else:

		
		return keypoints
			
	
	


	


# def find_white(frame,lower,upper): ### detect white blob when jucha_stage

# 	cv2.line(frame,(int(frame.shape[1]*6.1/9),0),(int(frame.shape[1]*6.1/9),frame.shape[0]),(255,0,255),4) ### draw ROI
# 	cv2.line(frame,(int(frame.shape[1]*2.9/9),0),(int(frame.shape[1]*2.9/9),frame.shape[0]),(255,0,255),4)
# 	cv2.line(frame,(int(frame.shape[1]*2.9/9),frame.shape[0]),(int(frame.shape[1]*6.1/9),frame.shape[0]),(255,0,255),4)
# 	cv2.line(frame,(int(frame.shape[1]*2.9/9),0),(int(frame.shape[1]*6.1/9),0),(255,0,255),4)


# 	detector=blob_param_siljun.white_setting()
	

# 	blob_ROI=frame[:,frame.shape[1]*3/9:frame.shape[1]*6/9] ### setting ROI
		

# 	hsv=cv2.cvtColor(blob_ROI,cv2.COLOR_BGR2HSV) ### process rgb_image to hsv_image 
# 	mask_red=cv2.inRange(hsv,lower,upper)


# 	reversmask=255-mask_red ### Detect blobs
# 	keypoints = detector.detect(reversmask)


# 	return keypoints






# def find_line(frame): ### detect line when jucha_stage	

	


	
# 	blob_ROI=frame[:,frame.shape[1]*3.8/9:frame.shape[1]*5.2/9] ### setting ROI

# 	gray=cv2.cvtColor(blob_ROI,cv2.COLOR_BGR2GRAY) ### Image process to make detecting line easy	
# 	ROI=cv2.GaussianBlur(gray,(7,7),0)
# 	thr=cv2.adaptiveThreshold(ROI,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# 	blur=cv2.medianBlur(thr,9)	
# 	edge=cv2.Canny(blur,180,360)


# 	lines=cv2.HoughLines(edge,1,np.pi/180,120) ### detecting lines
# 	i=0 ### line initializing


# 	if lines is not None:
# 		lines=[l[0] for l in lines]
# 		for line in lines:
# 			r,th=line
# 			a=np.cos(th)
# 			b=np.sin(th)
# 			x0=a*r
# 			y0=b*r
# 			x1=int(x0+1000*(-b))
# 			y1=int(y0+1000*a)
# 			x2=int(x0-1000*(-b))
# 			y2=int(y0-1000*a)
			
			
# 			cv2.line(frame,(x1+int(frame.shape[1]*3.8/9),y1),(x2+int(frame.shape[1]*3.8/9),y2),(255,0,255),5)
# 			i+=1 ### line count

	
# 	return i ### return number of line

##########################################################################################################################
##########################################################################################################################









def line_trace(frame,stage,verbose): ### line을 찾은 후 각속도를 return

	global t1; global t2; global d1; global d2  ### 각 변수값은 0으로 설정되어 있는 상태

	cv2.imshow('frame',frame)
	lower_white=np.array([0,0,200])    ### 흰색 / 노란색 검출시 hsv의 low-high값지정 
	upper_white=np.array([180,15,255])
	lower_yellow=np.array([27,75,163])
	upper_yellow=np.array([35,163,225])


	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    	mask_white=cv2.inRange(hsv,lower_white,upper_white) ### mask_white는 흰색 라인만 남기고 나머지 모두 검은색
	white=cv2.bitwise_and(frame,frame,mask=mask_white) ### frame에 mask_white를 씌우면 마스크에서 검은색이였던 부분은 덮여진다

	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)          ### yellow도 같은 방식으로 마스크를 씌우는 작업
    	mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)
	yellow=cv2.bitwise_and(frame,frame,mask=mask_yellow)
	

	cv2.line(frame,(165,205),(475,205),(253,244,8),2) ### 두 점(165,205) & (475,205)을 이어주는 선을 4개 그려서 
	cv2.line(frame,(165,235),(475,235),(253,244,8),2) ### ROI 영역을 생성 (253,244,8) 은 색상, 2는 두께
	cv2.line(frame,(165,205),(165,235),(253,244,8),2)
	cv2.line(frame,(475,205),(475,235),(253,244,8),2)


	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) ### 읽어들인 frame을 grayscale로
	ROI=gray[210:230,170:470]                   ### 영역을 지정 해주고
	ROI=cv2.GaussianBlur(ROI,(21,21),0)         ### 가우시안 블러 필터 적용
	thr=cv2.adaptiveThreshold(ROI,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)   ### adaptive함수를 이용한 이진화
	blur=cv2.medianBlur(thr,9)
	
	edge=cv2.Canny(blur,180,360)   # blur 이미지에서 360값 이상에 포함된 것을 가장자리로 검출
	cv2.imshow('sdsd',edge)
# 	if stage==1:### parking
								
# 		left_edge=edge[:,:edge.shape[1]/2] ### in left side, it finds only '/'type when jucha stage
# 		right_edge=edge[:,edge.shape[1]/2:] ### in right side, it finds only '\'type when jucha stage
# 		L_lines=cv2.HoughLines(left_edge,1,np.pi/180,50)
# 		R_lines=cv2.HoughLines(right_edge,1,np.pi/180,50)


# 		lineL=[] ### value initializing
# 		lineR=[]
# 		L=0	
# 		R=0
# 		i=0
# 		Ldegree=0
# 		Rdegree=0
	


# 		if R_lines is not None:
# 			R_lines=[l[0] for l in R_lines]
# 			for rho,theta in R_lines:
#     				a = np.cos(theta)
#     				b = np.sin(theta)
#     				x0 = a*rho
#     				y0 = b*rho
#     				x1 = int(x0 + 1000*(-b))
#     				y1 = int(y0 + 1000*(a))
#     				x2 = int(x0 - 1000*(-b))
#     				y2 = int(y0 - 1000*(a))
# 				degree=np.arctan2(y2-y1,x2-x1)*180/np.pi
# 				if degree>3 and R==0:
# 					i+=1
# 					Rdegree=degree
	
# 					R+=2
# 					cv2.line(frame,(x1+320,y1+110),(x2+320,y2+110),(0,100,100),3)
# 					break
# 				else:
# 					continue
			
# 		if L_lines is not None:
# 			L_lines=[l[0] for l in L_lines]
# 			for rho,theta in L_lines:
#     				a = np.cos(theta)
#     				b = np.sin(theta)
#     				x0 = a*rho
#     				y0 = b*rho
#     				x1 = int(x0 + 1000*(-b))
#     				y1 = int(y0 + 1000*(a))
#     				x2 = int(x0 - 1000*(-b))
#     				y2 = int(y0 - 1000*(a))
# 				degree=np.arctan2(y2-y1,x2-x1)*180/np.pi
# 				if degree<-3 and L==0:	
# 					i+=1
# 					Ldegree=degree
	
# 					L+=2
# 					cv2.line(frame,(x1+180,y1+110),(x2+180,y2+110),(0,100,100),3)
# 					break
# 				else:
# 					continue

	else: ### in most of stage

		lines = cv2.HoughLinesP(edge,1,np.pi/180,10,5,10)


		lineL=[]
		lineR=[]
		L=0	
		R=0
		i=0
		Ldegree=0
		Rdegree=0
		L_x=0
		R_x=0
	
			
		if lines is not None:
			lines=[l[0] for l in lines]
			for x1,y1,x2,y2 in lines:
				degree=np.arctan2(y2-y1,x2-x1)*180/np.pi			
				if i==2:
					break
				if x1>150 and R==0:
					i+=1
					Rdegree=degree
					R_x=x1
					R+=2
					cv2.line(frame,(x1+170,y1+210),(x2+170,y2+210),(0,0,255),10)
					print('R')
				elif x1<150 and L==0:
					i+=1
					Ldegree=degree
					L_x=x1
					L+=2
					cv2.line(frame,(x1+170,y1+210),(x2+170,y2+210),(0,0,255),10)
					print(x1)
					print('L')
				else:
					continue
		
	if i==2:
		cv2.circle(frame,((R_x+L_x)/2+170,220),5,(255,0,0),3,-1)
				
	t1=t2
	t2=time.time()

	interval=t2-t1
	

	
	if verbose is True: ### discribe the existence of line and angle and number
		print('lineL is')
		print(lineL)
		print(Ldegree)
		print('lineR is')
		print(lineR)
		print(Rdegree)
		print('there is %d lines'%(i))


	#cv2.imshow('undistorted', yellow)
	#cv2.imshow('unsdforted', white)
	#cv2.waitKey(1)&0xFF

	

	if i==2:
		return frame,-(Ldegree+Rdegree)*0.065 ### if there are two lines, then angular_vel depends on difference of angle


	elif i==1:
		if Ldegree==0:
			return frame,-(Rdegree-90)*0.06-(0.002)*(Rdegree-90)/interval

		else:
			return frame,-(Ldegree+90)*0.06-(0.002)*(Ldegree+90)/interval

	else:
		return frame,-0.01
'''
	elif i==1: ### if there are one line, then angular_vel depends on that's inverse number
		if Ldegree==0:
			return frame,(20/(Rdegree+1.9))
		else:
			return frame,(20/(Ldegree-1.9))

	else:
		return frame,0 ### if line not exist, then return 0 angular_vel
'''
