import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
 
cap = cv2.VideoCapture(0) 
 
mpHands = mp.solutions.hands 
hands = mpHands.Hands()   
mpDraw = mp.solutions.drawing_utils
 
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volbar=400
volper=0
 
volMin,volMax = volume.GetVolumeRange()[:2]
 
while True:
    success,img = cap.read() 
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    
    
    results = hands.process(imgRGB) 
  
    lmList = [] 
    if results.multi_hand_landmarks: 
        
        for handlandmark in results.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark): 
                # Finger joint points
                h,w,_ = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy]) 
            mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)
    
    if lmList != []:
        #getting the value at a point
                        #x      #y
        x1,y1 = lmList[4][1],lmList[4][2]  #thumb
        x2,y2 = lmList[8][1],lmList[8][2]  #index finger
        #creating circle at the tips of thumb and index finger
        cv2.circle(img,(x1,y1),13,(255,0,0),cv2.FILLED) #image #fingers #radius #rgb
        cv2.circle(img,(x2,y2),13,(255,0,0),cv2.FILLED) #image #fingers #radius #rgb
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)  #create a line between tips of index finger and thumb
 
        length = hypot(x2-x1,y2-y1) #distance between tips using hypotenuse
 
        vol = np.interp(length,[30,350],[volMin,volMax]) 
        volbar=np.interp(length,[30,350],[400,150])
        volper=np.interp(length,[30,350],[0,100])
        
        
        print(vol,int(length))
        volume.SetMasterVolumeLevel(vol, None)
        
        #Volume bar for volume level 
        cv2.rectangle(img,(50,150),(85,400),(0,0,255),4) # video, initial position, ending position, rgb, thickness
        cv2.rectangle(img,(50,int(volbar)),(85,400),(0,0,255),cv2.FILLED)
        cv2.putText(img,f"{int(volper)}%",(10,40),cv2.FONT_ITALIC,1,(0, 255, 98),3)
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xff==ord(' '): #Spacebar action to stop camera
        break
        
cap.release()       
cv2.destroyAllWindows()