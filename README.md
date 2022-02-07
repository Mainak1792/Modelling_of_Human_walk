# Modelling_of_Human_walk
## Aim
To model the human walking, to trace hip joint , knee Joint, foot Joint.
## Experimental Design
- Record video of a Human walking
-  Extract Frames
- Draw the joints and link 
- trace the Joints and Plot
## Experiment 1
- Through manual Video
### Test Video 1: 
#### Test Subject
- Age: 55
- Height: 5'11
- Weight : 70
- Captured at home using off-the-shelve smartphone
#### Discreetization
![Pasted image 20220202150821](https://user-images.githubusercontent.com/76518189/152793893-5cd37732-3041-4e8d-8f40-0e1070933d00.png)
#### Remove the context
![Pasted image 20220202151249](https://user-images.githubusercontent.com/76518189/152793924-622254e8-a58d-44c0-a338-176f1300010e.png)
#### Trace the Joints 
![Pasted image 20220202153138](https://user-images.githubusercontent.com/76518189/152793937-884554bc-e6c0-436a-be66-7c21228bd10c.png)
#### Results and Observations 
- Joints are being traced and plotted
- Although tracing of Hip Joint is showing some pattern, however no significant change in other joints is being observed.
- Video need to be taken from back to observe the trend
### Test Video 2:
-  Youtube Link : https://www.youtube.com/watch?v=ATM0HD5d43w

- Python Code to Extract Frames
```   
import cv2
cap = cv2.VideoCapture(r'C:\Users\maina\OneDrive\Desktop\BackgroundSubtraction-master\WavingTrees\Normal Gait_ Hip and Pelvic Kinematics.mp4')
i = 0
while (cap.isOpened()):
	ret, frame = cap.read()
	if ret == False:
		break
	cv2.imwrite('p1' + str(i) + '.jpg', frame)
	i += 1
cap.release()
cv2.destroyAllWindows 
```

####  Extract each frame 
![Pasted image 20220202170508](https://user-images.githubusercontent.com/76518189/152794037-9a281155-0a41-48eb-bbc6-57ecc89a28e0.png)
#### Mark the Joints
![Pasted image 20220202170558](https://user-images.githubusercontent.com/76518189/152794038-39a572f2-75f2-4175-bfc9-553ae66c9c68.png)
#### Remove Context
![Pasted image 20220202170636](https://user-images.githubusercontent.com/76518189/152794062-cf55dd8c-4ee1-4e64-b3d0-2e89b5269698.png)
####  Plot
![Pasted image 20220202170715](https://user-images.githubusercontent.com/76518189/152794081-228d7902-c29b-4aea-a4c9-9e0a38835b80.png)
#### Results and Observations 
- Although Some patterns are obsereved in the joints, however due to discreetization most of the pattern is lost
- The video needs to annotated and some kind of motion recognition is to applied
- Similar work has been done using simulink 
## Experiment 2
- Through pose recognition
- Using MediaPipe : https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
- After pose estimation using mediapipe we are tracing the hip joint through the video.
- Steps are as Follows
### Test Video
- Video 1 :
- Video 2 :
### Script
```   
import cv2  
import mediapipe as mp  
import time  
import numpy as np  
import glob  
mpDraw = mp.solutions.drawing_utils  
mpPose = mp.solutions.pose  
pose = mpPose.Pose(static_image_mode = True,  
             model_complexity= 1,  
             smooth_landmarks= True,  
             enable_segmentation= False,  
             smooth_segmentation = True,  
             min_detection_confidence = 0.5,  
             min_tracking_confidence = 0.5)  
  
cap = cv2.VideoCapture(r'C:\Users\maina\OneDrive\Desktop\human_Walk\45.mp4')  
pTime = 0  
curr_frame =0  
  
img_array = []  
while True:  
    success, img = cap.read()  
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    results = pose.process(imgRGB)  
    # print(results.pose_landmarks)  
 if results.pose_landmarks:  
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  
        for id, lm in enumerate(results.pose_landmarks.landmark):  
            h, w, c = img.shape  
            print(id, lm)  
            cx, cy = int(lm.x * w), int(lm.y * h)  
            cv2.circle(img, (cx, cy), 1, (110, 50, 50), cv2.FILLED)  
  
    cv2.imshow("Image", img)  
  
    cv2.waitKey(1)
```
###  Snippets 

### Output Video Link
- Output : 
### Observations
- Although hip joint is traced through the video. However, Consistency is not maintained.
- There are fluctuations in the tracing
-  Some type of pattern is observable. However, it is not reusable.
## Future Work
- Modelling the biped movement in Simulink [9/02/2021]
## Refrence
1. Kajita, S., Kanehiro, F., Kaneko, K., Yokoi, K. and Hirukawa, H., 2001, October. The 3D linear inverted pendulum mode: A simple modeling for a biped walking pattern generation. In _Proceedings 2001 IEEE/RSJ International Conference on Intelligent Robots and Systems. Expanding the Societal Role of Robotics in the the Next Millennium (Cat. No. 01CH37180)_ (Vol. 1, pp. 239-246). IEEE.
2. https://musculoskeletalkey.com/fundamentals-of-human-gait/
