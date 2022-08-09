#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 # pip install opencv-python
import matplotlib.pyplot as plt


# In[ ]:


configuration_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


# In[ ]:


model = cv2.dnn_DetectionModel(frozen_model,configuration_file)


# In[ ]:


classLabel =[]
file_name = 'label.txt'
with open (file_name,'rt') as file :
    classLabel = file.read().rstrip('\n').split('\n')


# In[ ]:


print(len(classLabel))


# In[ ]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,27,5,127.5))
model.setInputSwapRB(True)


# # Read Image 

# In[ ]:


img = cv2.imread('photo-1503023345310-bd7c1de61c7d.jpeg')


# In[ ]:


plt.imshow(img)


# In[ ]:


ClassIndex ,confidence,bbox = model.detect(img,confThreshold=0.5)


# In[ ]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd ,confidence,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(250,0,0),2)
    cv2.putText(img,classLabel[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255, 0, 0),thickness = 3)


# In[ ]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[ ]:


video = cv2.VideoCapture('160929_124_London_Buses_1080p.mp4')
if not video.isOpened():
    video = cv2.VideoCapture(0)
if not video.isOpened():
    raise IOError('Video cannt Open')
font_scale = 3
font = cv2.FONT_HERSHEY_COMPLEX
while True :
    ret,frame = video.read()
    ClassIndex ,confidence,bbox = model.detect(frame,confThreshold=0.5)
    print(ClassIndex)
    if len(ClassIndex) != 0:
        for ClassInd ,confidence,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame,boxes,(250,0,0),2)
                cv2.putText(frame,classLabel[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255, 0, 0),thickness = 3)
    cv2.imshow('Object Detection',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindow()


# In[ ]:




