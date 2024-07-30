# calling all the files
from ultralytics import YOLO
import cv2 as cv
#logical variables
accident=False
emergency=False 
deadtraffic =1
count=0
#  calling the pretrained and trained model
# pretrained
model1=YOLO("best.pt")
# emergency trainde
# model1=YOLO("best.pt")
# accident trained


# giving the souce file from where we have to take the video for processing
source=(1)

# configuring the result of the modules by  selecting the classes and 

#  function for manging pretrained model 

# def pretrainde():
#    result0=model0.predict(source,show=True,stream=True,verbose=False)
#    name=model0.names
#    for r in result0:
#         for c in r.boxes.cls:
#             if (name[int(c)]=='without helmet'):
#                 count=count+1
#                 print("haan bhai")
#    return count
          
# function for managing the emergency vehicles in the yolo

def emergency():
    result1=model1.predict(source,show=True,stream=True,verbose=False)
    name=model1.names
    for r in result1:
        for c in r.boxes.cls:
            if name[int(c)]=='without helmet':
                print("Emergency")
                emergency=True
               
            
# if __name__== "__main__": 
#     emergency()

# function for managing the accident that occures at the sqiares

# def accident():
#     result2=model2.predict(source,show=False,stream=True,verbose=False,conf=0.20,classes=0)
#     name=model2.names
#     for r in result2:
#         for c in r.boxes.cls:
#             if (name[int(c)]=="Accident"):
#                 # print("accident")
#                 accident=True
#                 return accident


# cv.waitKey(0)              