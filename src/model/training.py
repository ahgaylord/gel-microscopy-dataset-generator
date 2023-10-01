import os
from ultralytics import YOLO
# from ultralytics import SAM
# from ultralytics import RTDETR
# import cv2
# import numpy as np
import time

"""
Created on Mon Jun 12 13:05:20 2023

@author: Amory Gaylord

@software{yolov8_ultralytics,
  author       = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title        = {YOLO by Ultralytics},
  version      = {8.0.0},
  year         = {2023},
  url          = {https://github.com/ultralytics/ultralytics},
  orcid        = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license      = {AGPL-3.0}
}
@misc{lin2015microsoft,
      title={Microsoft COCO: Common Objects in Context}, 
      author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
      year={2015},
      eprint={1405.0312},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

"""
train_argo = False
train_coco = False
train_smith = True

model = YOLO('yolov8n-seg.pt')  # load a pretrained model
# model = SAM('sam_b.pt')
# model = RTDETR("rtdetr-l.pt")
model.predict("vid/gel_screenshot3.png", save=True)
exit()
start = time.time()

if train_coco:
    model.train(data='coco128-seg.yaml', epochs=100, imgsz=520, save_period=5, patience=25, save=True, single_cls=True, overlap_mask=False)
coco = time.time()

if train_argo:
    model.train(data='Argoverse.yaml', epochs=100, imgsz=520, save_period=5, patience=25, save=True, single_cls=True)
argo = time.time()

path = os.path.abspath(os.path.dirname(__file__))
path = path + "\\training.yaml"
if train_smith:
    model.train(data=path, epochs=100, imgsz=520, save_period=1, patience=25, save=True, single_cls=True)
smith = time.time()

dur_1 = coco - start
dur_2 = argo - coco
dur_3 = smith - argo
total = smith - start
print("Total time: " + str(total))
print("Coco: " + str(dur_1))
print("Argo: " + str(dur_2))
print("Smith: " + str(dur_3))
