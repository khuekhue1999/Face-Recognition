# import pickle
import json
import cv2
import numpy as np
import time
import sys, os
import json
import os.path
import pickle
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace
from scipy.spatial.distance import euclidean, cosine
with open("config/dev_config.json") as f:
	config = json.load(f)
__version__=config["version"]
detect_threshold = config["detdetect_threshold"]
# def load_SVM(self, selected):
#     if selected:
#         self.model_using = "SVM"
#         path_modelSVM = "arcface_SVM_finalized_model.sav"
#         self.Model_SVM = pickle.load(open(path_modelSVM, 'rb'))
#
#         self.Model_SVM_ClassName = []
#         self.Model_SVM_ClassCode = []
#         f = open("arcface_SVM_finalized_model.json", 'r')
#         dataname = json.load(f)
#         for i in range(len(dataname["ClassName"])):
#             self.Model_SVM_ClassName.append(dataname["ClassName"][i])
#             self.Model_SVM_ClassCode.append(dataname["code"][i])
model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
ctx = mx.cpu()
mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True, threshold=detect_threshold)


image = cv2.imread(r"Test/120184873_3332865433447047_5845175485502121141_o.jpg")
image_detect= image.copy()
image_show = image.copy()
ret_mtcnn = detector.detect_face(image_detect, det_type=0)
rectangles, points_mtcnn = ret_mtcnn
				#count_numbox = 0
for (rectangle, point_mtcnn) in zip(rectangles, points_mtcnn):
    height_bbox = int(rectangle[3]) - int(rectangle[1])
    with_bbox = int(rectangle[2]) - int(rectangle[0])
    if rectangle[4] < 0.8 or height_bbox * with_bbox < 80 * 70:
        continue
    cv2.rectangle(image_show, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                  (255, 0, 0), 2)
    feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)



feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)
f = open("arcface_SVM_finalized_model.json", 'r')
dataname = json.load(f)
print(dataname['ClassName'])
