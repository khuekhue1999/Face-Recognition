import sys, os
import numpy as np
import cv2
import json
import threading
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace
from scipy.spatial.distance import euclidean, cosine

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
with open("config/dev_config.json") as f:
	config = json.load(f)
__version__=config["version"]
detect_threshold = config["detdetect_threshold"]
recognition_threshold = config["recognition_threshold"]
ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

def cosine_face ( feature_face, feat_facesbank):
    temp_min = 1000
    temp_index = 0
    for i in range(len(feat_facesbank)):
        temp_similarity = cosine(feature_face, feat_facesbank[i])
        if temp_min > temp_similarity:
            temp_min = temp_similarity
            temp_index = i
    temp_score = temp_min
    temp_indexs = temp_index
    return temp_score, temp_indexs

f = open("database_featureface.json", 'r')
datajson = json.load(f)
facesbank_Name = []
facesbank_feat = []
for i in range (len (datajson["data"])):
	facesbank_Name.append(datajson["data"][i]["NameFace"])
	facesbank_feat.append(datajson["data"][i]["feature"])

def Recognition_img():
	
	model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
	extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))

	image= cv2.imread("BoxFace/Test10/Face_0.jpg")
	image_show = image.copy()
	
	feature_face = extract_feature.Feature_face(image, None, None)
	temp_score, temp_indexs = cosine_face (feature_face, facesbank_feat)
	if temp_score <= recognition_threshold:
		Name_face = facesbank_Name[temp_indexs]
	else:
		Name_face = "Stranger"
	cv2.putText(image_show, Name_face , (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
	
	cv2.imshow("Face", image_show)
	cv2.waitKey(0)

def Recognition_Folder():
	
	model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
	extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
	
	Path_Folder = "BoxFace/CASIAV5_222/"
	count_numimg =0
	for fi in os.listdir(Path_Folder):
		if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS:
			continue
		image= cv2.imread(Path_Folder + fi)
		image_show = image.copy()
		feature_face = extract_feature.Feature_face(image, None, None)
		temp_score, temp_indexs = cosine_face (feature_face, facesbank_feat)
		if temp_score <= recognition_threshold:
			Name_face = facesbank_Name[temp_indexs]
		else:
			Name_face = "Stranger"
		cv2.putText(image_show, Name_face , (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
				
		# cv2.imwrite("imgDR_%d.jpg" %count_numimg , image_show)
		count_numimg +=1
		cv2.imshow("CamFace", image_show)
		cv2.waitKey(1)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

if __name__ == '__main__':
	Recognition_img()
	#Recognition_Folder()


		
