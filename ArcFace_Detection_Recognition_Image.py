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
ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])

def cosine_face ( feature_face, feat_facesbank, idfeat_facesbank):
	temps_similarity = []
	for i in range(len(feat_facesbank)):
		temp_similarity = cosine(feature_face, feat_facesbank[i])
		temps_similarity.append(temp_similarity)

	similarity_sort = np.sort(temps_similarity) 
	indexs_sort = np.argsort(temps_similarity)
	temp_scores = [similarity_sort[0], similarity_sort[1], similarity_sort[2]]
	temp_indexs = [indexs_sort[0], indexs_sort[1], indexs_sort[2]]
	temp_idfeat = [idfeat_facesbank[temp_indexs[0]], idfeat_facesbank[temp_indexs[1]], idfeat_facesbank[temp_indexs[2]]]
	print (temp_idfeat[1])
	if temp_idfeat[0] == temp_idfeat[1] or temp_idfeat[0] == temp_idfeat[2]:
		temp_score = temp_scores[0]
		temp_index = temp_indexs[0]
	elif temp_idfeat[1] == temp_idfeat[2] and temp_idfeat[0] != temp_idfeat[1]:
		if temp_scores[0] <= 0.2:
			temp_score = temp_scores[0]
			temp_index = temp_indexs[0]
		elif temp_scores[1] <= recognition_threshold:
			temp_score = temp_scores[1]
			temp_index = temp_indexs[1]
		else:
			temp_score = temp_scores[0]
			temp_index = temp_indexs[0]
	else:
		temp_score = temp_scores[0]
		temp_index = temp_indexs[0]

	return temp_score, temp_index

f = open("Arcface_database_featureface.json", 'r')
datajson = json.load(f)
facesbank_Name = []
facesbank_feat = []
facesbank_idface = []
for i in range (len (datajson["data"])):
	for j in range (len(datajson["data"][i]["features"])):
		facesbank_Name.append(datajson["data"][i]["NameFace"])
		facesbank_feat.append(datajson["data"][i]["features"][j])
		facesbank_idface.append(datajson["data"][i]["idface"])


def Detection_recognition_img():
	cv2.namedWindow('CamFace', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CamFace', 960, 640)

	model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
	extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
	ctx = mx.cpu()
	mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
	detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold= detect_threshold)

	image= cv2.imread("Img/Test10.png")
	image_detect =image.copy()
	image_show = image_detect.copy()
	
	ret_mtcnn = detector.detect_face(image_detect, det_type = 0)
	if ret_mtcnn is None:
		rectangles , points_mtcnn = [],[]
		text = "Face not detected in an image"
		cv2.putText(image_show, text , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
	else:
		rectangles , points_mtcnn = ret_mtcnn
		count_numbox =0
		for ( rectangle, point_mtcnn) in zip (rectangles, points_mtcnn):
			if rectangle[4] < 0.79:
				continue
			cv2.rectangle(image_show,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
			feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)
			temp_score, temp_indexs = cosine_face (feature_face, facesbank_feat)
			if temp_score <= recognition_threshold:
				Name_face = facesbank_Name[temp_indexs]
			else:
				Name_face = "Stranger"
			cv2.putText(image_show, Name_face , (int(rectangle[0]),int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

	cv2.imshow("CamFace", image_show)
	cv2.waitKey(0)

def Detection_recognition_Folder():
	cv2.namedWindow('CamFace', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CamFace', 960, 640)

	model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
	extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
	ctx = mx.cpu()
	mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
	detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold= detect_threshold)

	Path_Folder = "dataimagedemo/Test/"
	count_numimg =0
	for fi in os.listdir(Path_Folder):
		if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS:
			continue
		image= cv2.imread(Path_Folder + fi)
		image_detect =image.copy()
		image_show = image_detect.copy()
		
		ret_mtcnn = detector.detect_face(image_detect, det_type = 0)
		if ret_mtcnn is None:
			rectangles , points_mtcnn = [],[]
			text = "Face not detected in an image"
			cv2.putText(image_show, text , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		else:
			rectangles , points_mtcnn = ret_mtcnn
			count_numbox =0
			for ( rectangle, point_mtcnn) in zip (rectangles, points_mtcnn):
				if rectangle[4] < 0.79:
					continue
				cv2.rectangle(image_show,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
				feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)
				temp_score, temp_indexs = cosine_face (feature_face, facesbank_feat, facesbank_idface)

				if temp_score <= recognition_threshold:
					Name_face = facesbank_Name[temp_indexs]
				else:
					Name_face = "Stranger"
				cv2.putText(image_show, Name_face , (int(rectangle[0]),int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

		# cv2.imwrite("imgDR_%d.jpg" %count_numimg , image_show)
		count_numimg +=1
		cv2.imshow("CamFace", image_show)
		cv2.waitKey(0)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("c") or key == 32:
			continue
		elif key == ord("q"):
			break

if __name__ == '__main__':
	# Detection_recognition_img()
	Detection_recognition_Folder()

