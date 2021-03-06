import sys, os
import numpy as np
import cv2
import json
import threading
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace
import uuid, io


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
with open("config/dev_config.json") as f:
	config = json.load(f)
__version__=config["version"]

detect_threshold =config["detdetect_threshold"]
recognition_threshold =config["recognition_threshold"]
ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])


def ExtractFeature_Savedatabase():

	Path_Dir ="dataimagedemo/BoxFace/"
	model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
	extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
	ctx = mx.cpu()
	Data_facebank =[]
	for Fol in os.listdir(Path_Dir):
		Path_DirFolder = Path_Dir + Fol +"/"
		features_face =[]
		for fi in os.listdir(Path_DirFolder):
			if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS :
				continue
			image = cv2.imread(Path_DirFolder+ fi)
			image_extract =image.copy()
			feature_face = extract_feature.Feature_face(image_extract, None, None)
			features_face.append(feature_face.tolist())
		name_face = Fol
		person_id = str(uuid.uuid4().hex)
		person_info= {"NameFace": str(name_face), "features":features_face, "idface":str(person_id)}
		Data_facebank.append(person_info)

	with open("Arcface_database_featureface.json", "w") as outfile:
		data = json.dumps({"data":Data_facebank})
		outfile.write(data)

if __name__ == '__main__':
	ExtractFeature_Savedatabase()