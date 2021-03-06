import sys, os
import numpy as np
import cv2
import json
import threading
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace
from mtcnn.mtcnn import MTCNN
import cv2
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
with open("config/dev_config.json") as f:
	config = json.load(f)
__version__=config["version"]

detect_threshold =config["detdetect_threshold"]
recognition_threshold =config["recognition_threshold"]

def cut_face(filename, required_size):
    results = []
    image = cv2.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(image)
    x1, y1, width, height = results[0]['box']
    crop_img = image[y1:y1 + height, x1:x1 + width]
    img = cv2.resize(crop_img, required_size)
    return img

def MTCNNdetection_SaveBoxFace_Folder():

	cv2.namedWindow('CamFace', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CamFace', 960, 640)

	model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
	extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
	ctx = mx.cpu()
	mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
	detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold= detect_threshold)

	ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'bmp'])

	Path_Dir = "dataimagedemo/images/"
	for Fol in os.listdir(Path_Dir):
		Path_DirFolder = Path_Dir + Fol +"/"
		count_numimg = 0
		Path_folder_save_boxface = "dataimagedemo/BoxFace/" + Fol
		if not os.path.exists(Path_folder_save_boxface):
			os.makedirs(Path_folder_save_boxface)
		for fi in os.listdir(Path_DirFolder):
			if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS:
				continue
			path_img = (Path_DirFolder + fi)
			image= cv2.imread(path_img)
			image_detect =image.copy()
			image_show = image_detect.copy()

			#ret_mtcnn = detector.detect_face(image_detect, det_type = 0)
			detector = MTCNN()
			ret_mtcnn = detector.detect_faces(image_detect)
			if ret_mtcnn is None:
				rectangles , points_mtcnn = [],[]
				text = "Face not detected in an image"
				cv2.putText(image_show, text , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			else:
				try:
					bounding_box = ret_mtcnn[0]['box']
					confidence = ret_mtcnn[0]['confidence']
					if (confidence >= 0.7):
						cv2.rectangle(image,
									  (bounding_box[0], bounding_box[1]),
									  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
									  (0, 0, 255),
									  2)
						x1, y1, width, height = ret_mtcnn[0]['box']
						crop_img = image[y1:y1 + height, x1:x1 + width]
						boxface = cv2.resize(crop_img,(112,112))
						cv2.imwrite(Path_folder_save_boxface + "/" + fi, boxface)
						count_numimg += 1
				except IndexError:
					pass
			cv2.imshow("CamFace", image_show)
			cv2.waitKey(1)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

if __name__ == '__main__':
	MTCNNdetection_SaveBoxFace_Folder()