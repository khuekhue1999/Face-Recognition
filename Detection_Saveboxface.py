import sys, os
import numpy as np
import cv2
import json
import threading
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
with open("config/dev_config.json") as f:
	config = json.load(f)
__version__=config["version"]

detect_threshold =config["detdetect_threshold"]
recognition_threshold =config["recognition_threshold"]

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

			ret_mtcnn = detector.detect_face(image_detect, det_type = 0)
			if ret_mtcnn is None:
				rectangles , points_mtcnn = [],[]
				text = "Face not detected in an image"
				cv2.putText(image_show, text , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			else:
				rectangles , points_mtcnn = ret_mtcnn

				for ( rectangle, point_mtcnn) in zip (rectangles, points_mtcnn):
					height_bbox = int(rectangle[3]) - int(rectangle[1])
					with_bbox = int(rectangle[2]) - int(rectangle[0])
					if rectangle[4] < 0.8 or height_bbox*with_bbox < 80*70:
						continue
					feature_face = extract_feature.Feature_face(image, rectangle, point_mtcnn)
					cv2.rectangle(image_show,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
					# boxface = extract_feature.Aglineface(image, rectangle, point_mtcnn)
					boxface = image_detect[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
					cv2.imwrite(Path_folder_save_boxface + "/Face_%d.jpg" %count_numimg , boxface)
					# cv2.imwrite(Path_folder_save_boxface + "/Frame_%d" + fi, boxface)
					count_numimg +=1
			cv2.imshow("CamFace", image_show)
			cv2.waitKey(1)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

if __name__ == '__main__':
	MTCNNdetection_SaveBoxFace_Folder()