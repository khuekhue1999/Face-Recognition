
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
try:
    from PyQt5 import sip
except ImportError:
    import sip
from PyQt5 import QtCore, QtGui, QtWidgets
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
from keras.models import load_model
import tensorflow as tf

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
with open("config/dev_config.json") as f:
	config = json.load(f)
__version__=config["version"]
detect_threshold = config["detdetect_threshold"]
recognition_threshold = config["recognition_threshold"]

ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])
ALLOWED_VIDEO_EXTENSIONS = set([ 'avi', 'mp4', 'mov', 'mkv'])

model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
ctx = mx.cpu()
mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True, threshold=detect_threshold)

model_path = 'modelfacenet/model/facenet_keras.h5'
modelfacenet = load_model(model_path)

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

class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("FaceRecognition")
		MainWindow.resize(1280, 680)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")

		self.label_Slect = QtWidgets.QLabel(self.centralwidget) 
		self.label_Slect.setGeometry(QtCore.QRect(140, 10, 230, 30))
		myFont=QtGui.QFont()
		myFont.setPointSize(16)
		myFont.setWeight(75)
		myFont.setBold(True)
		self.label_Slect.setFont(myFont)
		self.label_Slect.setStyleSheet('color: Red')

		self.radioButton_Cosine = QtWidgets.QRadioButton(self.centralwidget) 
		self.radioButton_Cosine.setGeometry(QtCore.QRect(160, 45, 95, 20))
		self.radioButton_Cosine.toggled.connect(self.load_cosin)  

		self.radioButton_SVM = QtWidgets.QRadioButton(self.centralwidget) 
		self.radioButton_SVM.setGeometry(QtCore.QRect(260, 45, 95, 20))
		self.radioButton_SVM.toggled.connect(self.load_SVM) 

		self.openCamera = QtWidgets.QPushButton(self.centralwidget)
		self.openCamera.setGeometry(QtCore.QRect(500, 35, 100, 30))
		self.openCamera.setObjectName("openCamera")

		self.Quiter_camera = QtWidgets.QPushButton(self.centralwidget)
		self.Quiter_camera.setGeometry(QtCore.QRect(750, 35, 100, 30))
		self.Quiter_camera.setObjectName("Exit")

		self.camerashow = QtWidgets.QLabel(self.centralwidget)
		self.camerashow.setGeometry(QtCore.QRect(60, 100, 960, 540))
		self.camerashow.setText("")
		# self.camerashow.setPixmap(QtGui.QPixmap("frame.png"))
		self.camerashow.setScaledContents(True)
		self.camerashow.setObjectName("camerashow")

		self.show_boxface = QtWidgets.QLabel(self.centralwidget)
		self.show_boxface.setGeometry(QtCore.QRect(1080, 200, 112, 112))
		self.show_boxface.setText("")
		self.show_boxface.setScaledContents(True)
		self.show_boxface.setObjectName("show_boxface")

		self.label_face = QtWidgets.QLabel(self.centralwidget) 
		self.label_face.setGeometry(QtCore.QRect(1050, 310, 200, 30))
		myFont=QtGui.QFont()
		myFont.setPointSize(16)
		myFont.setWeight(40)
		# myFont.setBold(True)
		self.label_face.setFont(myFont)
		self.label_face.setStyleSheet('color: Red')

		self.score_face = QtWidgets.QLabel(self.centralwidget) 
		self.score_face.setGeometry(QtCore.QRect(1050, 350, 200, 30))
		myFont=QtGui.QFont()
		myFont.setPointSize(16)
		myFont.setWeight(40)
		# myFont.setBold(True)
		self.score_face.setFont(myFont)
		self.score_face.setStyleSheet('color: Red')


		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 1172, 25))
		self.menubar.setObjectName("menubar")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

		self.openCamera.clicked.connect(self.slect_file)
		self.Quiter_camera.clicked.connect(sys.exit)

	def retranslateUi(self, MainWindow):
		self._translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(self._translate("Face Recognition", "FaceRecognition"))
		self.openCamera.setText(self._translate("Face Recognition", "Browse"))
		self.Quiter_camera.setText(self._translate("Face Recognition", "Exit"))
		self.radioButton_Cosine.setText(self._translate("Face Recognition", "Cosine")) 
		self.label_Slect.setText(self._translate("Face Recognition", "You click to select:")) 
		self.radioButton_SVM.setText(self._translate("Face Recognition", "SVM"))
		self.label_face.setText(self._translate("Face Recognition", ""))
		self.score_face.setText(self._translate("Face Recognition", ""))   

	def load_cosin(self, selected): 
		if selected:
			self.model_using = "Cosine"
			f = open("Facenet_database_featureface.json", 'r')
			datajson = json.load(f)
			self.facesbank_Name = []
			self.facesbank_feat = []
			self.facesbank_idface = []
			for i in range (len (datajson["data"])):
				for j in range (len(datajson["data"][i]["features"])):
					self.facesbank_Name.append(datajson["data"][i]["NameFace"])
					self.facesbank_feat.append(datajson["data"][i]["features"][j])
					self.facesbank_idface.append(datajson["data"][i]["idface"])

	def load_SVM(self, selected): 
		if selected:
			self.model_using = "SVM"
			path_modelSVM = "facesnet_SVM_finalized_model.sav"
			self.Model_SVM = pickle.load(open(path_modelSVM, 'rb'))

			self.Model_SVM_ClassName = []
			self.Model_SVM_ClassCode = []
			f = open("facesnet_SVM_finalized_model.json", 'r')
			dataname = json.load(f)
			for i in range(len(dataname["ClassName"])):
				self.Model_SVM_ClassName.append(dataname["ClassName"][i])
				self.Model_SVM_ClassCode.append(dataname["code"][i])

	def slect_file(self):
		path=None
		path_select = '/home/'
		self.targetPath = QFileDialog.getOpenFileName(None, "Open video file or image", path_select, "(*.jpg *.png *jpeg *bmp *tif *mp4 *avi)")
		if self.targetPath[0].split(".")[-1] in ALLOWED_IMAGE_EXTENSIONS:
			image = cv2.imread(self.targetPath[0])
			image_detect= image.copy()
			image_show = image.copy()
			ret_mtcnn = detector.detect_face(image_detect, det_type=0)
			if ret_mtcnn is None:
				rectangles, points_mtcnn = [], []
				text = "Face not detected in an image"
				cv2.putText(image_show, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			else:
				rectangles, points_mtcnn = ret_mtcnn
				count_numbox = 0
				for (rectangle, point_mtcnn) in zip(rectangles, points_mtcnn):
					height_bbox = int(rectangle[3]) - int(rectangle[1])
					with_bbox = int(rectangle[2]) - int(rectangle[0])
					if rectangle[4] < 0.8 or height_bbox * with_bbox < 80 * 70:
						continue
					cv2.rectangle(image_show, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 2)
					# feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)

					# boxface = extract_feature.Aglineface(image_detect, rectangle, point_mtcnn)
					boxface = image_detect[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
					aligned = cv2.resize(boxface, (160, 160))
					face_pixels = aligned.astype('float32')
					mean, std = face_pixels.mean(), face_pixels.std()
					face_pixels = (face_pixels - mean) / std
					samples_face = np.expand_dims(face_pixels, axis=0)
					feature_face = modelfacenet.predict(samples_face)

					image_face= image_detect[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
					frame_face = cv2.cvtColor(image_face.copy(), cv2.COLOR_BGR2RGB)
					frame_bboxface = QImage(frame_face, frame_face.shape[1], frame_face.shape[0], frame_face.strides[0], QImage.Format_RGB888)
					self.show_boxface.setPixmap(QPixmap.fromImage(frame_bboxface))
					self.score_face.clear()
					self.label_face.clear()
					if self.model_using =="Cosine":
						temp_score, temp_indexs = cosine_face(feature_face, self.facesbank_feat, self.facesbank_idface)
						if temp_score <= recognition_threshold:
							Name_face = self.facesbank_Name[temp_indexs] +" - " + str(round(temp_score, 3))
						else:
							Name_face = "Stranger " +" - " + str(round(temp_score, 3))
						self.score_face.setText("Score: " + Name_face.split("-")[-1])
						cv2.putText(image_show, Name_face, (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
					elif self.model_using == "SVM":
						yhat_class = self.Model_SVM.predict(feature_face)
						yhat_prob = self.Model_SVM.predict_proba(feature_face)
						class_index = yhat_class[0]
						class_probability = yhat_prob[0, class_index] * 100
						# if (class_probability>99.5):
						# Name_face = self.Model_SVM_ClassName[self.Model_SVM_ClassCode.index(class_index)] + " - " +str(round(float(class_probability/100), 3))
						Name_face = self.Model_SVM_ClassName[self.Model_SVM_ClassCode.index(class_index)]
						# else:
						# else:
						# 	Name_face = "Stranger" + " - " +str(round(float(class_probability/100), 3))
						cv2.putText(image_show, Name_face , (int(rectangle[0]), int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
					self.label_face.setText("Name: " + Name_face.split("-")[0])
					# self.score_face.setText("Score: " + Name_face.split("-")[-1])
			frame_color = cv2.cvtColor(image_show.copy(), cv2.COLOR_BGR2RGB)
			frame_show = QImage(frame_color, frame_color.shape[1], frame_color.shape[0], frame_color.strides[0], QImage.Format_RGB888)
			self.camerashow.setPixmap(QPixmap.fromImage(frame_show))
		else:
			self.capture = cv2.VideoCapture(self.targetPath[0])
			# self.capture = cv2.VideoCapture(0)
			self.timer = QTimer()
			self.timer.timeout.connect(self.display_video_stream)
			self.timer.start(30)

	def display_video_stream(self):
		_, self.frame = self.capture.read()
		time.sleep(0.03)
		image = self.frame.copy()
		image_detect = image.copy()
		image_show = image.copy()
		ret_mtcnn = detector.detect_face(image_detect, det_type=0)
		if ret_mtcnn is None:
			text = "Face not detected in an image"
			cv2.putText(image_show, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		else:
			rectangles, points_mtcnn = ret_mtcnn
			for (rectangle, point_mtcnn) in zip(rectangles, points_mtcnn):
				height_bbox = int(rectangle[3]) - int(rectangle[1])
				with_bbox = int(rectangle[2]) - int(rectangle[0])
				if rectangle[4] < 0.8 or height_bbox * with_bbox < 80 * 70:
					continue
				cv2.rectangle(image_show, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 2)
				boxface = image_detect[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
				aligned = cv2.resize(boxface, (160, 160))
				face_pixels = aligned.astype('float32')
				mean, std = face_pixels.mean(), face_pixels.std()
				face_pixels = (face_pixels - mean) / std
				samples_face = np.expand_dims(face_pixels, axis=0)
				feature_face = modelfacenet.predict(samples_face)

				image_face = image_detect[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
				frame_face = cv2.cvtColor(image_face.copy(), cv2.COLOR_BGR2RGB)
				frame_bboxface = QImage(frame_face, frame_face.shape[1], frame_face.shape[0], frame_face.strides[0], QImage.Format_RGB888)
				self.show_boxface.setPixmap(QPixmap.fromImage(frame_bboxface))
				self.score_face.clear()
				self.label_face.clear()
				if self.model_using == "Cosine":
					temp_score, temp_indexs = cosine_face(feature_face, self.facesbank_feat, self.facesbank_idface)
					if temp_score <= recognition_threshold:
						Name_face = self.facesbank_Name[temp_indexs] + " - " + str(round(temp_score, 3))
					else:
						Name_face = "Stranger " + " - " + str(round(temp_score, 3))
					cv2.putText(image_show, Name_face, (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX,
								0.75, (255, 0, 0), 1)
					self.score_face.setText("Score: " + Name_face.split("-")[-1])
				elif self.model_using == "SVM":
					yhat_class = self.Model_SVM.predict(feature_face)
					yhat_prob = self.Model_SVM.predict_proba(feature_face)
					class_index = yhat_class[0]
					class_probability = yhat_prob[0, class_index] * 100
					Name_face = self.Model_SVM_ClassName[self.Model_SVM_ClassCode.index(class_index)]
					cv2.putText(image_show, Name_face, (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX,
								0.75, (255, 0, 0), 1)
				self.label_face.setText("Name: " + Name_face.split("-")[0])
				# self.score_face.setText("Score: " + Name_face.split("-")[-1])
		frame_color = cv2.cvtColor(image_show.copy(), cv2.COLOR_BGR2RGB)
		frame_show = QImage(frame_color, frame_color.shape[1], frame_color.shape[0], frame_color.strides[0], QImage.Format_RGB888)
		self.camerashow.setPixmap(QPixmap.fromImage(frame_show))

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())