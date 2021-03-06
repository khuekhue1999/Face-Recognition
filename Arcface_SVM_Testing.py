import sys, os
import numpy as np
import cv2
import json
import threading
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace
from scipy.spatial.distance import euclidean, cosine
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
with open("config/dev_config.json") as f:
    config = json.load(f)
__version__ = config["version"]
detect_threshold = config["detdetect_threshold"]
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])
path_modelSVM = "Arcface_Model_SVM.pkl"
with open(path_modelSVM, 'rb') as file:
    Model_SVM = pickle.load(file)

Model_SVM_ClassName =[]
f = open("Arcface_Model_SVM.json", 'r')
dataname = json.load(f)
for i in range (len(dataname["data"])):
    Model_SVM_ClassName.append(dataname["data"][i])

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def Detection_recognition_img():
    cv2.namedWindow('CamFace', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CamFace', 960, 640)

    model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
    extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
    ctx = mx.cpu()
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                             threshold=detect_threshold)

    image = cv2.imread("dataimagedemo/Test/Bich_Phuong_0000.jpg")
    image_detect = image.copy()
    image_show = image_detect.copy()

    ret_mtcnn = detector.detect_face(image_detect, det_type=0)
    if ret_mtcnn is None:
        rectangles, points_mtcnn = [], []
        text = "Face not detected in an image"
        cv2.putText(image_show, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        rectangles, points_mtcnn = ret_mtcnn
        count_numbox = 0
        for (rectangle, point_mtcnn) in zip(rectangles, points_mtcnn):
            if rectangle[4] < 0.79:
                continue
            features_face =[]
            cv2.rectangle(image_show, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 2)
            # feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)
            feature_face = [extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)]
            features_face.append(feature_face)
            embs_face = l2_normalize(np.concatenate(features_face))
            Name_face = Model_SVM_ClassName[Model_SVM.predict(embs_face)[0]]
            print(Name_face)

            cv2.putText(image_show, Name_face, (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)

    cv2.imshow("CamFace", image_show)
    cv2.waitKey(0)


def Detection_recognition_Folder():
    cv2.namedWindow('CamFace', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CamFace', 960, 640)

    model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
    extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
    ctx = mx.cpu()
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                             threshold=detect_threshold)

    Path_Folder = "dataimagedemo/Test/"
    count_numimg = 0
    for fi in os.listdir(Path_Folder):
        if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(Path_Folder + fi)
        image_detect = image.copy()
        image_show = image_detect.copy()

        ret_mtcnn = detector.detect_face(image_detect, det_type=0)
        if ret_mtcnn is None:
            rectangles, points_mtcnn = [], []
            text = "Face not detected in an image"
            cv2.putText(image_show, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            rectangles, points_mtcnn = ret_mtcnn
            count_numbox = 0
            for (rectangle, point_mtcnn) in zip(rectangles, points_mtcnn):
                if rectangle[4] < 0.79:
                    continue
                features_face = []
                cv2.rectangle(image_show, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),(255, 0, 0), 2)
                # feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)
                feature_face = [extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)]
                features_face.append(feature_face)
                embs_face = l2_normalize(np.concatenate(features_face))
                Name_face = Model_SVM_ClassName[Model_SVM.predict(embs_face)[0]]
                print(Name_face)
                cv2.putText(image_show, Name_face, (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        count_numimg += 1
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

