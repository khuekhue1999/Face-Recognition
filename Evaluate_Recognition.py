import sys, os
import numpy as np
import cv2
import json
import threading
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace
from scipy.spatial.distance import euclidean, cosine
from keras.models import load_model

ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

def cosine_face ( feature_face1, feature_face2):
    temp_similarity = cosine(feature_face1, feature_face2)
    return temp_similarity

class ArcFace():
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
        self.extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
    def extract_featurearcface(self, image1, image2):
        feature_face1 = self.extract_feature.Feature_face(image1, None, None)
        feature_face2 = self.extract_feature.Feature_face(image2, None, None)
        temp_similarity = cosine_face(feature_face1, feature_face2)
        return temp_similarity

class Facenet():
    def __init__(self):
        model_path = 'modelfacenet/model/facenet_keras.h5'
        self.modelfacenet = load_model(model_path)

    def prewhiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj
        return y
    def extract_featurefacenet(self, image1, image2):
        image1 = cv2.resize(image1, (160, 160))
        image2 = cv2.resize(image2, (160, 160))
        image_extract1 = []
        image_extract1.append(image1.copy())
        image_extract1 = self.prewhiten(np.array(image_extract1))
        feature_face1 = self.modelfacenet.predict_on_batch(image_extract1[0:1])

        image_extract2 = []
        image_extract2.append(image2.copy())
        image_extract2 = self.prewhiten(np.array(image_extract2))
        feature_face2 = self.modelfacenet.predict_on_batch(image_extract2[0:1])

        temp_similarity = cosine_face(feature_face1, feature_face2)
        return temp_similarity

def Eval_ArcFace():
    extractfeature_arcface = ArcFace()
    Path_Dir = "Data_evaluate _recognition/Negative/"
    threshold = 0.6
    # Action_Eval = "Positive"
    Action_Eval = "Negative"
    count_True =0
    count_False =0
    for Fol in os.listdir(Path_Dir):
        Path_DirFolder = Path_Dir + Fol + "/"
        if len(os.listdir(Path_DirFolder)) !=2:
            continue
        image1= cv2.imread(Path_DirFolder + os.listdir(Path_DirFolder)[0])
        image2 = cv2.imread(Path_DirFolder + os.listdir(Path_DirFolder)[1])
        temp_similarity = extractfeature_arcface.extract_featurearcface(image1, image2)
        if Action_Eval == "Positive":
            if temp_similarity <= threshold:
                count_True +=1
            else:
                count_False +=1
        elif Action_Eval == "Negative":
            if temp_similarity > threshold:
                count_True +=1
            else:
                count_False +=1
    Acc= float(count_True/(count_True+count_False)*100)
    print("Arcface Evaluate recognition - " , Action_Eval , "- Accuracy(%): ", Acc)

def Eval_Facenet():
    extractfeature_arcface = Facenet()
    Path_Dir = "Data_evaluate _recognition/Positive/"
    threshold = 0.4
    Action_Eval = "Positive"
    # Action_Eval = "Negative"
    count_True =0
    count_False =0
    for Fol in os.listdir(Path_Dir):
        Path_DirFolder = Path_Dir + Fol + "/"
        if len(os.listdir(Path_DirFolder)) !=2:
            continue
        image1= cv2.imread(Path_DirFolder + os.listdir(Path_DirFolder)[0])
        image2 = cv2.imread(Path_DirFolder + os.listdir(Path_DirFolder)[1])
        temp_similarity = extractfeature_arcface.extract_featurefacenet(image1, image2)
        if Action_Eval == "Positive":
            if temp_similarity <= threshold:
                count_True +=1
            else:
                count_False +=1
        elif Action_Eval == "Negative":
            if temp_similarity > threshold:
                count_True +=1
            else:
                count_False +=1
    Acc= float(count_True/(count_True+count_False)*100)
    print("Facenet Evaluate recognition - " , Action_Eval , "- Accuracy(%): ", Acc)

if __name__ == '__main__':
	# Eval_ArcFace()
    Eval_Facenet()



