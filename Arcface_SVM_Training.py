import matplotlib.pyplot as plt
import sys, os
import numpy as np
import cv2
import json
import threading
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from recogniton import featureFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

image_dir_basepath = 'dataimagedemo/BoxFace/'
names = os.listdir(image_dir_basepath)

def prewhiten(x):
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
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])

def calc_embs(filepaths, batch_size=1):
    features_face =[]
    for fi in filepaths:
        if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS :
            continue
        img = cv2.imread(fi)
        image = cv2.resize(img, (112,112))
        image_extract =image.copy()
        feature_face = [extract_feature.Feature_face(image_extract, None, None)]
        features_face.append(feature_face)
    embs = l2_normalize(np.concatenate(features_face))
    return embs

def train(dir_basepath, names, max_num_img=10):
    labels = []
    embs = []
    for name in names:
        dirpath = os.path.abspath(dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths)
        labels.extend([name] * len(embs_))
        embs.append(embs_)
    out_encoder = LabelEncoder()
    out_encoder.fit(names)
    trainy_name = out_encoder.transform(names)
    with open("arcface_SVM_finalized_model.json", "w") as outfile:
        data = json.dumps({"ClassName": names, "code": trainy_name.tolist()})
        outfile.write(data)

    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    print(y)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    Model_SVM = "arcface_SVM_finalized_model.sav"
    with open(Model_SVM, 'wb') as file:
        pickle.dump(clf, file)
    # return le, clf


if __name__ == '__main__':
    train(image_dir_basepath, names)