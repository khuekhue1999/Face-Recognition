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
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn.manifold import TSNE
image_dir_basepath = 'dataimagedemo/BoxFace/'
names = os.listdir(image_dir_basepath)

with open("Arcface_Model_SVM.json", "w") as outfile:
    data = json.dumps({"data": names})
    outfile.write(data)

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
        image = cv2.imread(fi)
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

    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    Model_SVM = "Arcface_Model_SVM.pkl"
    with open(Model_SVM, 'wb') as file:
        pickle.dump(clf, file)
    # return le, clf

def test(dir_basepath, names, max_num_img=10):
    labels = []
    embs = []
    for name in names:
        dirpath = os.path.abspath(dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths)
        labels.extend([name] * len(embs_))
        embs.append(embs_)

    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    return embs, y

if __name__ == '__main__':
    #train(image_dir_basepath, names)
    # print(names)
    embs_test, y_test = test("dataimagedemo/test/", names)
    path_modelSVM = "facesnet_SVM_finalized_model.sav"
    Model_SVM = pickle.load(open(path_modelSVM, 'rb'))

    plt.figure(figsize=(10, 10))
    y_pred = Model_SVM.predict(embs_test)

    print("classification_report",  classification_report(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    mat = confusion_matrix(y_pred, y_test)
    confu = sns.heatmap(mat, cmap="Blues", annot=True, fmt="d", cbar=False,
                        xticklabels=names,
                        yticklabels=names,
                        linewidths=1.5, linecolor="lightblue")
    confu.set(xlabel="Predicted", ylabel="True")
    confu.set_yticklabels(confu.get_yticklabels(), va="center", rotation=90)
    # plt.savefig('abc.png',bbox_inches='tight')

    plt.show()
    #//////
    # X_embedded = TSNE(n_components=2).fit_transform(embs_test)
    #
    # for i, t in enumerate(set(y_test)):
    #     idx = y_test == t
    #     plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=names[t])
    #
    # plt.legend(bbox_to_anchor=(1, 1))
    # plt.show()
