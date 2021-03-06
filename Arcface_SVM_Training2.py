import matplotlib.pyplot as plt
import sys, os
import numpy as np
import cv2
import json
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from skimage.transform import resize
from recogniton import featureFace
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.manifold import TSNE

image_dir_train = 'dataimagedemo/BoxFace/'
image_dir_test= 'dataimagedemo/test/'
names = os.listdir(image_dir_train)

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])
image_size = 112
model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))

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
    # print(y)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    Model_SVM = "arcface_SVM_finalized_model.sav"
    with open(Model_SVM, 'wb') as file:
        pickle.dump(clf, file)
def load_faces(filepaths):
    faces = list()
    for fi in os.listdir(filepaths):
        if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS :
            continue
        img = cv2.imread(filepaths +fi)
        image = cv2.resize(img, (112, 112))
        face_img = np.asarray(image)
        faces.append(face_img)
    return faces


def load_dataset (directory):
    X_face, Y_label = list(), list()
    names_label = []
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'
        if not os.path.isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X_face.extend(faces)
        Y_label.extend(labels)
        names_label.append(subdir)
    return np.asarray(X_face), np.asarray(Y_label), names_label

def get_embedding(model, face_pixels):
    # face_pixels = face_pixels.astype('float32')
    # mean, std = face_pixels.mean(), face_pixels.std()
    # face_pixels = (face_pixels - mean) / std
    # samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.Feature_face(face_pixels, None, None)
    return yhat

def get_dataTrain():
    trainX, trainy, name_labeltrain = load_dataset(image_dir_train)
    testX, testy, name_labeltest = load_dataset(image_dir_test)
    out_encoder = LabelEncoder()
    out_encoder.fit(name_labeltrain)
    trainy_name = out_encoder.transform(name_labeltrain)

    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(extract_feature, face_pixels)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)
    X_embedded = TSNE(n_components=2).fit_transform(newTrainX)
    for i, t in enumerate(set(trainy)):
        idx = trainy == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1));

    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(extract_feature, face_pixels)
        newTestX.append(embedding)
    newTestX = np.asarray(newTestX)
    np.savez_compressed('arcface_embeddings.npz', newTrainX,trainy, newTestX, testy)

if __name__ == '__main__':
    # train(image_dir_train, names)
    # get_dataTrain()
    data = np.load('arcface_embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
    # precision, recall, fscore, support = score(testy, yhat_test)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    print("classification_report", classification_report(testy, yhat_test))
    print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
    # precision, recall, fscore, support = score(testy, yhat_test)

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:

        disp = plot_confusion_matrix(model, testX, testy, cmap=plt.cm.Blues,normalize=normalize,display_labels=names)
        disp.ax_.set_title(title)
    plt.show()