import cv2
import json
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from skimage.transform import resize
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.manifold import TSNE
import numpy as np
image_dir_train = 'dataimagedemo/BoxFace/'
image_dir_test= 'dataimagedemo/te/'
import matplotlib.pyplot as plt
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])
image_size = 160
model_path = 'modelfacenet/model/facenet_keras.h5'


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
print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
precision, recall, fscore, support = score(testy, yhat_test)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, testX, testy, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
plt.show()