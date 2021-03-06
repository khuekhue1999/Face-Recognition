import matplotlib.pyplot as plt
import sys, os
import numpy as np
import cv2
import json, uuid
from keras.models import load_model

model_path = 'modelfacenet/model/facenet_keras.h5'
modelfacenet = load_model(model_path)
ALLOWED_IMAGE_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'bmp', 'jfif', 'tif', 'tiff'])

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

def ExtractFeature_Savedatabase():
    Path_Dir = "dataimagedemo/BoxFace/"
    Data_facebank = []
    for Fol in os.listdir(Path_Dir):
        Path_DirFolder = Path_Dir + Fol + "/"
        features_face =[]
        for fi in os.listdir(Path_DirFolder):
            if fi.split(".")[-1] not in ALLOWED_IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(Path_DirFolder + fi)
            image = cv2.resize(image, (160, 160))
            image_extract =[]
            image_extract.append(image.copy())
            image_extract = prewhiten(np.array(image_extract))
            # image_extract = image_extract.reshape(list(image_extract.shape) + [1])
            feature_face = modelfacenet.predict_on_batch(image_extract[0:1])
            features_face.append(feature_face[0].tolist())
        name_face = Fol
        person_id = str(uuid.uuid4().hex)
        person_info = {"NameFace": str(name_face), "features": features_face, "idface": str(person_id)}
        Data_facebank.append(person_info)

    with open("Facenet_database_featureface.json", "w") as outfile:
        data = json.dumps({"data": Data_facebank})
        outfile.write(data)

if __name__ == '__main__':
    ExtractFeature_Savedatabase()