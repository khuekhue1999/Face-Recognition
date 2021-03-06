import  pickle
import numpy as np
import  cv2
import  json
import  os
from recogniton import featureFace
from mtcnn_detector import MtcnnDetector
import  mxnet as mx
with open("config/dev_config.json") as f:
	config = json.load(f)
__version__=config["version"]
detect_threshold = config["detdetect_threshold"]

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


model_path = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model')
extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
ctx = mx.cpu()

extract_feature = featureFace(model_path=model_path, epoch_num='0000', image_size=(112, 112))
mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True, threshold=detect_threshold)

image = cv2.imread(r"dataimagedemo/BoxFace/Amme/116426018_781380959358593_7752236382008300809_o.jpg")
image_detect = image.copy()
image_show = image.copy()
ret_mtcnn = detector.detect_face(image_detect, det_type=0)
rectangle, point_mtcnn = ret_mtcnn
feature_face = extract_feature.Feature_face(image_detect, rectangle, point_mtcnn)
features_face = []
features_face.append([feature_face])
embs_face = l2_normalize(np.concatenate(features_face))
print(embs_face.shape)
temp = embs_face.reshape(-1, 1)
print(temp.shape)
# def load_SVM(self, selected):
#     if selected:
#         self.model_using = "SVM"
#         path_modelSVM = "arcface_SVM_finalized_model.sav"
#         self.Model_SVM = pickle.load(open(path_modelSVM, 'rb'))
#
#         self.Model_SVM_ClassName = []
#         self.Model_SVM_ClassCode = []
#         f = open("arcface_SVM_finalized_model.json", 'r')
#         dataname = json.load(f)
#         for i in range(len(dataname["ClassName"])):
#             self.Model_SVM_ClassName.append(dataname["ClassName"][i])
#             self.Model_SVM_ClassCode.append(dataname["code"][i])

#
path_modelSVM = "arcface_SVM_finalized_model.sav"
Model_SVM = pickle.load(open(path_modelSVM, 'rb'))
# class_index=Model_SVM.predict((embs_face.reshape(1, -1))[0])
#
print(Model_SVM)
