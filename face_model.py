from __future__ import absolute_import

import os
import numpy as np
import mxnet as mx
import cv2
from sklearn.preprocessing import normalize
from skimage import transform as trans

def preprocess(img, bbox=None, landmark=None, **kwargs):
  M = None
  image_size = []
  str_image_size = kwargs.get('image_size', '')
  if len(str_image_size)>0:
    image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size)==1:
      image_size = [image_size[0], image_size[0]]
    assert len(image_size)==2
    assert image_size[0]==112
    assert image_size[0]==112 or image_size[1]==96
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

  if M is None:
    if bbox is None:
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = kwargs.get('margin', 44)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret 
  else:
    assert len(image_size)==2
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    return warped

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model


class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.cpu()
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

  def get_img(self, face_img, bbox,points  ):
    
    # bbox = bbox[0,0:4]
    points = points[:].reshape((2,5)).T

    nimg = preprocess(face_img, bbox, points, image_size=self.args.image_size)
    # aligned = np.transpose(nimg, (2,0,1))
    return nimg


  def get_input(self, face_img, bbox,points  ):
    
    # bbox = bbox[0,0:4]
    if points is not None:
      points = points[:].reshape((2,5)).T
    

    nimg = preprocess(face_img, bbox, points, image_size=self.args.image_size)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = normalize(embedding).flatten()
    return embedding
