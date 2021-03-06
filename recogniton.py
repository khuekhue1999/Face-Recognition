import face_model as face_model
import argparse


class featureFace:
    def __init__(self, model_path, epoch_num='0000', image_size=(112, 112),
                 no_face_raise=True):
        self.model_path = ','.join([model_path, epoch_num])
        self.no_face_raise = no_face_raise
        args = argparse.Namespace()
        args.model = self.model_path
        args.det = 0
        args.flip = 0
        args.threshold = 1.24
        args.ga_model = ''
        args.image_size = ",".join([str(i) for i in image_size])
        self.model = face_model.FaceModel(args)

    def Feature_face(self, image, bbox, points ):
        preprocessed = self.model.get_input(image, bbox, points )
        if preprocessed is None:
            if self.no_face_raise:
                raise Exception("No face detected!")
            else:
                return None

        embedding = self.model.get_feature(preprocessed)
        return embedding

    def Aglineface(self, image, bbox, points ):
        preprocessed = self.model.get_img(image, bbox, points )
        return preprocessed
        
    def Extract_Featuer (self, image):
        embedding = self.model.get_feature(image)
        return embedding
