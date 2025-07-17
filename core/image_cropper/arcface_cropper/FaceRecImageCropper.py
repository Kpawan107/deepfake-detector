import cv2
import numpy as np
from insightface.utils import face_align

class FaceRecImageCropper:
    def crop_image_by_mat(self, image, landmarks):
        landmarks = np.array(landmarks).reshape((5, 2))
        aligned_face = face_align.norm_crop(image, landmarks)
        return aligned_face