import face_recognition
import cv2
import numpy as np


class FacialRecognizer:
    def __init__(self):
        pass
    
    def extract_boxes(self, image, model="hog"):
        return face_recognition.face_locations(image,model=model)
    
    def get_cropped_faces(self, image, boxes=None):
        if boxes == None:
            boxes = self.extract_boxes(image)
        found_faces = []
        for (top, right, bottom, left) in boxes:
                found_faces.append(image[top:bottom, left:right])    
        return found_faces
    
    def calculate_embeddings(self, image, boxes=None):
        return face_recognition.face_encodings(image, boxes)
    
    def _read_image_path(self, path):
        image = cv2.imread(path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    
    def _read_st_file(self, st_file):
        file_bytes = np.asarray(bytearray(st_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
