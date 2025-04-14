import face_recognition
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


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
    
if __name__=="__main__":
    fc = FacialRecognizer()
    
    # Get images
    image = fc._read_image("input_images/input.jpg")
    
    boxes = fc.extract_boxes(image)
    
    # Extract all faces in all images
    extracted_faces = fc.get_cropped_faces(image, boxes)
    plt.figure(figsize=(3 * len(extracted_faces), 3))  # width scales with number of faces
    for i, img in enumerate(extracted_faces):
        plt.subplot(1, len(extracted_faces), i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('output.png', dpi=300)  # Lower DPI is often enough if figure is larger
    plt.show()
    
    # Choose the images to save
    boxes = boxes[:4]
    
    # Create embeddings
    embeddings = fc.calculate_embeddings(image, boxes)
    # print(embeddings)
    # print(len(embeddings))
