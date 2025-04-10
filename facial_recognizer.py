import cv2
import matplotlib.pyplot as plt
from imgbeddings import imgbeddings
from PIL import Image


class FacialRecognizer:
    def __init__(self, face_model_path="haarcascade_frontalface_default.xml"):
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.image_embedder = imgbeddings()
    
    def extract_faces(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = self.haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

        # for each face detected
        cropped_faces = []
        for x, y, w, h in faces:
            # crop the image to select only the face
            cropped_faces.append(image[y : y + h, x : x + w])
            
            # loading the target image path into target_file_name variable
            # target_file_name = '<INSERT YOUR OUTPUT FACE IMAGE NAME HERE> for eg-> X2.jpg'
            # cv2.imwrite(target_file_name, cropped_image)
        return cropped_faces
    
    def calculate_embeddings(self, images):
        # img = [Image.open(image) for image in images]
        embeddings = [self.image_embedder.to_embeddings(image)[0] for image in images]
        # embedding = self.image_embedder.to_embeddings(img)[0]
        return embeddings
    
if __name__=="__main__":
    facial_recognizer = FacialRecognizer()
    image = cv2.imread("input_images/input.jpg")
    extracted_faces = facial_recognizer.extract_faces(image=image)
    faces_embeddings = facial_recognizer.calculate_embeddings(extracted_faces)
    print(faces_embeddings)

    # plt.figure(figsize=(3 * len(extracted_faces), 3))  # width scales with number of faces

    # for i, img in enumerate(extracted_faces):
    #     plt.subplot(1, len(extracted_faces), i+1)
    #     plt.imshow(img, cmap='gray')
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.savefig('my_figure.png', dpi=300)  # Lower DPI is often enough if figure is larger
    # plt.show()

