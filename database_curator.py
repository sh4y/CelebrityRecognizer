import cv2 as cv2
import numpy as np
import operator
import matplotlib.pyplot as plt
import os
import face_recognition
import PIL


# For each image of the actor, crop according to the recognized face and save them in a database
def crop_images(actor_name):
    #folder = "/Users/Bipen/Documents/University/*CSC420/Project/Avengers_Dataset/" + actor_name
    folder = "./tensor_classifier/training_dataset/" + actor_name
    print("<Crop_Images> Attempting to curate database for actor '" + actor_name + "'")
    print("Found " + str(len(os.listdir(folder))) + " images to look through.")
    for image_name in os.listdir(folder):
        if ".png" in image_name or ".jpg" in image_name:
            print("\tAttempting image: " + image_name)
            image = face_recognition.load_image_file(folder + "/" + image_name)
            face_locations = face_recognition.face_locations(image)
            print("\t\tFound " + str(len(face_locations)) + " face(s).")
            image_num = 1
            for face in face_locations:
                top, right, bottom, left = face

                face_image = image[top:bottom, left:right]
                pil_image = PIL.Image.fromarray(face_image)
                #pil_image.show()

                if not os.path.exists("./Cropped_Actor_Database/" + actor_name):
                    os.makedirs("./Cropped_Actor_Database/" + actor_name)

                pil_image.save("./Cropped_Actor_Database/" + actor_name + "/" + str(image_num) + "_" + image_name)


                # cv2.imwrite("./Actor_Database/" + actor_name + "/[CROPPED]" + actor_name + "_" + str(image_num) + ".png", face_image)
                image_num += 1


if __name__ == "__main__":
    actors = ["Scarlett_Johansson"]

    for actor in actors:
        crop_images(actor)
