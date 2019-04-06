import cv2 as cv2
import numpy as np
import operator
import matplotlib.pyplot as plt
import os
import face_recognition
import PIL


# For each image of the actor, crop according to the recognized face and save them in a database
def crop_images(actor_name):
    folder = "/Users/Bipen/Documents/University/*CSC420/Project/Avengers_Dataset/" + actor_name
    print("<Crop_Images> Attempting to curate database for actor '" + actor_name + "'")
    print("Found " + str(len(os.listdir(folder))) + " images to look through.")
    for image_name in os.listdir(folder):
        if ".png" in image_name or ".jpg" in image_name:
            print("\tAttempting image: " + image_name)
            image = face_recognition.load_image_file(folder + "/" + image_name)
            face_locations = face_recognition.face_locations(image)

            image_num = 1
            for face in face_locations:
                top, right, bottom, left = face

                face_image = image[top:bottom, left:right]
                pil_image = PIL.Image.fromarray(face_image)
                #pil_image.show()

                pil_image.save("./Actor_Database/" + actor_name + "/" + image_name + "_" + str(image_num) + ".png")


                # cv2.imwrite("./Actor_Database/" + actor_name + "/[CROPPED]" + actor_name + "_" + str(image_num) + ".png", face_image)
                image_num += 1


if __name__ == "__main__":
    actors = ["chris_evans", "scarlett_johansson", "robert_downey"]

    for actor in actors:
        crop_images(actor)