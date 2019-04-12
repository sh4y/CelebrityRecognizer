import cv2 as cv2
import os
import numpy as np
import operator
import matplotlib.pyplot as plt
import face_recognition

import subprocess
# from subprocess import call, Popen, PIPE

#from tensor_classifier.classify import classify_face

print("PATH of main.py is: " + str(os.environ['PYTHONPATH']))

def get_video_frames(file):
    capture = cv2.VideoCapture(file)
    result, image = capture.read()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    count = 0
    frames[count] = image
    while result:
        result, image = capture.read()
        count += 1
        frames[count] = image
    return frames


def compute_descriptor_dictionaries(kp_frame1, kp_frame2, des_frame1, des_frame2):
    des_ref_dict = {}
    des_test_dict = {}
    for i in range(len(kp_frame1)):
        des_ref_dict[i] = des_frame1[i]
    for i in range(len(kp_frame2)):
        des_test_dict[i] = des_frame2[i]
    return des_ref_dict, des_test_dict


def compute_potential_matches(des_ref, des_test):
    potential_matches = {}
    for descriptor in des_ref:
        #print '==============='
        distances = {}
        for pair in des_test:
            dist = np.linalg.norm(descriptor-pair)
            distances[tuple(pair)] = dist
            #distances.update({tuple(pair): dist})
        distances = sorted(distances.items(), key=operator.itemgetter(1))
        major_distance = distances[0][1]
        #print 'major: ' + str(major_distance)
        minor_distance = distances[1][1]
        #print 'minor: ' + str(minor_distance)
        ratio = major_distance / minor_distance
        #print ratio
        if ratio < 0.7:
            #append to potential matches
            #print 'Appending to pot match'
            if tuple(descriptor) not in potential_matches:
                potential_matches[tuple(descriptor)] = distances[0][0]
    return potential_matches


def compute_kp_pairs(potential_matches, des_frame1, des_frame2, kp_frame1, kp_frame2):
    kp_pairs = {}
    for descriptor in potential_matches:
        pair = potential_matches[descriptor]
        for i in range(len(kp_frame1)):
            flattened_descriptor = np.array(descriptor).flatten()
            if np.array_equal(np.array(des_frame1[i]), flattened_descriptor):
                # found descriptor, use i to find keypoint
                ref_kp = kp_frame1[i]
                break
        for i in range(len(kp_frame2)):
            if np.array_equal(np.array(des_frame2[i]), np.array(pair)):
                test_kp = kp_frame2[i]
                break
        kp_pairs[ref_kp.pt] = test_kp.pt
    return kp_pairs


def visualize_matches(ref_img, test_img, ref_x, ref_y, test_x, test_y, n=10):
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[0].imshow(ref_img)
    axarr[0].scatter(ref_x[:n],ref_y[:n], s=50, c = ["r", "b", "g"])
    axarr[1].axis('off')
    axarr[1].imshow(test_img)
    axarr[1].scatter(test_x[:n], test_y[:n],s=50, c = ["r", "b", "g"])
    plt.show()


def get_sift_frame_kp_desc(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


# returns a dictionary of key point matches between frames
def compute_shared_keypoints(frame1, frame2):
    kp1, des1 = get_sift_frame_kp_desc(frame1)
    kp2, des2 = get_sift_frame_kp_desc(frame2)
    des_frame1, des_frame2 = compute_descriptor_dictionaries(kp1, kp2, des1, des2)
    matches_between_frames = compute_potential_matches(des1, des2)
    kp_pairs = compute_kp_pairs(matches_between_frames, des_frame1, des_frame2, kp1, kp2)
    return kp_pairs


def get_shots(frames):
    shots = {}
    for frame_count in range(len(frames)-2):
        print (frame_count)
        frame1 = frames[frame_count]
        frame2 = frames[frame_count+1]
        kp_pairs = compute_shared_keypoints(frame1, frame2)
        if len(kp_pairs) < 10:
            print ('Threshold not met. Changed scene.')
            shot_count = len(shots.keys())
            if shot_count not in shots:
                shots[shot_count] = [frame_count, frame_count+1]
            else:
                shots[shot_count].append(frame_count)
                shots[shot_count].append(frame_count+1)
    return shots

print ('Opening video file')
video_frames = get_video_frames('./avengers_hangar_scene.mp4')
#print(get_shots(video_frames))
print ('Loading Classifier')
face_cascade = cv2.CascadeClassifier()

cascade = cv2.CascadeClassifier()
#cascade.load(r'C:\Users\haksh\Documents\CSC420\PROJECT\face.xml')

annotated_movie = {}

print ('Opening output file')
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1280,720))

#face = cv2.imread(r'C:\Users\haksh\Documents\CSC420\PROJECT\avengers.jpg')
print ('Entering loop.')
for frame_count in range(len(video_frames)-1):
    #print video_frames[frame_count].shape
    #gray = cv2.cvtColor(video_frames[frame_count], cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(video_frames[frame_count], cv2.COLOR_BGR2RGB)
    #faces = cascade.detectMultiScale(gray, 1.2, 5)
    faces = face_recognition.face_locations(rgb)

    img = video_frames[frame_count]
    for face_location in faces:
        print("\tLocated face on Frame " + str(frame_count))
        top, right, bottom, left = face_location
        img = cv2.rectangle(video_frames[frame_count], (left, top), (right, bottom), (255,0,0), 2)
        face = video_frames[frame_count][top:bottom, left:right]

        cv2.imwrite('temp.jpg', face)

        print("\tRunning Command")
        p = subprocess.Popen(['/Users/Bipen/anaconda3/bin/python classify.py temp.jpg'], stdout=subprocess.PIPE, shell=True)
        stdout, stderr = p.communicate()
        value = p.returncode

        # p.wait()
        #output= subprocess.check_output("grep 'hello' tmp", shell=True)
        print("\tSTDOUT: ")
        print("\t" + stdout)

        result = stdout.split("\n")[-2]

        cv2.imshow("[Frame " + str(frame_count) + "] " + result, face)
        cv2.waitKey(5000)
        #print p.stdout.read()
        # print("\tSTDERR: ")
        # print(stderr)
        # print("\tRETURN VALUE: ")
        # print(value)
        #output = p.stderr.read()
        #print(output)

        #print("return code: " + str(rc))
        #print("output: " + str(output))
        #print("------------------")
        #name = os.system('python3.7 ./classify.py temp.jpg')

        os.remove('temp.jpg')
        # print(name)

    annotated_movie[frame_count] = img

    out.write(img)


#for image in cropped_faces:
 #   cv2.imshow('', image)
  #  cv2.waitKey(1000)

out.release()


#ref_x, ref_y = map(list,zip(*kp_pairs.keys()))
#test_x, test_y = map(list,zip(*kp_pairs.values()))
#visualize_matches(frame1, frame2, ref_x, ref_y, test_x, test_y)


