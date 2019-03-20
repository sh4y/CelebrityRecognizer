import cv2 as cv2
import numpy as np
import operator
import matplotlib.pyplot as plt

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

frames = get_video_frames('C:\Users\Shay\Documents\CSC420\Project\\test.mp4')
frame1 = frames[0]
frame2 = frames[1]
kp1, des1 = get_sift_frame_kp_desc(frame1)
kp2, des2 = get_sift_frame_kp_desc(frame2)
des_frame1, des_frame2 = compute_descriptor_dictionaries(kp1, kp2, des1, des2)
matches_between_frames = compute_potential_matches(des1, des2)
kp_pairs = compute_kp_pairs(matches_between_frames, des_frame1, des_frame2, kp1, kp2)
ref_x, ref_y = map(list,zip(*kp_pairs.keys()))
test_x, test_y = map(list,zip(*kp_pairs.values()))
visualize_matches(frame1, frame2, ref_x, ref_y, test_x, test_y)
