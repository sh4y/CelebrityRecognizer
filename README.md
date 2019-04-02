# CelebrityRecognizer
CSC420 Final Project


Project Proposal: Actor Face Detector in Movies

This project was born out of our love for movies and is a twist on the Project 1 as defined in the assignment handout. Instead of news clips, we plan to use movie clips.

1)	Split movie clip into individual frames - DONE! 
See function: get_video_frames(file)
Pass movie clip in as parameter, returns dictionary of {frame_number:frame} where frame is a RGB matrix.


2)	Detect shots in movie clip. Compute percentage of how well shots are found.
a.	Compare frame to the previous by computing SIFT keypoints on each frame of the scene => switched scenes when number of keypoints are less than threshold
b.	Manually annotate scenes, find where algorithm detects scene and obtain percentage of how many are correct
3)	Detect faces in the clip. Track faces.
a.	Use OpenCV’s HaarCascades library to detect these faces – find optimal parameters to reduce false positives

Example:

![Screenshot](https://i.imgur.com/0dzuBnY.png)

b.	Track faces by drawing a rectangle around each face
c.	Store location of face
4)	Train a classifier to detect actor faces using Microsoft dataset (MSRA-CFW: Dataset of Celebrity Faces on the Web). 
a.	Curate dataset by cropping out non face related content (ie; only select headshots)
b.	For ease of use purposes, remove celebrities which won’t appear in the clip
c.	Go through curated dataset and manually remove any potential false positives (extreme costumes/makeup from other movies or out of date (wrong age) headshots)
5)	Classify each detected face in movie clip as actor or not (if actor, write names)
a.	Using previously detected faces (in step 2), run face through classifier and assign a label
b.	Do this for every single face detected in the shot
6)	Bounding box saying name of actor(s) in each shot.
