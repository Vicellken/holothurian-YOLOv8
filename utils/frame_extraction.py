import cv2
import os
import numpy as np

######################
##### user input #####
video_directory = "C:\\Users\\user\\Desktop\\video"  # directory of the video
video_name = "video.mp4"  # name of the video
interval = 4  # frame every _ seconds
##### user input #####
######################

os.chdir(video_directory)  # change directory
video = cv2.VideoCapture(video_name)  # import video

nframe = video.get(cv2.CAP_PROP_FRAME_COUNT)  # no. of frames
fps = video.get(cv2.CAP_PROP_FPS)  # frame per second
duration = nframe / fps  # duration
aspect_ratio = video.get(cv2.CAP_PROP_FRAME_WIDTH) / \
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # aspect ratio

qc_frames = (np.arange(-2, 3) * fps + nframe / 2).astype(
    int)  # grab five frames from the middle to see if the interval is appropriate
qc_imgs = []  # intialize a list to store extracted frames
for f in qc_frames:
    video.set(cv2.CAP_PROP_POS_FRAMES, f - 1)  # set to the desired frame
    _, img = video.read()  # extract that frame
    qc_imgs.append(img)  # add frame to list
qc_imgs = [cv2.resize(i, (320, int(320 // aspect_ratio)))
           for i in qc_imgs]  # lower the resolutions of extracted frames
qc_out = np.concatenate(qc_imgs, axis=1)  # combine frames into one image
for i, j in enumerate(np.arange(-2, 3)):
    qc_out = cv2.putText(qc_out, f"{j * interval}s", (0 + i * 320, int(320 // aspect_ratio) - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # put text to image to indicate time
cv2.imshow("Quality Check", qc_out)
cv2.waitKey()  # show the combined frames for quality check; proceed if good

# create a new sub-directory to store extracted frames
os.mkdir(f".\\{video_name.split('.')[0]}")
# go into the newly created sub-directory
os.chdir(f".\\{video_name.split('.')[0]}")
ex_frames = np.arange(1, nframe, int(fps * interval)
                      ).astype(int)  # frames to be extracted
for i, f in enumerate(ex_frames, start=1):
    print(f"Progress: {int(i / len(ex_frames) * 100)}%")
    video.set(cv2.CAP_PROP_POS_FRAMES, f - 1)  # set to the desired frame
    _, img = video.read()  # extract that frame
    # save extracted frame as jpg files
    cv2.imwrite(f"{video_name.split('.')[0]}_{i}.jpg", img)
