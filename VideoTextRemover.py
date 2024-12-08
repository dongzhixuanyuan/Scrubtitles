#Imports
import cv2
import os
import numpy as np
import re 
import time
import sys
try:
    # pip uninstall moviepy
    # pip install moviepy==1.0.3
    from moviepy.editor import VideoFileClip
    print("MoviePy 已成功安装")
except ImportError:
    print("MoviePy 未安装，请运行: pip install moviepy")
    sys.exit(1)


#This is the folder the video will be in, and a subfolder for the temporary processing files.
dir = '/Users/dongliu/android/code/others/video_subtitles_remove/Scrubtitles/videos_src/' #Place video directory here
video_name = 'changcheng_bj.mp4' #Place video name here

#Sorts alphanumerically with frame formatting
def sortedproper( l ):    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#Used for finding execution time
start_time = time.time()

#Video capture
vid = cv2.VideoCapture(os.path.join(dir, video_name))
frame_counter = 0

#Checks if Temp folder exists, if not, make one
if not os.path.exists(os.path.join(dir,'Temp')):
    os.mkdir(os.path.join(dir,'Temp'))
    print("Directory " , os.path.join(dir,'Temp') ,  " Created ")
else:    
    print("Directory " , os.path.join(dir,'Temp') ,  " already exists")

os.chdir(os.path.join(dir,'Temp'))

#Main process
#Runs through every frame to detect subtitles
#Saves each frame into Temp folder
while (frame_counter < vid.get(cv2.CAP_PROP_FRAME_COUNT)):
    ret, img = vid.read()
    name = "frame%d.jpg" % (frame_counter)

    if not ret:
        break

    if not os.path.exists(os.path.join(dir,"Temp/frame%d.jpg" % (frame_counter))):
        mask = np.zeros(img.shape, np.uint8)
        recogImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        recogImg = cv2.threshold(recogImg, 240, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        recogImg = cv2.morphologyEx(recogImg, cv2.MORPH_CLOSE, kernel)
        recogImg = cv2.dilate(recogImg, kernel, iterations=3)

        contours, hierarchy = cv2.findContours(recogImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)        
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[y:y+h, x:x+w] = recogImg[y:y+h, x:x+w]
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (3,3), 0)

        if len(mask.shape) > 2:  
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        cleanedImg = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)      
        cv2.imwrite(name, cleanedImg)
        frame_counter += 1

#Move back to original video directory and begin saving frames into one video
os.chdir(dir)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from 0 to mp4v codec
temp_output = 'temp_output.mp4'  # Temporary file for video without audio
video = cv2.VideoWriter(temp_output, fourcc, int(vid.get(cv2.CAP_PROP_FPS)), (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

images = [img for img in os.listdir(os.path.join(dir,'Temp')) if img.endswith(".jpg")]
images = [os.path.join(dir, 'Temp', img) for img in os.listdir(os.path.join(dir,'Temp')) if img.endswith(".jpg")]

images = sortedproper(images)
for image in images:
    print(image)
    video.write(cv2.imread(image))

#Cleanup and print execution time
video.release()
vid.release()

# Add audio from original video to the new video
original = VideoFileClip(os.path.join(dir, video_name))
new_video = VideoFileClip(temp_output)
final_video = new_video.set_audio(original.audio)
final_video.write_videofile('output.mp4', codec='libx264', audio_codec='aac')

# Clean up temporary files
os.remove(temp_output)
original.close()
new_video.close()
final_video.close()

print("--- %s seconds ---" % (time.time() - start_time))
