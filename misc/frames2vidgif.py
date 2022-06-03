import cv2
import numpy as np
import glob
import os
from moviepy.editor import VideoFileClip

demo_path = 'misc/plots/gif/baseline-cond.mrqa-01/'
demo_frames_path = os.path.join(demo_path, '*.png')
out_vid = os.path.join(demo_path, 'plot.mov')
out_gif = os.path.join(demo_path, 'plot.gif')
img_array = []
n_frames = 1000

print('reading frames...')
filenames = sorted(glob.glob(demo_frames_path))
for i, filename in enumerate(filenames):
    if i == n_frames:
        break
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

print('writing video...')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15
out = cv2.VideoWriter(out_vid, fourcc, fps, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print('done!')

print('writing gif...')
# gif hyperparams
resize_factor = 0.5
speed_factor = 2.0
duration = len(filenames) / (fps * speed_factor)
start_ts = (0, 0.0) # minute, seconds
stop_ts = (0, duration) # minute, seconds

# get clip
clip = (VideoFileClip(out_vid)
        .subclip(start_ts,stop_ts)
        .resize(resize_factor))

# write to gif
clip.speedx(speed_factor).write_gif(out_gif, fps=int(clip.fps / speed_factor))

# # using ffmpeg
# os.system('ffmpeg -ss {} -t {} -i {} -vf "fps={},scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {}'.format(start_ts[1], stop_ts[1], out_vid, int(clip.fps/speed_factor), out_gif))
print('done!')