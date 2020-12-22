import cv2
import numpy as np
from glob import glob
import requests
import os


# https://towardsdatascience.com/self-supervised-attention-mechanism-for-dense-optical-flow-estimation-b7709af48efd
def get_video(video_url):
    r = requests.get(video_url, stream=True)
    with open('./vid.mp4', 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)


def estimate_optical_flow(video, frame_dir):
    ret, frame1 = video.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    seq = 1
    while(1):
        ret, frame2 = video.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(f"{frame_dir}/{seq}.png", rgb)
        seq += 1
        if seq == 200:
            break
    video.release()


def generate_output(frame_dir):
    img_array = []
    for filename in sorted(glob(f"{frame_dir}/*.png")):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./Dense-optical-flow.mp4', fourcc, 20.0, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def main():
    video_url = "https://viratdata.org/video/VIRAT_S_010204_05_000856_000890.mp4"
    get_video(video_url)
    video = cv2.VideoCapture("./vid.mp4")
    if not os.path.exists('./frames'):
        os.mkdir('./frames')
    estimate_optical_flow(video, './frames')
    generate_output('./frames')


if __name__ == "__main__":
    main()
