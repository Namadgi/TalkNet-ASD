import os
import cv2
import time
import numpy as np
import python_speech_features
from scipy import signal
from scipy.interpolate import interp1d

IOU_THRESHOLD = 0.5
MIN_TRACK = 10
MIN_FAILED_DET = 10
MIN_FACE_SIZE = 1
THRESHOLD = 0.9
CROP_SCALE = 0.4

def postprocess_det(Y, frames, audio) -> dict:
    res_bboxes = []
    for frame_n, bboxes in enumerate(Y):
        # TODO something with multiple or zero faces
        if len(bboxes) == 0:
            continue
        bbox = bboxes[0]
        res_bboxes.append({'frame': frame_n, 'bbox': bbox})
    track = track_shot(res_bboxes)
    if isinstance(track, list):
        return []
    cur_time = time.time()
    try:
        res = crop_video(track, frames, audio)
    except (IndexError, KeyError, ValueError) as crop_err:
        print(f'Crop guard hit: {crop_err}')
        return []
    print('FC:', time.time() - cur_time)
    return res
    

def track_shot(frameFaces: list):
    # CPU: Face tracking
    print(frameFaces)
    while True:
        track = []
        for face in frameFaces[:]:
            if track == []:
                track.append(face)
                frameFaces.remove(face)
            elif face['frame'] - track[-1]['frame'] <= MIN_FAILED_DET:
                iou = __bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                if iou > IOU_THRESHOLD:
                    track.append(face)
                    frameFaces.remove(face)
        
        # if no tracks, then break
        if len(track) == 0:
            break
        # if length of tracks less than 11, continue loop
        if len(track) <= MIN_TRACK:
            continue
        
        frameNum = np.array([f['frame'] for f in track])
        bboxes   = np.array([np.array(f['bbox']) for f in track])
        frameI   = np.arange(frameNum[0], frameNum[-1] + 1)
        bboxesI  = []
        for ij in range(4):
            interpfn = interp1d(frameNum, bboxes[:,ij])
            bboxesI.append(interpfn(frameI))
        bboxesI  = np.stack(bboxesI, axis=1)
        if max(
                np.mean(bboxesI[:,2] - bboxesI[:,0]), 
                np.mean(bboxesI[:,3] - bboxesI[:,1])
            ) > MIN_FACE_SIZE:
            # only first track, can be modified
            return {'frame': frameI, 'bbox': bboxesI}
    return []


def __bb_intersection_over_union(boxA, boxB):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def crop_video(track, frames, audio) -> dict:
    # CPU: crop the face clips

    dets = {'x': [], 'y': [], 's': []}
    if 'bbox' not in track or 'frame' not in track:
        raise ValueError('Track data missing bbox or frame information')
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2) 
        dets['y'].append((det[1] + det[3]) / 2) # crop center x 
        dets['x'].append((det[0] + det[2]) / 2) # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    vOut = cv2.VideoWriter(
            'temp.avi', 
            cv2.VideoWriter_fourcc(*'XVID'), 
            25, 
            (224, 224)
        )# Write video
    
    for fidx, frame in enumerate(track['frame']):
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * CROP_SCALE))  # Pad videos by this amount 
        try:
            # Raw normalization supplies RGB; the crop/encoder path uses BGR.
            image = cv2.cvtColor(frames[frame], cv2.COLOR_RGB2BGR)
        except IndexError as e:
            raise IndexError(f'Frame index {frame} is outside the available range ({len(frames)} frames)') from e
        image = np.pad(
                image, 
                ((bsi, bsi), (bsi, bsi), (0, 0)),
                'constant', 
                constant_values=(110, 110)
            )
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = image[
                int(my - bs)                    : int(my + bs * (1 + 2 * CROP_SCALE)),
                int(mx - bs * (1 + CROP_SCALE)) : int(mx + bs * (1 + CROP_SCALE))
            ]
        vOut.write(cv2.resize(face, (224, 224)))
    vOut.release()
    
    if len(track['frame']) == 0:
        raise ValueError('Track contains no frame indices')

    # Slice PCM to the face-track span: 16 kHz / 25 fps = 640 samples per frame.
    # The cropped AVI is video-only; MFCC consumes the audio array directly.
    s0 = int(track['frame'][0]) * 640
    s1 = min((int(track['frame'][-1]) + 1) * 640, len(audio))
    audio_clip = audio[s0:s1]

    video_path = os.path.join(os.getcwd(), 'temp.avi')
    return {
            'video_path': video_path, 'audio': audio_clip
        }

def preprocess_talk(input_datas):
    # Take the input data and make it inference ready
    video_path = input_datas['video_path']
    audio = input_datas['audio']

    audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
    
    videoFeature = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        ret, frames = vidcap.read()
        if not ret:
            break
        face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224, 224))
        face = face[224 // 4 : 224 * 3 // 4, 224 // 4 : 224 * 3 // 4]
        videoFeature.append(face)
    vidcap.release()
    videoFeature = np.array(videoFeature)
    print(videoFeature.shape)
    length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25
        )
    audioFeature = audioFeature[:int(round(length * 100)), :]
    videoFeature = videoFeature[:int(round(length * 25)), :, :]
    return audioFeature, videoFeature, length


def nms_(dets, thresh):
    """
    Courtesy of Ross Girshick
    [https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep).astype(int)
