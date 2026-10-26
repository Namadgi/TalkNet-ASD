import time
import math
import torch
import torch.nn as nn

from src.ml_models.talk import TalkNet
from src.ml_models.mono_utils import postprocess_det, preprocess_talk
from ibug.face_detection import RetinaFacePredictor

class Mono(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        device = 'cuda'
        model_name = 'mobilenet0.25'
        self.talk = TalkNet()
        self.refd = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )

    def forward(self, X, v_path, a_path, device):
        cur_time = time.time()
        step = 10
        Y = []
        for i, x in enumerate(X):
            if i != len(X) - 1 and i % step != 0:
                Y.append([])
                continue

            bbox = self.refd(x, rgb=False)[:, :4]
            Y.append(bbox)

        print('FD:', time.time() - cur_time)
        det_res = postprocess_det(Y, v_path, a_path)
        
        if isinstance(det_res, list):
            return []
        cur_time = time.time()
        audio_X, video_X, length = preprocess_talk(det_res)
        print('VE:', time.time() - cur_time)
        cur_time = time.time()
        duration_set = set(range(1, 7))
        all_scores = []
        for d in duration_set:
            batchSize = int(math.ceil(length / d))
            scores = []
            for i in range(batchSize):
                inputA = torch.FloatTensor(audio_X[
                        i * d * 100 : (i + 1) * d * 100, :
                    ]).unsqueeze(0).to(device)
                inputV = torch.FloatTensor(video_X[
                        i * d *  25 : (i + 1) * d *  25, :, :
                    ]).unsqueeze(0).to(device)
                with torch.no_grad():
                    score = self.talk(inputA, inputV)
                scores.extend(score)
            all_scores.append(scores)
        print('VI:', time.time() - cur_time)
        
        return all_scores
        