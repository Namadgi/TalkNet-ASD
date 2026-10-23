import torch.nn as nn

from src.ml_models.talk_modules import TalkNetModel, CLF

class TalkNet(nn.Module):

    def __init__(self):
        super(TalkNet, self).__init__()        
        self.backbone = TalkNetModel()
        self.classifier = CLF()

    def forward(self, x_audio, x_video):
        embedA = self.backbone.forward_audio_frontend(x_audio)
        embedV = self.backbone.forward_visual_frontend(x_video)    
        embedA, embedV = self.backbone.forward_cross_attention(embedA, embedV)
        out = self.backbone.forward_audio_visual_backend(embedA, embedV)
        score = self.classifier(out)
        return score