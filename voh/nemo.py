from torch.nn import functional as F

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils import logging


def read_nemo(f=None):

    logging.setLevel(logging.ERROR)
    return (
        nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        if f is None
        else nemo_asr.models.EncDecSpeakerLabelModel.restore_from(f)
    )


class TT:
    def __init__(self, model):

        if isinstance(model, EncDecSpeakerLabelModel):
            self.from_nemo = True
        else:
            self.from_nemo = False
        model.eval()
        self.md = model

    def embed(self, f):
        return (self.md.get_embedding if self.from_nemo else self.md.embed)(f)

    def cosim(self, a, b):
        return F.cosine_similarity(self.embed(a), self.embed(b)).item()

    def verify(self, a, b, threshold=0.7):
        return self.cosim(a, b) >= threshold
