# cls_model.py
import io, base64
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


class BeeClassifier:
    """
    ResNet18 벌 분류기 (8클래스: BI/CA/AP/LI × AB/QB)
    - ckpt는 사용자가 제공한 학습 스크립트 포맷을 그대로 가정
      keys: "model", "idx2class", "mean", "std"
    """
    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None):
        self.ckpt_path = ckpt_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.idx2class: List[str] = []
        self.tf = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self):
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        # 모델 구성
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)

        mean = ckpt.get("mean", [0.485, 0.456, 0.406])
        std  = ckpt.get("std",  [0.229, 0.224, 0.225])
        self.idx2class = ckpt.get("idx2class", [
            "BI_AB","BI_QB","CA_AB","CA_QB","AP_AB","AP_QB","LI_AB","LI_QB"
        ])

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.model = model
        self._loaded = True

    # --------------- 유틸 ---------------
    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    @staticmethod
    def _pil_from_b64_png(png_b64: str) -> Image.Image:
        raw = base64.b64decode(png_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # --------------- 추론 ---------------
    @torch.inference_mode()
    def predict_pils(self, crops: List[Image.Image], batch_size: int = 64) -> List[Dict]:
        """
        crops: PIL 이미지 리스트(크롭)
        return: [{label, prob, idx}, ...]
        """
        self._ensure_loaded()
        if not crops:
            return []

        # 전처리
        xs = [self.tf(im) for im in crops]
        X = torch.stack(xs, dim=0).to(self.device)

        probs_all: List[float] = []
        idx_all: List[int] = []

        for i in range(0, len(X), batch_size):
            logits = self.model(X[i:i+batch_size])
            probs  = F.softmax(logits, dim=1)
            p, idx = probs.max(dim=1)
            probs_all.extend(p.detach().tolist())
            idx_all.extend(idx.detach().tolist())

        out = []
        for p, ci in zip(probs_all, idx_all):
            label = self.idx2class[ci] if 0 <= ci < len(self.idx2class) else str(ci)
            out.append({"label": label, "prob": float(p), "idx": int(ci)})
        return out

    @torch.inference_mode()
    def predict_b64_pngs(self, b64_list: List[str], batch_size: int = 64) -> List[Dict]:
        crops = [self._pil_from_b64_png(b) for b in b64_list]
        return self.predict_pils(crops, batch_size=batch_size)
