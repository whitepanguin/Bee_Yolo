# routes_cls.py
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Depends
from pydantic import BaseModel
from PIL import Image
import io, base64

import torch
from cls_model import BeeClassifier

router = APIRouter(prefix="/classify", tags=["classify"])

# --- 단순 DI 컨테이너(애플리케이션에서 주입) ---
def get_classifier() -> BeeClassifier:
    # 앱 시작 시 생성해 app.state에 넣고 여기서 꺼내 써도 됨.
    # 임시로 전역 싱글턴처럼 보관:
    global _CLS_SINGLETON
    try:
        c = _CLS_SINGLETON
    except NameError:
        _CLS_SINGLETON = BeeClassifier(ckpt_path="./models/best_resnet18_bee.pth")
        c = _CLS_SINGLETON
    if not c.loaded:
        c.load()
    return c

# ---------- 스키마 ----------
class ClassifyOneResp(BaseModel):
    label: str
    prob: float
    idx: int

class ClassifyManyReq(BaseModel):
    crops_b64: List[str]   # "data:image/png;base64,..." 도 허용
    strip_header: bool = True  # True면 dataURL prefix를 자동 제거

class ClassifyManyResp(BaseModel):
    results: List[ClassifyOneResp]

# ---------- 라우트 ----------
@router.get("/health")
def clf_health(cls: BeeClassifier = Depends(get_classifier)):
    return {
        "classifier_loaded": cls.loaded,
        "ckpt": cls.ckpt_path,
        "device": str(cls.device),
        "num_classes": len(cls.idx2class) or 8,
        "idx2class": cls.idx2class,
    }

@router.post("/crop", response_model=ClassifyOneResp)
async def classify_crop(file: UploadFile = File(...), cls: BeeClassifier = Depends(get_classifier)):
    # 단일 크롭 파일(jpg/png) 업로드
    try:
        im = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")
    res = cls.predict_pils([im])[0]
    return res

@router.post("/crops", response_model=ClassifyManyResp)
def classify_crops(req: ClassifyManyReq, cls: BeeClassifier = Depends(get_classifier)):
    # YOLO 감지에서 반환한 base64 png 리스트를 그대로 분류
    b64s = []
    for s in req.crops_b64:
        if req.strip_header and s.startswith("data:"):
            # "data:image/png;base64,...."
            s = s.split(",", 1)[-1]
        b64s.append(s)
    results = cls.predict_b64_pngs(b64s)
    return {"results": results}
