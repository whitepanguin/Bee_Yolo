import os
import io
import cv2
import base64
import time
import math
import numpy as np
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes_cls import router as cls_router
# -------------------------
# Config (robust weights resolver)
# -------------------------
import os
from pathlib import Path

IMG_SIZE = int(os.getenv("IMG_SIZE", "416"))
NC = int(os.getenv("NC", "1"))  # bee=1

DEFAULT_WEIGHTS_NAME = "yolov3_best(50E).pt"   # 파일명 고정
ENV_WEIGHTS = os.getenv("WEIGHTS_PATH", "")    # 주어지면 최우선

BASE_DIR = Path(__file__).resolve().parent     # 이 파일이 있는 폴더
CWD = Path.cwd()                               # 현재 작업 디렉터리

def _candidate_paths(name_or_path: str):
    p = Path(name_or_path)
    if p.is_absolute():
        # 절대경로면 그대로 시도
        yield p
        return
    # 상대경로면 여러 후보를 순서대로 시도
    # 1) 이 파일 옆 models/
    yield BASE_DIR / "models" / name_or_path
    # 2) 이 파일 옆
    yield BASE_DIR / name_or_path
    # 3) 현재 작업 디렉터리 models/
    yield CWD / "models" / name_or_path
    # 4) 현재 작업 디렉터리 바로 아래
    yield CWD / name_or_path
    # 5) 상위 디렉터리의 models/ (uvicorn을 상위에서 실행하는 경우 대비)
    yield BASE_DIR.parent / "models" / name_or_path

def resolve_weights_path() -> str:
    tried = []
    # 1) 환경변수 우선
    if ENV_WEIGHTS:
        for cand in _candidate_paths(ENV_WEIGHTS):
            tried.append(str(cand))
            if cand.is_file():
                return str(cand)
    # 2) 기본 파일명으로 탐색
    for cand in _candidate_paths(DEFAULT_WEIGHTS_NAME):
        tried.append(str(cand))
        if cand.is_file():
            return str(cand)
    # 못 찾으면 에러
    raise FileNotFoundError(
        "Could not locate YOLOv3 weights.\n"
        "Tried:\n  - " + "\n  - ".join(tried)
    )

try:
    DEFAULT_WEIGHTS = resolve_weights_path()
    print(f"[INFO] Using weights: {DEFAULT_WEIGHTS}")
except FileNotFoundError as e:
    # 로드 자체는 뒤에서 try/except로 한 번 더 처리되므로 경고만
    print(f"[WARN] {e}")
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_NAME  # 마지막 안전망(나중에 로드 시도)

CONF_THR_DEFAULT = float(os.getenv("CONF_THR", "0.50"))
IOU_THR_DEFAULT  = float(os.getenv("IOU_THR",  "0.45"))
BOX_MARGIN_RATIO_DEFAULT = float(os.getenv("BOX_MARGIN", "0.08"))

# -------------------------
# YOLOv3 (필요 최소 구현)
# -------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c//2, 1, 1)
        self.cv2 = Conv(c//2, c, 3, 1)
    def forward(self, x):
        return x + self.cv2(self.cv1(x))

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = Conv(3, 32, 3, 1)
        self.cv2 = Conv(32, 64, 3, 2);  self.res1 = nn.Sequential(*[ResBlock(64) for _ in range(1)])
        self.cv3 = Conv(64,128, 3, 2);  self.res2 = nn.Sequential(*[ResBlock(128) for _ in range(2)])
        self.cv4 = Conv(128,256,3, 2);  self.res3 = nn.Sequential(*[ResBlock(256) for _ in range(8)])
        self.cv5 = Conv(256,512,3, 2);  self.res4 = nn.Sequential(*[ResBlock(512) for _ in range(8)])
        self.cv6 = Conv(512,1024,3,2);  self.res5 = nn.Sequential(*[ResBlock(1024) for _ in range(4)])
    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x); x = self.res1(x)
        x = self.cv3(x); x = self.res2(x)
        x = self.cv4(x); x3 = self.res3(x)   # 1/8
        x = self.cv5(x); x2 = self.res4(x)   # 1/16
        x = self.cv6(x); x1 = self.res5(x)   # 1/32
        return x1, x2, x3

class YOLOv3(nn.Module):
    def __init__(self, num_classes=1, anchors=None, img_size=416):
        super().__init__()
        self.nc = num_classes
        self.na = 3
        self.anchors = anchors or [
            [(116,90), (156,198), (373,326)],   # stride 32
            [(30,61),  (62,45),   (59,119)],    # stride 16
            [(10,13),  (16,30),   (33,23)]      # stride 8
        ]
        self.img_size = img_size
        self.backbone = Darknet53()
        self.head1 = nn.Sequential(Conv(1024, 512, 1), Conv(512, 1024, 3),
                                   Conv(1024,512, 1), Conv(512, 1024, 3),
                                   Conv(1024,512, 1))
        self.pred1 = nn.Conv2d(512, self.na*(5+self.nc), 1, 1, 0)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce1 = Conv(512, 256, 1)

        self.head2 = nn.Sequential(Conv(768, 256, 1), Conv(256, 512, 3),
                                   Conv(512,256, 1), Conv(256, 512, 3),
                                   Conv(512,256, 1))
        self.pred2 = nn.Conv2d(256, self.na*(5+self.nc), 1, 1, 0)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce2 = Conv(256, 128, 1)

        self.head3 = nn.Sequential(Conv(384, 128, 1), Conv(128, 256, 3),
                                   Conv(256,128, 1), Conv(128, 256, 3),
                                   Conv(256,128, 1))
        self.pred3 = nn.Conv2d(128, self.na*(5+self.nc), 1, 1, 0)

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        p1 = self.head1(x1); out1 = self.pred1(p1)
        u1 = self.up1(self.reduce1(p1))
        p2 = self.head2(torch.cat([u1, x2], 1)); out2 = self.pred2(p2)
        u2 = self.up2(self.reduce2(p2))
        p3 = self.head3(torch.cat([u2, x3], 1)); out3 = self.pred3(p3)
        return [out1, out2, out3]

def try_torchvision_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float):
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_thr)
    except Exception:
        # fallback 간단 NMS
        b = boxes.cpu().numpy()
        s = scores.cpu().numpy()
        x1,y1,x2,y2 = b[:,0], b[:,1], b[:,2], b[:,3]
        areas = (x2-x1)*(y2-y1)
        order = s.argsort()[::-1]
        keep = []
        while order.size>0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
            inter = w*h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds+1]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

@torch.no_grad()
def yolo_decode(outputs: List[torch.Tensor], img_size: int, num_classes: int, anchors):
    decoded = []
    device = outputs[0].device
    strides = [32, 16, 8]
    for i, out in enumerate(outputs):
        bs, ch, ny, nx = out.shape
        na = 3; no = 5 + num_classes
        out = out.view(bs, na, no, ny, nx).permute(0,1,3,4,2).contiguous()
        stride = strides[i]
        anc = torch.tensor(anchors[i], device=device).float() / stride
        xv, yv = torch.meshgrid(torch.arange(nx, device=device), torch.arange(ny, device=device), indexing='xy')

        x = (out[...,0].sigmoid() + xv) * stride
        y = (out[...,1].sigmoid() + yv) * stride
        w = (out[...,2].exp() * anc[:,0].view(na,1,1)) * stride
        h = (out[...,3].exp() * anc[:,1].view(na,1,1)) * stride
        obj = out[...,4].sigmoid()
        cls = out[...,5:].sigmoid() if num_classes>0 else None

        boxes = torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1).view(bs, -1, 4)
        obj = obj.view(bs, -1)
        if num_classes>0:
            cls = cls.view(bs, -1, num_classes)
        else:
            cls = torch.zeros(bs, boxes.shape[1], 0, device=device)
        decoded.append((boxes, obj, cls))
    boxes = torch.cat([d[0] for d in decoded], dim=1)
    obj = torch.cat([d[1] for d in decoded], dim=1)
    cls = torch.cat([d[2] for d in decoded], dim=1)
    return boxes, obj, cls

@torch.no_grad()
def postprocess(outputs, conf_thr=0.25, iou_thr=0.45, img_size=IMG_SIZE, num_classes=1, anchors=None):
    boxes, obj, cls = yolo_decode(outputs, img_size, num_classes, anchors)
    bs = boxes.size(0)
    results = []
    for b in range(bs):
        scores = obj[b]
        if num_classes>0 and cls.shape[-1]>0:
            cls_scores, cls_ids = cls[b].max(dim=1)
            scores = scores * cls_scores
        else:
            cls_ids = torch.zeros_like(scores, dtype=torch.long)
        keep = scores > conf_thr
        bxs = boxes[b][keep]
        scs = scores[keep]
        cids = cls_ids[keep]
        if bxs.numel()==0:
            results.append((torch.zeros((0,4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)))
            continue
        keep_idx = try_torchvision_nms(bxs, scs, iou_thr)
        results.append((bxs[keep_idx], scs[keep_idx], cids[keep_idx]))
    return results

# -------------------------
# Pre/Post utils
# -------------------------
def resize_with_padding(image_rgb: np.ndarray, target_size=IMG_SIZE, fill_value=114):
    h, w = image_rgb.shape[:2]
    r = min(target_size / h, target_size / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2; left = pad_w // 2
    canvas = np.full((target_size, target_size, 3), fill_value, dtype=np.uint8)
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas, r, left, top

def to_tensor(img_rgb_416: np.ndarray, device):
    t = torch.from_numpy(img_rgb_416).permute(2,0,1).float()/255.0
    return t.unsqueeze(0).to(device)

def boxes416_to_original(xyxy_416: np.ndarray, W0: int, H0: int, r: float, pad_x: float, pad_y: float):
    if len(xyxy_416)==0:
        return xyxy_416
    # 역-레터박스: (x - pad)/r
    xyxy = xyxy_416.copy().astype(np.float32)
    xyxy[:, [0,2]] = (xyxy[:, [0,2]] - pad_x) / r
    xyxy[:, [1,3]] = (xyxy[:, [1,3]] - pad_y) / r
    xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, W0-1)
    xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, H0-1)
    # invalid 제거
    w = np.maximum(0, xyxy[:,2]-xyxy[:,0]); h = np.maximum(0, xyxy[:,3]-xyxy[:,1])
    keep = (w>1) & (h>1)
    return xyxy[keep], keep

def crop_with_margin(img_bgr: np.ndarray, box: Tuple[int,int,int,int], margin_ratio=0.08):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw = x2 - x1; bh = y2 - y1
    mx = bw * margin_ratio; my = bh * margin_ratio
    x1 = int(max(0, math.floor(x1 - mx)))
    y1 = int(max(0, math.floor(y1 - my)))
    x2 = int(min(w-1, math.ceil (x2 + mx)))
    y2 = int(min(h-1, math.ceil (y2 + my)))
    if x2<=x1 or y2<=y1:  # 안정성
        return None
    return img_bgr[y1:y2, x1:x2].copy()

def bgr_to_png_base64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")

# -------------------------
# Load model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv3(num_classes=NC, img_size=IMG_SIZE).to(device).eval()

def load_weights(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weights not found: {path}")
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)  # {'model': state_dict} or state_dict
    model.load_state_dict(state, strict=False)
    return ckpt

_ckpt_meta = None
try:
    _ckpt_meta = load_weights(DEFAULT_WEIGHTS)
    print(f"[INFO] Loaded weights: {DEFAULT_WEIGHTS}")
except Exception as e:
    print(f"[WARN] Failed to load '{DEFAULT_WEIGHTS}': {e}\n"
          f"      → 서버는 기동하지만, 정확한 추론을 위해 올바른 가중치를 경로에 두세요.")

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="Bee YOLOv3 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(cls_router)

@app.get("/health")
def health():
    import torch
    ok = True  # 기존 체크들...
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # classifier health (optional)
    try:
        from routes_cls import get_classifier
        clf = get_classifier()
        cls_ok = clf.loaded
    except Exception:
        cls_ok = False
    return {
        "ok": ok,
        "device": device,
        "weights_loaded": True,   # 기존 값 유지
        "cls_loaded": cls_ok,     # ✅ 필드 추가(경로 변경 없음)
    }

# ---- 이미지 탐지+크롭 ----
@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(..., description="RGB image file"),
    conf_thr: float = Query(CONF_THR_DEFAULT, ge=0.0, le=1.0),
    iou_thr:  float = Query(IOU_THR_DEFAULT,  ge=0.0, le=1.0),
    return_crops: int = Query(1, description="1=base64 crops 반환, 0=메타만"),
    max_crops: int = Query(32, description="최대 크롭 수"),
    margin_ratio: float = Query(BOX_MARGIN_RATIO_DEFAULT, ge=0.0, le=0.5),
):
    data = await file.read()
    image_bytes = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse({"error": "failed to decode image"}, status_code=400)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]

    # letterbox → tensor
    canvas, r, pad_x, pad_y = resize_with_padding(img_rgb, IMG_SIZE)
    timg = to_tensor(canvas, device)

    # inference
    with torch.no_grad():
        outs = model(timg)
        (bxs, scs, cids) = postprocess(outs, conf_thr, iou_thr, IMG_SIZE, NC, model.anchors)[0]
        bxs = bxs.detach().cpu().numpy()
        scs = scs.detach().cpu().numpy()
        cids = cids.detach().cpu().numpy()

    # 416 → 원본 좌표
    bxs_orig, keep = boxes416_to_original(bxs, W0, H0, r, pad_x, pad_y)
    scs = scs[keep] if len(scs) else scs
    cids = cids[keep] if len(cids) else cids

    # 크롭
    dets = []
    crops = []
    order = np.argsort(-scs) if len(scs) else np.array([], int)
    order = order[:max_crops]
    for i in order:
        x1,y1,x2,y2 = map(int, bxs_orig[i])
        dets.append({
            "bbox": [int(x1),int(y1),int(x2),int(y2)],
            "score": float(scs[i]),
            "class_id": int(cids[i]) if len(cids) else 0,
            "class_name": "bee"
        })
        crop = crop_with_margin(img_bgr, (x1,y1,x2,y2), margin_ratio)
        if return_crops and crop is not None:
            crops.append({
                "bbox": [int(x1),int(y1),int(x2),int(y2)],
                "png_b64": bgr_to_png_base64(crop),
                # 분류기 연결 자리(나중에 ResNet18 연결): "cls": {"label": ..., "prob": ...}
            })

    return {
        "image_size": {"width": W0, "height": H0},
        "num_dets": int(len(dets)),
        "conf_thr": conf_thr, "iou_thr": iou_thr,
        "detections": dets,
        "crops": crops if return_crops else [],
    }

# ---- 동영상: N프레임마다 샘플링해서 감지 (요약 + 일부 크롭) ----
@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(..., description="MP4/AVI 등"),
    conf_thr: float = Query(CONF_THR_DEFAULT, ge=0.0, le=1.0),
    iou_thr:  float = Query(IOU_THR_DEFAULT,  ge=0.0, le=1.0),
    frame_stride: int = Query(5, ge=1, description="N프레임마다 추론"),
    max_frames: int = Query(300, ge=1, description="최대 처리 프레임 수(과도한 응답 방지)"),
    return_crops: int = Query(0, description="1이면 첫 몇 개 크롭 base64 포함"),
    crops_per_frame: int = Query(4, ge=0, le=16),
    margin_ratio: float = Query(BOX_MARGIN_RATIO_DEFAULT, ge=0.0, le=0.5),
):
    data = await file.read()
    tmp = "_tmp_video.bin"
    with open(tmp, "wb") as f:
        f.write(data)

    cap = cv2.VideoCapture(tmp)
    if not cap.isOpened():
        os.remove(tmp)
        return JSONResponse({"error": "failed to open video"}, status_code=400)

    results = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    processed = 0
    idx = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if idx % frame_stride != 0:
                idx += 1
                continue
            idx += 1
            processed += 1
            if processed > max_frames:
                break

            H0, W0 = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            canvas, r, pad_x, pad_y = resize_with_padding(rgb, IMG_SIZE)
            timg = to_tensor(canvas, device)

            outs = model(timg)
            bxs, scs, cids = postprocess(outs, conf_thr, iou_thr, IMG_SIZE, NC, model.anchors)[0]
            bxs = bxs.detach().cpu().numpy()
            scs = scs.detach().cpu().numpy()
            cids = cids.detach().cpu().numpy()

            bxs_orig, keep = boxes416_to_original(bxs, W0, H0, r, pad_x, pad_y)
            scs = scs[keep] if len(scs) else scs
            cids = cids[keep] if len(cids) else cids

            dets = []
            crops = []
            order = np.argsort(-scs) if len(scs) else np.array([], int)
            if return_crops:
                order = order[:crops_per_frame]

            for i2 in order:
                x1,y1,x2,y2 = map(int, bxs_orig[i2])
                dets.append({
                    "bbox": [x1,y1,x2,y2],
                    "score": float(scs[i2]),
                    "class_id": int(cids[i2]) if len(cids) else 0,
                })
                if return_crops:
                    crop = crop_with_margin(frame_bgr, (x1,y1,x2,y2), margin_ratio)
                    if crop is not None:
                        crops.append({"bbox": [x1,y1,x2,y2], "png_b64": bgr_to_png_base64(crop)})

            results.append({
                "frame_index": idx-1,
                "time_sec": float((idx-1)/fps) if fps>0 else None,
                "num_dets": int(len(bxs_orig)),
                "detections_preview": dets,       # 상위 일부(크롭 리밋 기준)
                "crops": crops if return_crops else []
            })

    cap.release()
    os.remove(tmp)
    return {
        "total_frames": total_frames,
        "fps": fps,
        "processed_frames": processed,
        "conf_thr": conf_thr, "iou_thr": iou_thr,
        "stride": frame_stride,
        "results": results
    }

# ---- 분류기 연결을 위한 자리 (ResNet18 훅) ----
# 나중에 ResNet18 코드 오면, crop ndarray(BGR) -> label/prob 반환하도록 붙이면 됩니다.
# def classify_bee_crop(crop_bgr: np.ndarray) -> dict:
#     return {"label": "수일벌-이탈리안", "prob": 0.93}
# 위 함수를 detect_image/detect_video 내부에서 crop 추출 직후 호출해 "cls": {...}로 담으면 끝.


if __name__ == "__main__":
    import uvicorn
    print(f"[INFO] Device: {device}, IMG_SIZE={IMG_SIZE}, NC={NC}")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1)

# uvicorn main:app --host 0.0.0.0 --port 8000