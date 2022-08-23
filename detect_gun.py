import argparse
import os
import platform
import sys
from pathlib import Path
import os
path = 'G://ML_Projects//gun_detection//yolo_streamlit_gun_detection//Lib//site-packages'
os.environ['PATH'] += os.pathsep + path

import torch
import torch.backends.cudnn as cudnn


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class DetectGun():

    def __init__(self):
        self.device = ''
        self.imgsz=(640, 640)
        self.dnn = False
        self.data = 'coco128.yml'
        self.half = False
        self.weights= 'last.pt'
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.bs = 1
        self.conf_thres=0.25
        self.iou_thres=0.45
        self.classes=None
        self.agnostic_nms=False
        self.max_det=1000
        self.save_crop=False
        self.line_thickness=2
        self.hide_labels=False
        self.hide_conf=False

    def detect_gun(self,img):
        dataset = LoadImages(img, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

        with dt[1]:
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=False, visualize=False)

        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)  # label format

                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    im0 = annotator.result()
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        
        return im0
        