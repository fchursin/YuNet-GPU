import cv2
import torch
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Sequence, Tuple

class YuNet:
    """
    Minimal YuNet ONNX wrapper (preprocess + postprocess) with optional CUDA post-processing.

    The wrapper:
      - Letterboxes an input image to 640x640 (no cropping), storing scale/padding.
      - Runs ONNX inference via onnxruntime.
      - Decodes multi-stride outputs into boxes + 5 keypoints + score.
      - Applies OpenCV DNN NMS (cv2.dnn.NMSBoxes).
      - Maps detections back from letterboxed coordinates to the original image coordinates.

    Output format per detection (15 floats):
      [x1, y1, w, h, kp0x, kp0y, kp1x, kp1y, kp2x, kp2y, kp3x, kp3y, kp4x, kp4y, score]
    """

    def __init__(self, model_path: str, cpu_only = False) -> None:
        """
        Args:
            model_path: Path to the YuNet ONNX model.
            providers: onnxruntime providers list, e.g.:
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                or [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"].
        """
        self.providers = self._build_providers(cpu_only)
        self.session: ort.InferenceSession = ort.InferenceSession(model_path, providers=self.providers)
        self.inp_name: str = self.session.get_inputs()[0].name

        self.scale: float = 1.0
        self.pad_x: int = 0
        self.pad_y: int = 0

        self.grid_cached: Dict[Tuple[int, int, str, Optional[int], torch.dtype], torch.Tensor] = {}

    def _build_providers(self, cpu_only) -> list:
        if cpu_only:
            return ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = []

        if "TensorrtExecutionProvider" in available:
            providers.append(("TensorrtExecutionProvider", {
            }))

        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {
            }))

        if "CoreMLExecutionProvider" in available:
            providers.append(("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "0",
                "EnableOnSubgraphs": "0",
            }))

        providers.append("CPUExecutionProvider")
        
        return providers


    def _resize_to_640(self, img: np.ndarray, size: int = 640, pad_color: Tuple[int, int, int] = (0, 0, 0),) -> np.ndarray:
        """
        Letterbox-resize to (size, size) without cropping.

        Stores:
            self.scale: resize scale applied to original image
            self.pad_x/self.pad_y: top-left padding applied to center the resized image

        Args:
            img: Input image (H,W,C) or (H,W).
            size: Target square size (default 640).
            pad_color: Padding color (B,G,R). For grayscale inputs, only the first value is used.

        Returns:
            Letterboxed image of shape (size, size, C) for color inputs, or (size, size, 1) for grayscale.
        """
        h, w = img.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        if img.ndim == 2:
            resized = resized[..., None]
            out = np.full((size, size, 1), pad_color[0], dtype=img.dtype)
        else:
            out = np.full((size, size, img.shape[2]), pad_color, dtype=img.dtype)

        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2
        out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        self.scale = float(scale)
        self.pad_x = int(pad_x)
        self.pad_y = int(pad_y)
        return out

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for YuNet ONNX inference.

        Steps:
          - Letterbox to 640x640.
          - Convert to float32.
          - Convert HWC -> CHW.
          - Add batch dimension: (1, C, 640, 640).

        Args:
            image: Input image as numpy array.

        Returns:
            Preprocessed tensor as numpy array float32.
        """
        image640 = self._resize_to_640(image)

        x = image640.astype(np.float32)
        x = np.transpose(x, (2, 0, 1)) 
        x = np.expand_dims(x, 0)        
        return x

    def _get_grid(self, rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get or create a cached (HW,2) grid of [r, c] indices for decoding.

        Args:
            rows: Feature map height.
            cols: Feature map width.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            Tensor of shape (rows*cols, 2) containing [r, c] for each cell.
        """
        key = (rows, cols)
        grid = self.grid_cached.get(key)
        if grid is None:
            r = torch.arange(rows, device=device, dtype=dtype)
            c = torch.arange(cols, device=device, dtype=dtype)
            rr, cc = torch.meshgrid(r, c, indexing="ij")
            grid = torch.stack((rr.reshape(-1), cc.reshape(-1)), dim=1)  
            self.grid_cached[key] = grid
        return grid

    @torch.no_grad()
    def _post_process(self, output: Sequence[np.ndarray], score_thresh: float, iou_thresh: float, top_k: int,) -> Optional[np.ndarray]:
        """
        Decode model outputs, apply thresholding + NMS, and map detections back to original coords.

        Args:
            output: List/tuple of 12 model outputs in the expected YuNet layout:
                [cls_s8, cls_s16, cls_s32,
                 obj_s8, obj_s16, obj_s32,
                 bbox_s8, bbox_s16, bbox_s32,
                 kps_s8, kps_s16, kps_s32]
                Shapes are typically (1, HW, 1) for cls/obj, (1, HW, 4) for bbox, (1, HW, 10) for kps.
            score_thresh: Score threshold applied before NMS.
            iou_thresh: IoU threshold for NMS.
            top_k: Max number of boxes kept by NMSBoxes.

        Returns:
            (N,15) numpy float array in original image coordinates, or None if no detections.
        """
        inputW = 640
        inputH = 640
        strides: Tuple[int, int, int] = (8, 16, 32)
        divisor = 32

        device = torch.device("cpu")
        pad_w = ((inputW - 1) // divisor + 1) * divisor
        pad_h = ((inputH - 1) // divisor + 1) * divisor

        all_faces: List[torch.Tensor] = []
        S = len(strides)

        for i, stride in enumerate(strides):
            cols = pad_w // stride
            rows = pad_h // stride
            HW = rows * cols

            cls = torch.as_tensor(output[i], device=device).reshape(HW).float()
            obj = torch.as_tensor(output[i + S], device=device).reshape(HW).float()
            bbox = torch.as_tensor(output[i + 2 * S], device=device).reshape(HW, 4).float()
            kps = torch.as_tensor(output[i + 3 * S], device=device).reshape(HW, 10).float()

            cls.clamp_(0.0, 1.0)
            obj.clamp_(0.0, 1.0)
            score = torch.sqrt(cls * obj)

            keep = score >= float(score_thresh)
            if not torch.any(keep):
                continue

            grid_rc = self._get_grid(rows, cols, device=device, dtype=bbox.dtype)
            r = grid_rc[:, 0]
            c = grid_rc[:, 1]

            cx = (c + bbox[:, 0]) * stride
            cy = (r + bbox[:, 1]) * stride
            w = torch.exp(bbox[:, 2]) * stride
            h = torch.exp(bbox[:, 3]) * stride
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h

            kp = kps.view(HW, 5, 2)
            kp_x = (kp[..., 0] + c.unsqueeze(1)) * stride
            kp_y = (kp[..., 1] + r.unsqueeze(1)) * stride
            kp_xy = torch.stack((kp_x, kp_y), dim=2).reshape(HW, 10)

            face = torch.empty((HW, 15), device=device, dtype=torch.float32)
            face[:, 0] = x1
            face[:, 1] = y1
            face[:, 2] = w
            face[:, 3] = h
            face[:, 4:14] = kp_xy
            face[:, 14] = score

            all_faces.append(face[keep])

        if not all_faces:
            return None

        faces = torch.cat(all_faces, dim=0) 

        boxes_xywh = faces[:, 0:4].detach().cpu().numpy()  
        scores_np = faces[:, 14].detach().cpu().numpy()    
        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=scores_np,
            score_threshold=float(score_thresh),
            nms_threshold=float(iou_thresh),
            top_k=int(top_k),
        )

        if idxs is None or len(idxs) == 0:
            return None

        keep_idx = np.array(idxs).reshape(-1).astype(np.int64)
        faces = faces[torch.from_numpy(keep_idx).to(device)]

        scale = float(self.scale)
        pad_x = float(self.pad_x)
        pad_y = float(self.pad_y)

        faces[:, 0] = (faces[:, 0] - pad_x) / scale
        faces[:, 1] = (faces[:, 1] - pad_y) / scale
        faces[:, 2] = faces[:, 2] / scale
        faces[:, 3] = faces[:, 3] / scale

        faces[:, 4:14:2] = (faces[:, 4:14:2] - pad_x) / scale  # kp x
        faces[:, 5:14:2] = (faces[:, 5:14:2] - pad_y) / scale  # kp y

        return faces.numpy()

    def detect(self, image: np.ndarray, score_threshold: float = 0.6, iou_threshold: float = 0.3, top_k: int = 5000,) -> Optional[np.ndarray]:
        """
        Run face detection on an image.

        Args:
            image: Input image (BGR recommended if you’re using OpenCV reads).
            score_threshold: Score threshold applied before NMS.
            iou_threshold: IoU threshold for NMS.
            top_k: Max number of boxes kept by NMS.

        Returns:
            (N,15) float32-ish numpy array in original image coordinates, or None if no detections.
        """
        x = self._preprocess_image(image)
        outs = self.session.run(None, {self.inp_name: x})
        return self._post_process(outs, score_threshold, iou_threshold, top_k)
