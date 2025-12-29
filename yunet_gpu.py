import cv2
import torch
import torchvision
import numpy as np
import onnxruntime as ort

class YuNet:

    def __init__(self, model_path: str, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.inp_name = self.session.get_inputs()[0].name
        self.sx, self.sy = 0.0, 0.0
        self.grid_cached = {}

    def _resize_to_640(self, img: np.ndarray, size: int = 640, pad_color=(0, 0, 0)):
        h, w = img.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        out = np.full((size, size, img.shape[2] if img.ndim == 3 else 1), pad_color, dtype=img.dtype)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2
        out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y

        return out
    
    def _preprocess_image(self, image: np.ndarray):
        image = self._resize_to_640(image)

        x = image.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)

        return x
    
    def _get_grid(self, rows: int, cols: int, device, dtype):
        key = (rows, cols, device.type, device.index, dtype)
        grid = self.grid_cached.get(key)

        if grid is None:
            r = torch.arange(rows, device=device, dtype=dtype)
            c = torch.arange(cols, device=device, dtype=dtype)
            rr, cc = torch.meshgrid(r, c, indexing="ij")
            grid = torch.stack((rr.reshape(-1), cc.reshape(-1)), dim=1)  # (HW,2) [r,c]
            self.grid_cached[key] = grid

        return grid
    
    def _move_to_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        return torch.as_tensor(x, device=self.device)
    
    @torch.no_grad()
    def _post_process(self, output, score_thresh=0.6, iou_thresh=0.3, inputW=640, inputH=640, strides=(8, 16, 32), divisor=32, device=None, return_numpy=True):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pad_w = ((inputW - 1) // divisor + 1) * divisor
        pad_h = ((inputH - 1) // divisor + 1) * divisor

        all_faces = []

        def to_torch(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            return torch.as_tensor(x, device=device)

        S = len(strides)

        for i, stride in enumerate(strides):
            cols = pad_w // stride
            rows = pad_h // stride
            HW = rows * cols

            cls = to_torch(output[i]).reshape(HW).float()
            obj = to_torch(output[i + S]).reshape(HW).float()
            bbox = to_torch(output[i + 2 * S]).reshape(HW, 4).float()
            kps  = to_torch(output[i + 3 * S]).reshape(HW, 10).float()

            cls.clamp_(0.0, 1.0)
            obj.clamp_(0.0, 1.0)
            score = torch.sqrt(cls * obj)

            keep = score >= score_thresh
            if not torch.any(keep):
                continue

            grid_rc = self._get_grid(rows, cols, device=device, dtype=bbox.dtype)
            r = grid_rc[:, 0]
            c = grid_rc[:, 1]

            # decode bbox
            cx = (c + bbox[:, 0]) * stride
            cy = (r + bbox[:, 1]) * stride
            w  = torch.exp(bbox[:, 2]) * stride
            h  = torch.exp(bbox[:, 3]) * stride
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h

            # decode keypoints
            kp = kps.view(HW, 5, 2)
            kp_x = (kp[..., 0] + c.unsqueeze(1)) * stride
            kp_y = (kp[..., 1] + r.unsqueeze(1)) * stride
            kp_xy = torch.stack((kp_x, kp_y), dim=2).reshape(HW, 10)

            # assemble face (HW,15)
            face = torch.empty((HW, 15), device=device, dtype=torch.float32)
            face[:, 0] = x1
            face[:, 1] = y1
            face[:, 2] = w
            face[:, 3] = h
            face[:, 4:14] = kp_xy
            face[:, 14] = score

            all_faces.append(face[keep])

        if not all_faces:
            return None if return_numpy else torch.empty((0, 15), device=device)

        faces = torch.cat(all_faces, dim=0)  # (N,15)

        # ---- OpenCV NMSBoxes expects [x, y, w, h] + scores on CPU ----
        boxes_xywh = faces[:, 0:4].detach().cpu().numpy()  # (N,4) float
        scores_np  = faces[:, 14].detach().cpu().numpy()   # (N,)  float

        # cv2.dnn.NMSBoxes wants Python lists (most robust across OpenCV builds)
        boxes_list = boxes_xywh.tolist()
        scores_list = scores_np.tolist()

        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes_list,
            scores=scores_list,
            score_threshold=float(score_thresh),   
            nms_threshold=float(iou_thresh),
        )

        if idxs is None or len(idxs) == 0:
            # nothing survived NMS
            return None if return_numpy else torch.empty((0, 15), device=device)

        keep_idx = np.array(idxs).reshape(-1).astype(np.int64)
        keep_idx_t = torch.from_numpy(keep_idx).to(device)

        faces = faces[keep_idx_t]

        # scale back to original space (your existing logic)
        scale, pad_x, pad_y = float(self.scale), float(self.pad_x), float(self.pad_y)

        faces[:, 0] = (faces[:, 0] - pad_x) / scale
        faces[:, 1] = (faces[:, 1] - pad_y) / scale

        # box w,h (no padding to subtract, only scale)
        faces[:, 2] = faces[:, 2] / scale
        faces[:, 3] = faces[:, 3] / scale

        # keypoints: subtract pad then divide by scale
        faces[:, 4:14:2] = (faces[:, 4:14:2] - pad_x) / scale  # kp x
        faces[:, 5:14:2] = (faces[:, 5:14:2] - pad_y) / scale  # kp y

        if return_numpy:
            return faces.detach().cpu().numpy()
        
        return faces
        
    def detect(self, image, score_threshold = 0.6, iou_threshold = 0.3):
        x = self._preprocess_image(image)
        outs = self.session.run(None, {self.inp_name: x})
        return self._post_process(outs, score_threshold, iou_threshold)