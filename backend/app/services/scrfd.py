import numpy as np
import cv2

def distance2bbox(points, distance, max_shapes=None):
    """
    Decode distance prediction to bounding box.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shapes is not None:
        x1 = x1.clamp(min=0, max=max_shapes[1])
        y1 = y1.clamp(min=0, max=max_shapes[0])
        x2 = x2.clamp(min=0, max=max_shapes[1])
        y2 = y2.clamp(min=0, max=max_shapes[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shapes=None):
    """
    Decode distance prediction to keypoints.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shapes is not None:
            px = px.clamp(min=0, max=max_shapes[1])
            py = py.clamp(min=0, max=max_shapes[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.input_shape = [640, 640] # Standard SCRFD input
        self.output_names = [x.name for x in session.get_outputs()]
        self.center_cache = {}
        self.nms_thresh = 0.4

    def _get_anchors(self, feat_stride, height, width):
        key = (feat_stride, height, width)
        if key in self.center_cache:
            return self.center_cache[key]
        
        anchor_centers = []
        for i in range(height):
            for j in range(width):
                # 2 anchors per pixel for SCRFD
                anchor_centers.append([j, i])
                anchor_centers.append([j, i])
        
        anchor_centers = np.array(anchor_centers, dtype=np.float32) * feat_stride
        self.center_cache[key] = anchor_centers
        return anchor_centers

    def detect(self, img, thresh=0.5):
        h, w = img.shape[:2]
        # Resize to 640x640 for consistency
        blob = cv2.resize(img, (640, 640))
        blob = blob.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose((2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        outs = self.session.run(self.output_names, {self.input_name: blob})
        
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        # SCRFD has 3 strides: 8, 16, 32
        for i, stride in enumerate([8, 16, 32]):
            score = outs[i]
            bbox = outs[i+3] * stride
            kps = outs[i+6] * stride
            
            # Anchor generation
            anchor_centers = self._get_anchors(stride, 640//stride, 640//stride)
            
            # Decode
            pos_inds = np.where(score > thresh)[0]
            if len(pos_inds) > 0:
                bboxes = distance2bbox(anchor_centers[pos_inds], bbox[pos_inds])
                kpss = distance2kps(anchor_centers[pos_inds], kps[pos_inds])
                
                scores_list.append(score[pos_inds])
                bboxes_list.append(bboxes)
                kpss_list.append(kpss)

        if not scores_list:
            return None, None, None

        scores = np.vstack(scores_list).flatten()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        
        # Rescale to original size
        sx = w / 640.0
        sy = h / 640.0
        bboxes[:, [0, 2]] *= sx
        bboxes[:, [1, 3]] *= sy
        
        # Reshape kpss from (N, 10) to (N, 5, 2) for easier scaling
        kpss = kpss.reshape((kpss.shape[0], 5, 2))
        kpss[:, :, 0] *= sx
        kpss[:, :, 1] *= sy
        
        # NMS
        keep = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), thresh, self.nms_thresh)
        if len(keep) > 0:
            if isinstance(keep, tuple): keep = keep[0] # Handle different CV2 versions
            keep = keep.flatten()
            return bboxes[keep], scores[keep], kpss[keep]
        
        return None, None, None
