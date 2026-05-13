import torch
import numpy as np
from torchreid.utils import FeatureExtractor

class ReIDExtractor:
    def __init__(self, device='cuda:0'):
        # Using OSNet-AIN (High performance, robust to illumination changes)
        print("Loading ReID Database Model (OSNet-AIN)...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            device=device,
            verbose=False
        )

    def extract(self, crops, input_color="bgr"):
        """
        Extract features in batches.
        Input: List of numpy images. Default expects BGR.
        Output: (N, 512) normalized numpy array.
        """
        if not crops: return np.empty((0, 512))
        
        # BGR -> RGB conversion without cv2
        if input_color.lower() == "bgr":
            rgb_crops = [c[:, :, ::-1] for c in crops]
        else:
            rgb_crops = crops
        
        # Feature extraction
        features = self.extractor(rgb_crops)
        
        # L2 Normalization (Essential for Cosine Similarity retrieval)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()
