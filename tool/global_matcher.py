import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import collections
from collections import deque

class GlobalMatcher:
    """
    Matcher responsible for cross-video and long-term person identity (Global ID) consistency.
    It maintains a database containing feature vectors of all known persons.
    """
    def __init__(self, threshold=0.75, max_history_per_person=5):
        # Similarity threshold for a successful match
        self.threshold = threshold
        # Maximum number of historical features stored per global ID (used for averaging or matching)
        self.max_history_per_person = max_history_per_person
        # Global feature database: {global_id: deque[feat_vector]}
        self.global_features_db = collections.defaultdict(
            lambda: deque(maxlen=self.max_history_per_person)
        )
        # Counter to track the next available Global ID
        self.next_global_id = 1
        
        # In-memory feature cache: used to accelerate feature comparison across all global IDs
        # Structure: {global_id: averaged_feat_vector}
        self.averaged_features_cache = {} 

    def _update_cache(self, global_id):
        """Calculates and updates the average feature vector for a specific ID in the cache"""
        features = list(self.global_features_db[global_id])
        if features:
            avg_feat = np.mean(features, axis=0)
            self.averaged_features_cache[global_id] = avg_feat
            return avg_feat
        return None

    def get_global_id(self, local_id, current_feature):
        """
        Receives the local ID and feature of the current frame and returns a long-term Global ID.
        If the feature matches an existing Global ID, the database is updated and the existing ID is returned;
        otherwise, a new Global ID is created.
        """
        # Check if the feature dimension is correct (fixes the (1,) error from previous iterations)
        if current_feature.size != 512:
            # If feature dimension is incorrect (typically happens during ReID sampling intervals for new IDs),
            # effective matching cannot be performed. Return a temporary ID or wait for the next ReID frame.
            # We assume GlobalMatcher should primarily receive valid 512D features.
            
            # If it's a zero/null vector, skip matching and return the existing Global ID for this local_id (if any).
            if current_feature.size == 1 and current_feature[0] == 0:
                # Search for an already assigned Global ID
                for gid, feats_deque in self.global_features_db.items():
                    # Note: This check depends on whether your DB structure supports local_id mapping
                    if local_id in feats_deque: 
                        return gid
                return -1 # Represents an invalid match
            
            # For valid 512D features, logic continues below (assertion recommended in production)
            pass

        # 1. Attempt to match against existing Global IDs
        best_match_id = -1
        max_similarity = -1.0
        
        # Retrieve all currently cached average features
        global_ids = list(self.averaged_features_cache.keys())
        if global_ids:
            # Construct feature matrix (N, 512)
            stored_feats = np.array([self.averaged_features_cache[gid] for gid in global_ids])
            
            # Calculate Cosine Similarity: (1, 512) x (N, 512)^T -> (1, N)
            similarities = cosine_similarity(current_feature.reshape(1, -1), stored_feats)
            max_similarity = np.max(similarities)
            best_match_index = np.argmax(similarities)
            best_match_id = global_ids[best_match_index]

        # 2. Determine if the match is successful
        if max_similarity >= self.threshold:
            # Match successful: reuse existing ID
            global_id = best_match_id
            
        else:
            # Match failed: create a new Global ID
            global_id = self.next_global_id
            self.next_global_id += 1

        # 3. Update the feature database and the cache
        self.global_features_db[global_id].append(current_feature)
        self._update_cache(global_id)
        
        return global_id
