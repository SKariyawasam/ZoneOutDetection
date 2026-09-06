import numpy as np

class MetaClassifier:
    def __init__(self):
        """
        Decision-Level Dynamic Adaptive Meta-Classifier.
        Implements Dynamic Signal-Quality Weighted Fusion (Wei et al., 2018)
        with stateful dual-threshold hysteresis to prevent boundary chatter.
        """
        self.current_state = "focused"
        print("Initialized MetaClassifier (Dynamic Adaptive Weighted Fusion with Dual-Threshold Hysteresis).")

    def reset_state(self, new_state="focused"):
        """Resets hysteresis state upon user departure or manual re-focus."""
        self.current_state = new_state

    def classify(self, visual_prob, physio_prob, is_calibrating=False, watch_connected=True, q_vis=1.0, q_phys=1.0):
        """
        Takes probability outputs from Module A and Module B alongside real-time quality indices.
        Computes dynamic reliability weights:
        W_phys = (0.50 * Q_phys) / (0.50 * Q_phys + 0.50 * Q_vis)
        W_vis = 1.0 - W_phys
        
        Applies dual-threshold hysteresis:
        - Trigger 'zoned out': fused_score < 0.45
        - Release back to 'focused': fused_score > 0.55
        - Deadband [0.45, 0.55]: preserves previous state, eliminating boundary chatter.
        
        Returns:
            state (str): 'focused' or 'zoned out'
            fused_score (float): integrated focus score [0.0 - 1.0]
            w_vis (float): dynamic visual weight [0.0 - 1.0]
            w_phys (float): dynamic physio weight [0.0 - 1.0]
        """
        if not watch_connected:
            # Watch offline: 100% visual fallback
            w_vis = 1.0
            w_phys = 0.0
            fused_score = float(visual_prob)
        elif is_calibrating:
            # Baseline calibration (0-30s): 90% visual / 10% physio
            w_vis = 0.90
            w_phys = 0.10
            fused_score = float((visual_prob * w_vis) + (physio_prob * w_phys))
        else:
            # Dynamic Signal-Quality Adaptive Fusion (Wei et al., 2018)
            # Symmetric baseline prior: 0.50 / 0.50 (quality factors Q_phys and Q_vis carry the full adaptation)
            base_phys = 0.50
            base_vis = 0.50
            
            raw_phys_weight = base_phys * max(0.1, q_phys)
            raw_vis_weight = base_vis * max(0.1, q_vis)
            
            total_weight = raw_phys_weight + raw_vis_weight + 1e-6
            w_phys = float(raw_phys_weight / total_weight)
            w_vis = float(raw_vis_weight / total_weight)
            
            fused_score = float((visual_prob * w_vis) + (physio_prob * w_phys))

        # Stateful Dual-Threshold Hysteresis
        if self.current_state == "focused":
            if fused_score < 0.45:
                self.current_state = "zoned out"
        else: # current_state == "zoned out"
            if fused_score > 0.55:
                self.current_state = "focused"

        return self.current_state, fused_score, w_vis, w_phys
