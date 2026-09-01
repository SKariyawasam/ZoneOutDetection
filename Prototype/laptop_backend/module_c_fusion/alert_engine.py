import subprocess
import os
import time
import winsound

last_alert_time = 0.0
ALERT_COOLDOWN = 6.0  # seconds

def trigger_intervention(state):
    """
    Triggers a laptop desktop popup alert + audio cue when state is 'zoned out'.
    Features a 6-second cooldown to avoid continuous popup spamming.
    """
    global last_alert_time
    if state == "zoned out":
        now = time.time()
        if now - last_alert_time < ALERT_COOLDOWN:
            return
        last_alert_time = now
        
        print("[ALERT] Mind-Wandering Detected! Triggering laptop popup & sound...", flush=True)
        
        # Sound disabled per user preference
        pass
            
        # 2. Native Windows Notification Popup (Non-blocking async subprocess)
        try:
            script_path = os.path.join(os.path.dirname(__file__), "show_alert.ps1")
            subprocess.Popen([
                "powershell",
                "-ExecutionPolicy", "Bypass",
                "-WindowStyle", "Hidden",
                "-File", script_path,
                "-Title", "DipSEER Focus Alert",
                "-Message", "Mind-Wandering Detected! Take a deep breath and refocus."
            ], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except Exception as e:
            print(f"Failed to launch desktop popup: {e}")
