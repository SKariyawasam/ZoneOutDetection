import os
import asyncio
import json
import websockets
from module_a_visual.svd_inference import VisualInferenceModel
from module_b_receiver.physio_inference import PhysioInferenceModel
from module_c_fusion.meta_classifier import MetaClassifier
from module_c_fusion.alert_engine import trigger_intervention

# Initialize models
vis_model = VisualInferenceModel()
phys_model = PhysioInferenceModel()
fusion_engine = MetaClassifier()

# Global sets of connected websocket clients
ui_clients = set()
watch_clients = set()

async def inference_loop():
    """Real-time inference loop and broadcasts data to Web UI."""
    while True:
        # 1. Get visual prob (grabs webcam frame)
        v_prob = vis_model.predict()
        
        # 2. Get physio prob (uses rolling buffer updated by watch)
        p_prob = phys_model.predict()
        
        # 3. Fuse and Classify
        face_detected = True
        if v_prob == -1.0:
            face_detected = False
            v_prob = 0.0
            state = "user away"
        else:
            state = fusion_engine.classify(v_prob, p_prob)
            
        print(f"Current State: {state} (Vis: {v_prob:.2f}, Phys: {p_prob:.2f}, Face: {face_detected})")
        
        # 3.5 Get true physio metrics
        hr, rmssd = phys_model.get_latest_metrics()
        
        # 4. Trigger Intervention (Local)
        if state != "user away":
            trigger_intervention(state)

        # 5. Broadcast to Web UI Dashboard
        if ui_clients:
            payload = json.dumps({
                "visual_prob": v_prob,
                "physio_prob": p_prob,
                "state": state,
                "face_detected": face_detected,
                "watch_connected": len(watch_clients) > 0,
                "hr": hr,
                "hrv": rmssd
            })
            # Send to all connected UI clients concurrently
            websockets.broadcast(ui_clients, payload)

        # 1-second inference loop instead of 2 for better responsiveness
        await asyncio.sleep(1) 

async def ws_handler(websocket):
    """Handles incoming websocket connections from both Web UI and Watch."""
    print("New WebSocket connection established.")
    is_watch = False
    ui_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if "type" in data:
                    # Identify and categorize as a watch client if we receive sensor data
                    if not is_watch:
                        is_watch = True
                        if websocket in ui_clients:
                            ui_clients.remove(websocket)
                        watch_clients.add(websocket)
                        print("Connection identified as Watch client.")
                    phys_model.update_buffer(message)
            except json.JSONDecodeError:
                pass
    finally:
        if websocket in ui_clients:
            ui_clients.remove(websocket)
        if websocket in watch_clients:
            watch_clients.remove(websocket)
        print("WebSocket connection closed.")

async def main():
    print("Starting Multimodal Zoning Out Detection System with Live Inference...")
    # Bind to 0.0.0.0 to allow incoming connections from the smartwatch on the local network
    server = await websockets.serve(ws_handler, "0.0.0.0", 8765, ping_interval=None, ping_timeout=None)
    
    # Run the inference loop concurrently
    try:
        await inference_loop()
    finally:
        vis_model.release()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("System shutting down.")
        vis_model.release()
