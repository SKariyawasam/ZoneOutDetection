import asyncio
import json
import websockets
from module_a_visual.visual_inference import VisualInferenceModel
from module_b_receiver.physio_inference import PhysioInferenceModel
from module_c_fusion.meta_classifier import MetaClassifier
from module_c_fusion.alert_engine import trigger_intervention

# Initialize models
vis_model = VisualInferenceModel()
phys_model = PhysioInferenceModel()
fusion_engine = MetaClassifier()

# Global set of connected websocket clients (Web UI)
ui_clients = set()

async def inference_loop():
    """Real-time inference loop and broadcasts data to Web UI."""
    while True:
        # 1. Get visual prob (grabs webcam frame)
        v_prob = vis_model.predict()
        
        # 2. Get physio prob (uses rolling buffer updated by watch)
        p_prob = phys_model.predict()
        
        # 3. Fuse and Classify
        state = fusion_engine.classify(v_prob, p_prob)
        print(f"Current State: {state} (Vis: {v_prob:.2f}, Phys: {p_prob:.2f})")
        
        # 4. Trigger Intervention (Local)
        trigger_intervention(state)

        # 5. Broadcast to Web UI Dashboard
        if ui_clients:
            payload = json.dumps({
                "visual_prob": v_prob,
                "physio_prob": p_prob,
                "state": state
            })
            # Send to all connected UI clients concurrently
            websockets.broadcast(ui_clients, payload)

        # 1-second inference loop instead of 2 for better responsiveness
        await asyncio.sleep(1) 

async def ws_handler(websocket):
    """Handles incoming websocket connections from both Web UI and Watch."""
    print("New WebSocket connection established.")
    ui_clients.add(websocket)
    try:
        async for message in websocket:
            # Check if this is sensor data from the watch
            try:
                data = json.loads(message)
                if "type" in data:
                    # It's watch data
                    phys_model.update_buffer(message)
            except json.JSONDecodeError:
                pass
    finally:
        ui_clients.remove(websocket)
        print("WebSocket connection closed.")

async def main():
    print("Starting Multimodal Zoning Out Detection System with Live Inference...")
    # Bind to 0.0.0.0 to allow incoming connections from the smartwatch on the local network
    server = await websockets.serve(ws_handler, "0.0.0.0", 8765)
    
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
