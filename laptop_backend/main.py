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

# Global set of connected websocket clients
connected_clients = set()

async def inference_loop():
    """Simulates the real-time inference loop and broadcasts data to Web UI."""
    while True:
        # 1. Get visual prob
        v_prob = vis_model.predict(None)
        
        # 2. Get physio prob
        p_prob = phys_model.predict(None)
        
        # 3. Fuse and Classify
        state = fusion_engine.classify(v_prob, p_prob)
        print(f"Current State: {state} (Vis: {v_prob:.2f}, Phys: {p_prob:.2f})")
        
        # 4. Trigger Intervention (Local)
        trigger_intervention(state)

        # 5. Broadcast to Web UI Dashboard
        if connected_clients:
            payload = json.dumps({
                "visual_prob": v_prob,
                "physio_prob": p_prob,
                "state": state
            })
            # Send to all clients concurrently
            websockets.broadcast(connected_clients, payload)

        await asyncio.sleep(2) # 2-second inference loop

async def ws_handler(websocket):
    """Handles incoming websocket connections."""
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)

async def main():
    print("Starting Multimodal Zoning Out Detection System with Web UI support...")
    # Start WebSocket Server on port 8765
    server = await websockets.serve(ws_handler, "localhost", 8765)
    
    # Run the inference loop concurrently
    await inference_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("System shutting down.")
