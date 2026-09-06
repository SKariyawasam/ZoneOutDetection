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
        # 1. Get visual prob and 4D image traits
        v_prob = vis_model.predict()
        image_traits = vis_model.get_image_traits()
        phys_model.update_image_traits(image_traits)
        
        # 2. Get multimodal physio prob (uses rolling buffer updated by watch)
        p_prob = phys_model.predict()
        
        # Extract real-time signal quality (Wei et al., 2018)
        q_vis = vis_model.get_visual_quality()
        q_phys = phys_model.get_physio_quality()
        
        # 3. Fuse and Classify (Dynamic Signal-Quality Adaptive Fusion)
        face_detected = True
        watch_connected = len(watch_clients) > 0
        watch_worn = phys_model.is_sensor_worn() if watch_connected else False
        
        if not watch_connected:
            watch_status = "disconnected"
        elif not watch_worn:
            watch_status = "not_worn"
        else:
            watch_status = "active"

        if v_prob == -1.0:
            face_detected = False
            v_prob = 0.0
            state = "user away"
            fused_score = 0.0
            w_vis = 0.0
            w_phys = 1.0 if watch_connected else 0.0
            # Reset hysteresis state so return from absence requires fresh evidence of distraction (< 0.45)
            fusion_engine.reset_state("focused")
        else:
            is_calib = phys_model.is_calibrating()
            # If watch is connected but off-wrist, gracefully fall back on visual
            active_watch = watch_connected and watch_worn
            state, fused_score, w_vis, w_phys = fusion_engine.classify(
                v_prob, p_prob, is_calib, active_watch, q_vis, q_phys
            )
            
        print(f"Current State: {state} (Fused: {fused_score:.2f}, Vis: {v_prob:.2f} [W: {w_vis:.2f}], Phys: {p_prob:.2f} [W: {w_phys:.2f}], Face: {face_detected}, Watch Status: {watch_status})")
        
        # 3.5 Get true physio metrics
        hr, rmssd = phys_model.get_latest_metrics()
        
        # 4. Trigger Intervention (Local Laptop Alert & Watch Haptics)
        if state != "user away":
            trigger_intervention(state)
            
        # Send Haptic Vibration Trigger to Watch when Zoned Out
        if state == "zoned out" and watch_clients:
            watch_alert = json.dumps({"action": "vibrate", "type": "zone_out_alert"})
            websockets.broadcast(watch_clients, watch_alert)

        # 5. Broadcast to Web UI Dashboard
        if ui_clients:
            payload = json.dumps({
                "visual_prob": v_prob,
                "physio_prob": p_prob,
                "fused_prob": fused_score,
                "w_vis": w_vis,
                "w_phys": w_phys,
                "q_vis": q_vis,
                "q_phys": q_phys,
                "state": state,
                "face_detected": face_detected,
                "watch_connected": watch_connected,
                "watch_worn": watch_worn,
                "watch_status": watch_status,
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
    print("Starting Multimodal Zoning Out Detection System with Live Inference (WSS)...")
    
    # Load SSL context for WSS
    import ssl
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    
    # Bind to 0.0.0.0 to allow incoming connections from the smartwatch on the local network
    server = await websockets.serve(ws_handler, "0.0.0.0", 8765, ssl=ssl_context, ping_interval=None, ping_timeout=None)
    
    from discovery import start_discovery_beacon
    
    # Run the inference loop and the UDP auto-discovery beacon concurrently
    try:
        await asyncio.gather(
            inference_loop(),
            start_discovery_beacon(server_port=8765)
        )
    finally:
        vis_model.release()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("System shutting down.")
        vis_model.release()
