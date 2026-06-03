// WebSocket Client Logic
const visualBar = document.getElementById('visual-bar');
const visualVal = document.getElementById('visual-val');

const physioBar = document.getElementById('physio-bar');
const physioVal = document.getElementById('physio-val');

const hrVal = document.getElementById('hr-val');
const hrvVal = document.getElementById('hrv-val');

const glowOrb = document.getElementById('glow-orb');
const finalState = document.getElementById('final-state');
const alertOverlay = document.getElementById('alert-overlay');

// Initialize Connection to Python Backend
function connectWebSocket() {
  const ws = new WebSocket('ws://localhost:8765');

  ws.onopen = () => {
    console.log('Connected to Multimodal Backend');
    document.querySelector('.status-pill').innerHTML = '<span class="dot"></span> Connected';
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch (e) {
      console.error("Error parsing WebSocket data:", e);
    }
  };

  ws.onclose = () => {
    console.log('Disconnected. Retrying in 3 seconds...');
    document.querySelector('.status-pill').innerHTML = '<span class="dot" style="background:var(--state-zoned)"></span> Disconnected';
    setTimeout(connectWebSocket, 3000);
  };
}

function updateDashboard(data) {
  // Update Visual Module
  const vProb = (data.visual_prob * 100).toFixed(1);
  visualBar.style.width = `${vProb}%`;
  visualVal.innerText = data.visual_prob.toFixed(2);

  // Update Physio Module
  const pProb = (data.physio_prob * 100).toFixed(1);
  physioBar.style.width = `${pProb}%`;
  physioVal.innerText = data.physio_prob.toFixed(2);

  // Update Mock HR/HRV based on state for visual effect
  hrVal.innerText = data.state === "zoned out" ? "65 bpm" : "82 bpm";
  hrvVal.innerText = data.state === "zoned out" ? "45 ms" : "30 ms";

  // Update Fusion State
  if (data.state === "zoned out") {
    glowOrb.classList.remove('focused');
    glowOrb.classList.add('zoned');
    finalState.innerText = "ZONED OUT";
    
    // Show Alert
    alertOverlay.classList.remove('hidden');
  } else {
    glowOrb.classList.remove('zoned');
    glowOrb.classList.add('focused');
    finalState.innerText = "FOCUSED";
    
    // Hide Alert
    alertOverlay.classList.add('hidden');
  }
}

// Simulated data for testing when Python server is not running
let simMode = false;
if (simMode) {
  setInterval(() => {
    const isZonedOut = Math.random() > 0.7;
    updateDashboard({
      visual_prob: isZonedOut ? 0.3 + Math.random()*0.1 : 0.7 + Math.random()*0.2,
      physio_prob: isZonedOut ? 0.2 + Math.random()*0.2 : 0.6 + Math.random()*0.3,
      state: isZonedOut ? "zoned out" : "focused"
    });
  }, 2000);
} else {
  // Connect to actual python backend
  connectWebSocket();
}
