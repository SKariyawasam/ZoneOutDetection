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

const systemStatus = document.getElementById('system-status');
const watchStatus = document.getElementById('watch-status');
const watchBadge = document.getElementById('watch-badge');

const watchStatusText = document.getElementById('watch-status-text');
const watchStatusIcon = document.getElementById('watch-status-icon');

function updateWatchStatus(isConnected, watchStatusState = 'disconnected', isWorn = false) {
  if (!isConnected || watchStatusState === 'disconnected') {
    watchStatus.className = 'status-pill disconnected';
    watchStatus.innerHTML = '<span class="dot"></span> Watch Offline';
    watchBadge.style.background = 'rgba(239, 68, 68, 0.15)';
    watchBadge.style.color = 'var(--state-zoned)';
    watchBadge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
    watchBadge.innerText = 'Offline';

    if (watchStatusText) {
      watchStatusText.innerText = "Watch Disconnected";
      watchStatusText.style.color = "var(--text-muted)";
    }
    if (watchStatusIcon) {
      watchStatusIcon.innerText = "⌚";
    }
  } else if (watchStatusState === 'not_worn' || !isWorn) {
    watchStatus.className = 'status-pill active';
    watchStatus.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    watchStatus.innerHTML = '<span class="dot" style="background-color: #f59e0b;"></span> Watch Standby';
    watchBadge.style.background = 'rgba(245, 158, 11, 0.15)';
    watchBadge.style.color = '#f59e0b';
    watchBadge.style.borderColor = 'rgba(245, 158, 11, 0.3)';
    watchBadge.innerText = 'Off-Body';

    if (watchStatusText) {
      watchStatusText.innerText = "Watch Connected (Off-Body / Acquiring Pulse)";
      watchStatusText.style.color = "#f59e0b";
    }
    if (watchStatusIcon) {
      watchStatusIcon.innerText = "⚠️";
    }
  } else {
    watchStatus.className = 'status-pill active';
    watchStatus.style.borderColor = '';
    watchStatus.innerHTML = '<span class="dot"></span> Watch Streaming';
    watchBadge.style.background = 'rgba(34, 211, 238, 0.15)';
    watchBadge.style.color = 'var(--accent-cyan)';
    watchBadge.style.borderColor = 'rgba(34, 211, 238, 0.2)';
    watchBadge.innerText = 'Live';

    if (watchStatusText) {
      watchStatusText.innerText = "Tracking Physiological Telemetry (Active)";
      watchStatusText.style.color = "var(--accent-cyan)";
    }
    if (watchStatusIcon) {
      watchStatusIcon.innerText = "❤️";
    }
  }
}

// Initialize Connection to Python Backend
function connectWebSocket() {
  const ws = new WebSocket('wss://localhost:8765');

  ws.onopen = () => {
    console.log('Connected to Multimodal Backend');
    systemStatus.className = 'status-pill active';
    systemStatus.innerHTML = '<span class="dot"></span> System Live';
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
    systemStatus.className = 'status-pill disconnected';
    systemStatus.innerHTML = '<span class="dot"></span> System Offline';
    updateWatchStatus(false);
    setTimeout(connectWebSocket, 3000);
  };
}

const camStatusText = document.getElementById('cam-status-text');
const visWeightBadge = document.getElementById('vis-weight-badge');
const physWeightBadge = document.getElementById('phys-weight-badge');

const dismissAlertBtn = document.getElementById('dismiss-alert-btn');
let alertDismissedByUser = false;

if (dismissAlertBtn) {
  dismissAlertBtn.addEventListener('click', () => {
    alertOverlay.classList.add('hidden');
    alertDismissedByUser = true;
    setTimeout(() => { alertDismissedByUser = false; }, 8000);
  });
}

document.addEventListener('keydown', (e) => {
  if ((e.code === 'Space' || e.code === 'Escape') && !alertOverlay.classList.contains('hidden')) {
    alertOverlay.classList.add('hidden');
    alertDismissedByUser = true;
    setTimeout(() => { alertDismissedByUser = false; }, 8000);
  }
});

function updateDashboard(data) {
  // Update Visual Module
  const vProb = (data.visual_prob * 100).toFixed(1);
  visualBar.style.width = `${vProb}%`;
  visualVal.innerText = data.visual_prob.toFixed(2);

  if (camStatusText) {
    if (data.face_detected) {
      camStatusText.innerText = "Tracking Face (Active)";
      camStatusText.style.color = "var(--accent-cyan)";
    } else {
      camStatusText.innerText = "No Face Detected";
      camStatusText.style.color = "var(--text-muted)";
    }
  }

  // Update Dynamic Adaptive Weights (Wei et al., 2018)
  if ('w_vis' in data && visWeightBadge) {
    visWeightBadge.innerText = `Weight: ${Math.round(data.w_vis * 100)}%`;
  }
  if ('w_phys' in data && physWeightBadge) {
    physWeightBadge.innerText = `Weight: ${Math.round(data.w_phys * 100)}%`;
  }

  // Update Physio Module
  const pProb = (data.physio_prob * 100).toFixed(1);
  physioBar.style.width = `${pProb}%`;
  physioVal.innerText = data.physio_prob.toFixed(2);

  // Update Watch Connection status
  if ('watch_connected' in data) {
    updateWatchStatus(data.watch_connected, data.watch_status, data.watch_worn);
  }

  // Update actual HR/HRV from watch payload
  if (data.hr && data.hrv) {
    hrVal.innerText = `${Math.round(data.hr)} bpm`;
    hrvVal.innerText = `${Math.round(data.hrv)} ms`;
  } else {
    // Fallback if no real data yet
    hrVal.innerText = "-- bpm";
    hrvVal.innerText = "-- ms";
  }

  // Update Fusion State
  if (data.state === "user away") {
    glowOrb.classList.remove('focused', 'zoned');
    glowOrb.classList.add('away');
    finalState.innerText = "USER AWAY";
    alertOverlay.classList.add('hidden');
  } else if (data.state === "zoned out") {
    glowOrb.classList.remove('focused', 'away');
    glowOrb.classList.add('zoned');
    finalState.innerText = "ZONED OUT";
    
    // Show Alert Popup
    if (!alertDismissedByUser) {
      alertOverlay.classList.remove('hidden');
    }
  } else {
    glowOrb.classList.remove('zoned', 'away');
    glowOrb.classList.add('focused');
    finalState.innerText = "FOCUSED";
    alertDismissedByUser = false;
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
