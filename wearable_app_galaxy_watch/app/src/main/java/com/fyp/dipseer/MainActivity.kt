package com.fyp.dipseer

import android.Manifest
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var heartRateSensor: Sensor? = null
    private var rotationVectorSensor: Sensor? = null

    private var client: OkHttpClient? = null
    private var webSocket: WebSocket? = null

    private var isRecording by mutableStateOf(false)
    private var statusMessage by mutableStateOf("Ready")

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val allGranted = permissions.entries.all { it.value }
            if (!allGranted) {
                statusMessage = "Permissions Denied"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        client = OkHttpClient.Builder()
            .readTimeout(3, TimeUnit.SECONDS)
            .build()

        requestPermissionLauncher.launch(
            arrayOf(
                Manifest.permission.BODY_SENSORS,
                Manifest.permission.INTERNET
            )
        )

        setContent {
            MaterialTheme {
                WearApp(
                    isRecording = isRecording,
                    statusMessage = statusMessage,
                    onToggleRecording = { toggleRecording() }
                )
            }
        }
    }

    private fun toggleRecording() {
        if (isRecording) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        // Connect WebSocket (replace with actual IP of the laptop backend)
        val request = Request.Builder().url("ws://192.168.1.100:8765").build()
        webSocket = client?.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                statusMessage = "Connected"
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                statusMessage = "Connection Failed"
            }
        })

        heartRateSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        rotationVectorSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }

        isRecording = true
    }

    private fun stopRecording() {
        sensorManager.unregisterListener(this)
        webSocket?.close(1000, "Stopped recording")
        webSocket = null
        isRecording = false
        statusMessage = "Ready"
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || webSocket == null) return

        val jsonObject = JSONObject()
        jsonObject.put("timestamp", System.currentTimeMillis())
        
        when (event.sensor.type) {
            Sensor.TYPE_HEART_RATE -> {
                jsonObject.put("type", "heart_rate")
                jsonObject.put("value", event.values[0])
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                jsonObject.put("type", "rotation_vector")
                jsonObject.put("x", event.values[0])
                jsonObject.put("y", event.values[1])
                jsonObject.put("z", event.values[2])
                jsonObject.put("w", event.values[3])
            }
        }
        
        // Send data asynchronously
        try {
            webSocket?.send(jsonObject.toString())
        } catch (e: Exception) {
            Log.e("DipSEER", "Error sending data", e)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    
    override fun onDestroy() {
        super.onDestroy()
        stopRecording()
    }
}

@Composable
fun WearApp(isRecording: Boolean, statusMessage: String, onToggleRecording: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "DipSEER", style = MaterialTheme.typography.title1)
        Spacer(modifier = Modifier.height(8.dp))
        Text(text = statusMessage, style = MaterialTheme.typography.body2)
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = onToggleRecording) {
            Text(if (isRecording) "Stop" else "Start")
        }
    }
}
