package com.fyp.dipseer

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class DipSeerService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var heartRateSensor: Sensor? = null
    private var rotationVectorSensor: Sensor? = null

    private var client: OkHttpClient? = null
    private var webSocket: WebSocket? = null

    private var isRecording = false
    private val handler = Handler(Looper.getMainLooper())

    companion object {
        const val ACTION_UPDATE_STATE = "com.fyp.dipseer.UPDATE_STATE"
        const val EXTRA_STATUS_MESSAGE = "status_message"
        const val EXTRA_IS_RECORDING = "is_recording"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "DipSeerServiceChannel"
    }

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        client = OkHttpClient.Builder()
            .readTimeout(3, TimeUnit.SECONDS)
            .pingInterval(10, TimeUnit.SECONDS)
            .build()
            
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = createNotification("Connecting...")
        startForeground(NOTIFICATION_ID, notification)

        isRecording = true
        broadcastState("Connecting...")
        connectWebSocket()

        heartRateSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        rotationVectorSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }

        return START_STICKY
    }

    private fun connectWebSocket() {
        if (!isRecording) return
        val request = Request.Builder()
            .url("ws://172.17.232.160:8765")
            .build()
        webSocket = client?.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                broadcastState("Connected")
                updateNotification("Connected and streaming")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("DipSEER", "WebSocket Failure: " + t.message, t)
                if (isRecording) {
                    broadcastState("Reconnecting...")
                    updateNotification("Reconnecting...")
                    handler.postDelayed({ connectWebSocket() }, 3000)
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (isRecording) {
                    broadcastState("Reconnecting...")
                    updateNotification("Reconnecting...")
                    handler.postDelayed({ connectWebSocket() }, 3000)
                }
            }
        })
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
        
        try {
            webSocket?.send(jsonObject.toString())
        } catch (e: Exception) {
            Log.e("DipSEER", "Error sending data", e)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onDestroy() {
        handler.removeCallbacksAndMessages(null)
        sensorManager.unregisterListener(this)
        webSocket?.close(1000, "Stopped recording")
        webSocket = null
        isRecording = false
        broadcastState("Ready")
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun broadcastState(statusMessage: String) {
        val intent = Intent(ACTION_UPDATE_STATE).apply {
            putExtra(EXTRA_STATUS_MESSAGE, statusMessage)
            putExtra(EXTRA_IS_RECORDING, isRecording)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(
                CHANNEL_ID,
                "DipSEER Background Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(serviceChannel)
        }
    }

    private fun createNotification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("DipSEER Active")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, createNotification(text))
    }
}
