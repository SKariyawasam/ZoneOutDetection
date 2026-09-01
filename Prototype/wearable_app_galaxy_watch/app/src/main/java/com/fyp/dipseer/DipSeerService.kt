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

    private var lastKnownIp = "10.2.4.71"
    private var lastKnownPort = 8765
    private val fallbackIps = listOf("10.2.4.71", "10.192.200.160", "10.122.92.57", "10.8.1.21")
    private var fallbackIndex = 0
    private var multicastLock: android.net.wifi.WifiManager.MulticastLock? = null
    private var wifiLock: android.net.wifi.WifiManager.WifiLock? = null
    private var vibrator: android.os.Vibrator? = null
    private var lastVibrationTime = 0L

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        // Initialize Vibrator for Haptic Intervention Alerts
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? android.os.VibratorManager
            vibratorManager?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as? android.os.Vibrator
        }

        // Acquire locks to prevent WearOS from sleeping Wi-Fi and UDP multicast
        try {
            val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as? android.net.wifi.WifiManager
            multicastLock = wifi?.createMulticastLock("DipSeerMulticastLock")?.apply {
                setReferenceCounted(false)
                acquire()
            }
            wifiLock = wifi?.createWifiLock(android.net.wifi.WifiManager.WIFI_MODE_FULL_HIGH_PERF, "DipSeerWifiLock")?.apply {
                setReferenceCounted(false)
                acquire()
            }
        } catch (e: Exception) {
            Log.w("DipSEER", "Could not acquire Wifi/MulticastLock: ${e.message}")
        }

        // Create a trust manager that does not validate certificate chains for local self-signed WSS
        val trustAllCerts = arrayOf<javax.net.ssl.TrustManager>(object : javax.net.ssl.X509TrustManager {
            override fun checkClientTrusted(chain: Array<out java.security.cert.X509Certificate>?, authType: String?) {}
            override fun checkServerTrusted(chain: Array<out java.security.cert.X509Certificate>?, authType: String?) {}
            override fun getAcceptedIssuers(): Array<java.security.cert.X509Certificate> = arrayOf()
        })
        val sslContext = javax.net.ssl.SSLContext.getInstance("SSL")
        sslContext.init(null, trustAllCerts, java.security.SecureRandom())
        val sslSocketFactory = sslContext.socketFactory

        client = OkHttpClient.Builder()
            .sslSocketFactory(sslSocketFactory, trustAllCerts[0] as javax.net.ssl.X509TrustManager)
            .hostnameVerifier { _, _ -> true }
            .readTimeout(5, TimeUnit.SECONDS)
            .pingInterval(5, TimeUnit.SECONDS)
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

        webSocket?.cancel()
        webSocket = null

        // Launch background UDP discovery to automatically detect laptop IP if it changed
        Thread {
            var discoveredIp = lastKnownIp
            var discoveredPort = lastKnownPort
            var socket: java.net.DatagramSocket? = null

            try {
                socket = java.net.DatagramSocket(null).apply {
                    reuseAddress = true
                    bind(java.net.InetSocketAddress(8766))
                    soTimeout = 2000 // 2 seconds timeout
                    broadcast = true
                }
                val buffer = ByteArray(256)
                val packet = java.net.DatagramPacket(buffer, buffer.size)
                
                Log.d("DipSEER", "Listening for UDP server beacon on port 8766...")
                socket.receive(packet)
                val message = String(packet.data, 0, packet.length).trim()
                Log.d("DipSEER", "Received UDP beacon: $message")

                if (message.startsWith("DIPSEER_SERVER:")) {
                    val parts = message.removePrefix("DIPSEER_SERVER:").split(":")
                    if (parts.isNotEmpty()) {
                        discoveredIp = parts[0]
                        if (parts.size > 1) {
                            discoveredPort = parts[1].toIntOrNull() ?: lastKnownPort
                        }
                        lastKnownIp = discoveredIp
                        lastKnownPort = discoveredPort
                        getSharedPreferences("DipSeerPrefs", Context.MODE_PRIVATE)
                            .edit()
                            .putString("last_ip", discoveredIp)
                            .putInt("last_port", discoveredPort)
                            .apply()
                        Log.i("DipSEER", "Auto-discovered and saved Server at wss://$discoveredIp:$discoveredPort")
                    }
                }
            } catch (e: Exception) {
                val prefs = getSharedPreferences("DipSeerPrefs", Context.MODE_PRIVATE)
                val savedIp = prefs.getString("last_ip", null)
                if (!savedIp.isNullOrEmpty()) {
                    discoveredIp = savedIp
                } else {
                    discoveredIp = fallbackIps[fallbackIndex % fallbackIps.size]
                    fallbackIndex++
                }
                lastKnownIp = discoveredIp
                Log.w("DipSEER", "UDP Discovery failed (${e.message}), using fallback: wss://$discoveredIp:$discoveredPort")
            } finally {
                socket?.close()
            }

            // Connect using the discovered/fallback IP on main thread
            handler.post {
                if (!isRecording) return@post
                val targetUrl = "wss://$discoveredIp:$discoveredPort"
                Log.i("DipSEER", "Connecting to WebSocket at $targetUrl")
                
                val request = Request.Builder().url(targetUrl).build()
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

                    override fun onMessage(webSocket: WebSocket, text: String) {
                        try {
                            val json = JSONObject(text)
                            if (json.optString("action") == "vibrate" || json.optString("type") == "zone_out_alert") {
                                val now = System.currentTimeMillis()
                                if (now - lastVibrationTime > 4000) { // 4-second cooldown to prevent excessive buzzing
                                    lastVibrationTime = now
                                    triggerHapticAlert()
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("DipSEER", "Error processing message: ${e.message}")
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
        }.start()
    }

    private fun triggerHapticAlert() {
        handler.post {
            try {
                Log.i("DipSEER", "Triggering Haptic Vibration: Mind Wandering Detected!")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    // Two distinct attention-grabbing haptic pulses: [delay, vibrate, pause, vibrate]
                    val timings = longArrayOf(0, 300, 150, 300)
                    val amplitudes = intArrayOf(0, 255, 0, 255)
                    val effect = android.os.VibrationEffect.createWaveform(timings, amplitudes, -1)
                    vibrator?.vibrate(effect)
                } else {
                    @Suppress("DEPRECATION")
                    vibrator?.vibrate(500)
                }
                updateNotification("Attention Required! Refocusing needed.")
            } catch (e: Exception) {
                Log.e("DipSEER", "Failed to execute vibration: ${e.message}")
            }
        }
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
        try {
            if (multicastLock?.isHeld == true) multicastLock?.release()
            if (wifiLock?.isHeld == true) wifiLock?.release()
        } catch (e: Exception) {}
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
