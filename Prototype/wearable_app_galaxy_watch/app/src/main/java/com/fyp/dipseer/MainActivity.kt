package com.fyp.dipseer

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text

class MainActivity : ComponentActivity() {

    private var isRecording by mutableStateOf(false)
    private var statusMessage by mutableStateOf("Ready")

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == DipSeerService.ACTION_UPDATE_STATE) {
                statusMessage = intent.getStringExtra(DipSeerService.EXTRA_STATUS_MESSAGE) ?: "Unknown"
                isRecording = intent.getBooleanExtra(DipSeerService.EXTRA_IS_RECORDING, false)
            }
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val allGranted = permissions.entries.all { it.value }
            if (!allGranted) {
                statusMessage = "Permissions Denied"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        requestPermissionLauncher.launch(
            arrayOf(
                Manifest.permission.BODY_SENSORS,
                Manifest.permission.INTERNET,
                Manifest.permission.POST_NOTIFICATIONS
            )
        )

        LocalBroadcastManager.getInstance(this).registerReceiver(
            statusReceiver,
            IntentFilter(DipSeerService.ACTION_UPDATE_STATE)
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

    override fun onDestroy() {
        super.onDestroy()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(statusReceiver)
    }

    private fun toggleRecording() {
        val serviceIntent = Intent(this, DipSeerService::class.java)
        if (isRecording) {
            stopService(serviceIntent)
        } else {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                startForegroundService(serviceIntent)
            } else {
                startService(serviceIntent)
            }
        }
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
