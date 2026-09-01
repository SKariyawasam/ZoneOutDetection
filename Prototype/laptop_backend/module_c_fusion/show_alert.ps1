param (
    [string]$Title = "DipSEER Focus Alert",
    [string]$Message = "Mind-Wandering Detected! Please refocus."
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Warning
$notify.Visible = $true
$notify.ShowBalloonTip(4000, $Title, $Message, [System.Windows.Forms.ToolTipIcon]::Warning)

# Keep alive briefly so balloon renders on Windows taskbar
Start-Sleep -Milliseconds 1500
$notify.Dispose()
