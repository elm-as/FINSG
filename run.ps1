$ErrorActionPreference = "Stop"
$PythonExe = "C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe"
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }
& $PythonExe main.py @args
