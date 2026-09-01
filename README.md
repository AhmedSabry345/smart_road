

# 🚦 smart_road

> **Turning a traffic video into a small AI-powered monitoring system.**

This project uses **YOLO + ByteTrack + OpenCV** to detect vehicles, track them, estimate their speed, and flag speed-limit violations — all in real time.

### 🔥 What it does

* 🚗 Detects vehicles with **YOLO**
* 🎯 Tracks multiple vehicles with **ByteTrack**
* 🆔 Gives each vehicle a unique track ID
* ⚡ Estimates vehicle speed
* 🚨 Detects speeding vehicles
* 📊 Analyzes the current traffic condition
* 🖥️ Displays everything through a real-time HUD

### 🧠 Pipeline

```text
🎥 Video
   ↓
🔍 YOLO Detection
   ↓
🎯 ByteTrack
   ↓
📍 Line Crossing
   ↓
⚡ Speed Estimation
   ↓
🚨 Violation Detection
   ↓
🖥️ Traffic HUD
```

### ⚙️ Tech Stack

`Python` • `YOLO` • `ByteTrack` • `OpenCV` • `NumPy` • `PyTorch`

### 🚀 Run it

```bash
https://github.com/AhmedSabry345/smart_road
cd smart_road

pip install -r requirements.txt
python main.py
```

Set your video and model paths in `main.py` before running.

The processed video will be saved as:

```text
output.mp4
```

### 🎮 Controls

`I` → Toggle measurement lines
`R` → Toggle HUD
`Q` → Quit

### 📌 Note

Speed estimation depends on **camera position and road calibration**, so the distance/line values need to be adjusted for different videos.

### 🔮 What's next?

Better perspective correction, automatic calibration, lane detection, improved tracking, and more traffic analysis.

---

**Built as a practical Computer Vision project — not just another YOLO training experiment.** 🚀
