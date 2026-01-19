# 🎥 Jetson TX2 Optical Flow Distance Estimation
Реальное время определение расстояния до ArUco-меток методом TV-L1 Optical Flow (CUDA) на Nvidia Jetson TX2.  
Двойная валидация: Flow → Depth Map + ArUco → 6DoF позиция.   
Бакалаврская работа по компьютерному зрению.

## 🎯 Алгоритм
    Видео (/dev/video0)   
        ↓ [CUDA TV-L1 Optical Flow]  
    Flow → Magnitude → Depth = f/disp × 0.1  
        ↓ [ArUco DICT_7X7_1000]  
    Pose → tvec → Distance  
        ↓ [Валидация]  
    "Dist 42: 2.374m OF: 2.41m"  

**Ключевые формулы:**
```math
\text{Depth} = \frac{0.1 \cdot f}{\text{Displacement}}, \quad f = \frac{K_{00} + K_{11}}{2}
```

## 📋 Файлы репозитория

| Файл       | Назначение                                              |
| ---------- | ------------------------------------------------------- |
| main.c   | Optical Flow + ArUco + GUI (два окна: original/depth)   |
| calibration.c | Калибровка камеры  |

## 🛠️ Технологии и платформа
  💾 Nvidia Jetson TX2 (Pascal GPU 256 CUDA cores, L4T 32.7.2+)  
  📷 USB/CSI камера (/dev/video0, 640×480, 15-25 FPS)  
  🧠 OpenCV 4.x CUDA modules:  
  ├── cudaoptflow (TV-L1 Optical Flow, alpha=0.25, nscales=4)  
  ├── cudaarithm/cudaimgproc (GPU magnitude/resizing)  
  ├── aruco (DICT_7X7_1000, маркеры 15cm)  
  └── GpuMat (zero-copy GPU↔CPU transfer)  

## 📚 Ссылки и документация

    OpenCV CUDA Optical Flow
    Jetson TX2 OpenCV CUDA Setup
    ArUco Pose Estimation

