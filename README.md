# Visión Artificial — UMU 2025–2026

Memoria de prácticas de la asignatura de **Visión Artificial** (Grado en Informática, Universidad de Murcia).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=flat-square&logo=google&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-11n-111F68?style=flat-square&logo=ultralytics&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-latest-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-dataset-A259FF?style=flat-square&logo=roboflow&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-GPU_T4-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram_Bot-alerts-26A5E4?style=flat-square&logo=telegram&logoColor=white)
![DroidCam](https://img.shields.io/badge/DroidCam-MJPEG-4CAF50?style=flat-square&logo=android&logoColor=white)
![Tailscale](https://img.shields.io/badge/Tailscale-VPN-242424?style=flat-square&logo=tailscale&logoColor=white)
![MkDocs](https://img.shields.io/badge/Docs-MkDocs_Material-526CFE?style=flat-square&logo=materialformkdocs&logoColor=white)

## Ejercicios

| # | Ejercicio | Técnica principal | Estado |
|---|-----------|-------------------|--------|
| 1 | 📐 Calibración | `cv2.calibrateCamera`, proyección 3D por rayos, SliderPanel interactivo | ✅ |
| 2 | 🚗 Análisis de Tráfico | MOG2, tracker de centroides con EMA, cruce de línea virtual bidireccional | ✅ |
| 3 | 🔔 Detector de Actividad | MOG2 + `face_recognition` (HOG/CNN), anonimización, bot Telegram, EventRecorder | ✅ |
| 4 | 🏷️ Clasificador | MediaPipe embedding (MobileNet V3, coseno), Procrustes, SIFT + filtro de Lowe | ✅ |
| 5 | 🧠 Deep Learning | Fine-tuning YOLO11n, dataset propio en Roboflow (`book`, `fruit`, `toy`), hilo de inferencia | ✅ |
| 6 | 📏 Rectificación | Homografía + RANSAC, `pick_refs` interactivo, medición de distancias reales en imágenes | ✅ |
| 7 | 🥽 Realidad Aumentada | ARViewer wireframe, ancla ratón + offset gestual, `cv.polylines` vectorizado, suavizado exponencial | ✅ |
| 8 | 🎲 Reconstrucción 3D | COLMAP SfM sparse, VGGT `facebook/vggt-1B`, malla Poisson con Open3D (Colab T4) | ✅ |
| 9 | 🖐️ Control Gestual | MediaPipe Hands, 4 GDL (distancia, yaw, roll, XY palma), `HandState` dataclass | ✅ |

## Documentación

👉 [a-egealopez.github.io/umu-via](https://a-egealopez.github.io/umu-via)