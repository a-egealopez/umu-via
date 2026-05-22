# Visión Artificial
### Memoria técnica de la entrega parcial
### Curso 2025–2026

## Presentación

Esta memoria documenta la resolución de los bloques propuestos para la entrega de la asignatura de **Visión Artificial**, incluyendo los cinco ejercicios de la entrega parcial, el ejercicio de rectificación de perspectiva y tres ejercicios opcionales (reconstrucción 3D, realidad aumentada y control gestual). Se incluyen soluciones implementadas, fragmentos de código relevantes, capturas de pantalla del sistema en funcionamiento y un análisis crítico de resultados, limitaciones y casos de fallo conocidos.

| Estudiante | Titulación | Fecha |
|------------|------------|-------|
| Alejandro Egea López | Grado en Informática | Curso 2026–2027 |

---

## Estructura del repositorio

```
proyecto/
├── actividad/
│   ├── actividad.py              # Detector de movimiento + alertas Telegram
│   ├── token.env                 # Credenciales del bot de Telegram
│   └── events/                   # Clips de vídeo generados automáticamente
├── calibracion/
│   ├── grid.py                   # Cuadrícula de medida interactiva
│   └── calib.txt                 # Parámetros intrínsecos de la cámara
├── clasificador/
│   ├── clasificador.py           # Aplicación principal
│   └── methods/
│       ├── mp_embedding.py       # Embedding MobileNet (similitud coseno)
│       ├── hand_procrustes.py    # Gestos de mano (distancia Procrustes)
│       └── sift_matching.py      # Emparejamiento de keypoints SIFT
├── dl/
│   ├── train.py                  # Fine-tuning YOLO11n
│   ├── run.py                    # Inferencia en tiempo real
│   ├── best.pt                   # Pesos del modelo entrenado
│   └── house_objects.yolov8/     # Dataset: book, fruit, toy
├── trafico/
│   └── trafico.py                # Contador bidireccional de vehículos
├── rectificacion/
│   ├── rectificacion.py          # Medición de distancias reales con homografía
│   ├── pick_refs.py              # Selección interactiva de puntos de referencia
│   └── *_refs.txt                # Ficheros de referencias por imagen
├── extra_8_7_2/
│   ├── run.py                    # Punto de entrada del sistema AR+gestual
│   ├── hand_controller.py        # Controlador gestual MediaPipe (4 GDL)
│   ├── ar_viewer.py              # Visor AR wireframe con carga de .obj
│   └── reconstruct_colab.ipynb   # Notebook Colab: COLMAP vs VGGT
├── config.py                     # Configuración centralizada
├── ui.py                         # SliderPanel y helpers de visualización
└── utils.py                      # Utilidades compartidas
```

---

## Resumen de ejercicios

| Ejercicio | Técnica principal | Estado |
|-----------|-------------------|--------|
| [📐 Calibración](calibracion/index.md) | `cv2.calibrateCamera`, proyección 3D, SliderPanel | ✅ Completado |
| [🚗 Tráfico](trafico/index.md) | MOG2, tracker de centroides, cruce de línea virtual | ✅ Completado |
| [🔔 Actividad](actividad/index.md) | MOG2, `face_recognition`, bot Telegram, EventRecorder | ✅ Completado |
| [🏷️ Clasificador](clasificador/index.md) | MediaPipe embedding, distancia Procrustes, SIFT | ✅ Completado |
| [🧠 Deep Learning](dl/index.md) | YOLO11n fine-tuning, dataset Roboflow propio | ✅ Completado |
| [📏 Rectificación](rectificacion/index.md) | Homografía, RANSAC, `pick_refs` + `rectificacion` | ✅ Completado |
| [🎲 Ej. 8 — Reconstrucción 3D](ejercicios_extra/reconstruccion.md) | COLMAP SfM, VGGT Transformer, Poisson mesh | ✅ Completado |
| [🥽 Ej. 7 — Realidad Aumentada](ejercicios_extra/ar.md) | ARViewer wireframe, ancla ratón, suavizado exponencial | ✅ Completado |
| [🖐️ Ej. 2 — Control Gestual](ejercicios_extra/controlador.md) | MediaPipe Hands, 4 GDL, HandState | ✅ Completado |

---

## Infraestructura del entorno

El stack de captura encadena tres piezas: DroidCam expone la cámara del móvil como stream MJPEG sobre HTTP, Tailscale asigna una IP privada estable al dispositivo (`100.81.54.29`) independientemente de la red física, y WSL2 consume ese stream desde Windows sin necesidad de abrir puertos en el router.

```
Android (DroidCam) → Tailscale → WSL2
CAMERA_URL = "http://100.81.54.29:4747/video.mjpg"
```

Para las alertas del módulo de actividad se usa un bot de Telegram (`t.me/ael_via_bot`) creado con BotFather; el `USER_ID` del destinatario se obtiene con `@userinfobot`. El dataset de objetos domésticos (`book`, `fruit`, `toy`) está etiquetado y alojado en Roboflow (`universe.roboflow.com/viaael/house_objects-otfqu`) y se descarga directamente antes del fine-tuning YOLO.