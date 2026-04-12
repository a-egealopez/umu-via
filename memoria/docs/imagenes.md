# Guía de imágenes — Qué subir y dónde

Esta página lista las **42 imágenes** que hay que reemplazar en la memoria.  
Cada imagen es actualmente un placeholder gris. Sustitúyela con tu captura real manteniendo **exactamente el mismo nombre de fichero**.

---

## Ruta base

Todas las imágenes van dentro de:

```
memoria/docs/assets/images/
```

---

## 📐 Calibración — `calibracion/`

| Fichero | Qué debe mostrar |
|---------|-----------------|
| `chessboard_detection.png` | `cv2.drawChessboardCorners` sobre una imagen de calibración |
| `undistort_comparison.png` | Imagen original (con distorsión) vs imagen corregida lado a lado |
| `grid_z_grande.png` | Pantalla del programa con Z demasiado alta (cuadrícula no alineada) |
| `grid_z_correcta.png` | Pantalla del programa con Z ajustada correctamente al suelo |
| `grid_measurement.png` | Dos clics en la imagen con la distancia calculada visible |
| `sliders_panel.png` | Ventana del SliderPanel con los tres sliders Z / A / X |

---

## 🚗 Tráfico — `trafico/`

| Fichero | Qué debe mostrar |
|---------|-----------------|
| `trafico_setup_roi.png` | ROI poligonal de 4 puntos dibujada sobre la carretera |
| `trafico_setup_split.png` | Línea divisoria de sentidos (cian) configurada |
| `trafico_setup_line.png` | Línea de conteo vertical (amarilla) colocada |
| `trafico_running.png` | Sistema completo en funcionamiento con HUD y contadores |
| `trafico_mask.png` | Ventana `mask` con la máscara binaria de vehículos |
| `trafico_tracks.png` | Bounding boxes verdes + centroides amarillos con IDs |
| `trafico_graph_hourly.png` | Gráfica matplotlib: vehículos detectados por hora |
| `trafico_graph_direction.png` | Gráfica matplotlib: flujo izq vs der en el tiempo |

---

## 🔔 Actividad — `actividad/`

| Fichero | Qué debe mostrar |
|---------|-----------------|
| `actividad_roi_setup.png` | Dibujando la ROI con dos clics del ratón |
| `actividad_persona_detected.png` | Bbox rojo sobre una persona detectada |
| `actividad_objeto_detected.png` | Bbox azul sobre un objeto detectado |
| `actividad_face_anonymized.png` | Cara censurada con bloque negro |
| `actividad_recording.png` | HUD con indicador de grabación activo |
| `actividad_telegram_photo.png` | Captura de pantalla del móvil con la notificación recibida |
| `actividad_events_folder.png` | Explorador de archivos mostrando los `.avi` en `events/` |
| `actividad_mask_mog2.png` | Máscara del foreground dentro del ROI |

---

## 🏷️ Clasificador — `clasificador/`

| Fichero | Qué debe mostrar |
|---------|-----------------|
| `clasificador_embedding_book.png` | Método embedding reconociendo un libro |
| `clasificador_embedding_office.png` | Método embedding reconociendo objeto de oficina |
| `clasificador_hands_rock.png` | Gesto rock reconocido con Procrustes |
| `clasificador_hands_paz.png` | Gesto paz reconocido con Procrustes |
| `clasificador_hands_ok.png` | Gesto ok reconocido con Procrustes |
| `clasificador_sift_libro.png` | SIFT reconociendo portada de libro |
| `clasificador_sift_keypoints.png` | Keypoints SIFT detectados en el frame |
| `clasificador_confidence_bar.png` | Barra de similitud con todos los modelos |
| `clasificador_add_model.png` | Capturando un nuevo modelo con tecla `S` |

---

## 🧠 Deep Learning — `dl/`

| Fichero | Qué debe mostrar |
|---------|-----------------|
| `dataset_samples.png` | Mosaico con ejemplos de las 3 clases del dataset |
| `dataset_labels_book.png` | Imágenes de entrenamiento con bbox de clase `book` |
| `dataset_labels_fruit.png` | Imágenes de entrenamiento con bbox de clase `fruit` |
| `dataset_labels_toy.png` | Imágenes de entrenamiento con bbox de clase `toy` |
| `training_curves.png` | Gráficas de `runs/detect/train/results.png` |
| `training_confusion_matrix.png` | Matriz de confusión de `runs/detect/train/confusion_matrix_normalized.png` |
| `training_pr_curve.png` | Curva PR de `runs/detect/train/PR_curve.png` |
| `yolo_inference_book.png` | Inferencia en tiempo real detectando un libro |
| `yolo_inference_fruit.png` | Inferencia en tiempo real detectando una fruta |
| `yolo_inference_toy.png` | Inferencia en tiempo real detectando un juguete |
| `yolo_inference_multi.png` | Inferencia con varios objetos a la vez |

---

!!! tip "Consejos"
    - **Formato**: PNG o JPG, cualquier resolución (se escalan automáticamente al 100% del ancho).
    - **Tamaño**: intenta que no superen los 500 KB para que el HTML final cargue rápido.
    - Para las gráficas de entrenamiento YOLO, Ultralytics las genera automáticamente en `runs/detect/train/` al acabar `train.py`. Solo tendrás que copiarlas con el nombre indicado.
    - Las curvas matplotlib (`trafico_graph_*.png`) las puedes generar con `plt.savefig("nombre.png")` al final de la sesión de tráfico.
