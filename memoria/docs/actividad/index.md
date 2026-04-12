# Detector de Actividad

## Descripción

`actividad/actividad.py` detecta movimiento en una ROI manual, clasifica objetos detectados (persona / objeto), **anonimiza caras** y envía notificaciones con foto y clip de vídeo vía **bot de Telegram**.

---

## Requisitos y ejecución { #requisitos }

!!! info "Entorno"
    Python 3.10+, OpenCV 4.9, NumPy 1.26, `face_recognition`, `python-telegram-bot`.

!!! warning "Credenciales necesarias"
    Antes de ejecutar, crea el fichero `token.env` en la raíz del proyecto con el token de tu bot de Telegram:
    ```
    TELEGRAM_TOKEN=<tu_token>
    TELEGRAM_CHAT_ID=<tu_chat_id>
    ```
    Si el fichero no existe, el programa lanza una excepción al arrancar (fallo rápido e intencionado).

```bash
python actividad/actividad.py
```

!!! tip "Configuración de la ROI"
    Al arrancar, haz **dos clics** sobre la imagen para definir la región de interés rectangular. Solo se procesará el movimiento dentro de esa región.

---

## Arquitectura { #arquitectura }

```mermaid
flowchart TD
    A[autoStream] --> B[cv.resize 640×480]
    B --> C{¿Frame par?}
    C -- Sí --> D[Reutilizar frame\nanonimizado cacheado]
    C -- No --> E[anonymize\nhilo face_recognition]
    E --> F[MOG2 + morfología]
    F --> G{¿ROI activa?}
    G -- Sí --> H[classify_motion\npersona / objeto]
    H --> I{¿Categoría\nde interés?}
    I -- Sí --> J[EventRecorder.start\nfoto Telegram]
    I -- No --> K[Dibujar HUD]
    J --> L[Grabar pre+post buffer]
    L --> M{¿Evento\nacabado?}
    M -- Sí --> N[save_frames_to_video\nevents/*.avi]
    M -- No --> K
    N --> K
```

<figure markdown>
  ![Setup ROI](actividad_roi_setup.png)
  <figcaption>Definición de la ROI rectangular mediante dos clics.</figcaption>
</figure>

---

## Parámetros clave { #parametros }

### Clasificación de movimiento

| Criterio | Umbral | Justificación |
|----------|--------|---------------|
| `ratio = h/w` | > 1.35 | Las personas son más altas que anchas |
| `h` | > 60 px | Descarta detecciones diminutas |
| `fill = área/(w×h)` | < 0.80 | Las personas son menos compactas que cajas |

!!! tip "Parámetro más sensible: ratio h/w"
    Una persona agachada o sentada puede tener `ratio < 1.35` y clasificarse como objeto. Para vigilancia estática donde las personas pasan de pie, los umbrales actuales funcionan razonablemente bien.

### Anonimización de caras

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `FACE_UPDATE_INTERVAL` | 100 frames | Frecuencia de actualización de la detección |
| `FACE_SCALE` | 0.25 | Resolución de inferencia (1/4 del frame original) |
| Margen del bloque | 35% | Ampliación del bbox antes de censurar |

!!! tip "HOG vs CNN para detección de caras"
    HOG es más rápido (~80 ms) pero falla con caras de perfil o parcialmente tapadas. CNN detecta mejor en condiciones difíciles pero triplica el tiempo de llamada (~300 ms). En exteriores o con cámaras en ángulo, considera cambiar a `model="cnn"`.

### Grabación de eventos

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `pre_sec` | 2 s | Duración del pre-buffer antes del evento |
| Post-buffer | 1 s tras último movimiento | Tiempo de cola al cerrar el evento |

---

## Código clave { #codigo }

### Detección y clasificación

<div class="img-grid-2">
<figure markdown>
  ![Detección persona](actividad_persona_detected.png)
  <figcaption>Persona detectada dentro de la ROI (bbox rojo).</figcaption>
</figure>
<figure markdown>
  ![Detección objeto](actividad_objeto_detected.png)
  <figcaption>Objeto detectado dentro de la ROI (bbox azul).</figcaption>
</figure>
</div>

<figure markdown>
  ![Máscara MOG2](actividad_mask_mog2.png)
  <figcaption>Máscara de foreground del MOG2 aplicada exclusivamente dentro del ROI.</figcaption>
</figure>

```python title="actividad/actividad.py — classify_motion()" linenums="1"
def classify_motion(frame, mask_roi, min_area=MIN_AREA_MOV):
    results = []
    for cnt in cv.findContours(mask_roi, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)[0]:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        ratio = h / max(w, 1)          # relación de aspecto
        fill  = area / max(w * h, 1)   # compacidad
        cat = "persona" if ratio > 1.35 and h > 60 and fill < 0.8 else "objeto"
        results.append((cat, x, y, w, h))
    return results
```

### Anonimización de caras

<figure markdown>
  ![Cara anonimizada](actividad_face_anonymized.png)
  <figcaption>Cara censurada con bloque negro con margen del 35%. La detección se ejecuta en hilo separado cada 100 frames.</figcaption>
</figure>

```python title="actividad/actividad.py — anonymize()" linenums="1"
FACE_UPDATE_INTERVAL = 100
FACE_SCALE = 0.25   # trabaja a 1/4 de resolución

def detect_faces(frame, model):
    inv   = 1 / FACE_SCALE
    small = cv.resize(frame, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
    locations = face_recognition.face_locations(
        cv.cvtColor(small, cv.COLOR_BGR2RGB), model=model
    )
    return [(int(t*inv), int(r*inv), int(b*inv), int(l*inv))
            for t, r, b, l in locations]

def anonymize(frame, face_state, face_queue):
    if face_state["frame_count"] % FACE_UPDATE_INTERVAL == 0 \
            and not face_queue.full():
        face_queue.put((frame.copy(), face_state["detector"]))
    face_state["frame_count"] += 1
    out = frame.copy()
    h, w = frame.shape[:2]
    with face_state["lock"]:
        faces = list(face_state["last_faces"])
    for t, r, b, l in faces:
        dx, dy = int((r - l) * 0.35), int((b - t) * 0.35)
        out[max(0,t-dy):min(h,b+dy), max(0,l-dx):min(w,r+dx)] = 0
    return out
```

### Grabación y alertas

<div class="img-grid-2">
<figure markdown>
  ![Grabación activa](actividad_recording.png)
  <figcaption>Indicador de grabación activo en el HUD.</figcaption>
</figure>
<figure markdown>
  ![Notificación Telegram](actividad_telegram_photo.png)
  <figcaption>Notificación recibida en Telegram con la captura del evento y la categoría detectada.</figcaption>
</figure>
</div>

<figure markdown>
  ![Carpeta events](actividad_events_folder.png)
  <figcaption>Carpeta <code>events/</code> con los clips guardados automáticamente. El nombre incluye la marca de tiempo.</figcaption>
</figure>

---

## Decisiones de diseño { #decisiones }

### Clasificación heurística frente a detector neural

En lugar de HOG+SVM o YOLO, los contornos del foreground se clasifican por tres criterios geométricos (→ ver [`classify_motion()`](#codigo), línea 1). Es una apuesta deliberada por velocidad y simplicidad — un detector neural añadiría cientos de milisegundos en CPU. El precio es conocido: personas agachadas o sentadas se clasifican como objeto.

### Detección de caras en hilo separado con caché de 100 frames

`face_recognition` tarda ~80 ms (HOG) o ~300 ms (CNN), demasiado para cada frame. La solución es un `face_worker` dedicado que reutiliza el último resultado durante 100 frames (→ ver [`anonymize()`](#codigo), línea 14). El stream no se bloquea aunque la máscara vaya ligeramente rezagada cuando una cara entra o sale de plano. La detección trabaja además a 1/4 de resolución (`FACE_SCALE=0.25`) para reducir el tiempo de inferencia.

### Pre-buffer con `deque` de tamaño dinámico

El clip guardado incluye los segundos previos al evento gracias a un `deque` circular. Su tamaño se recalcula cada frame a partir del FPS real (`max(1, int(pre_sec * fps))`), de forma que el pre-buffer siempre representa los mismos segundos reales independientemente de si la cámara va a 15 o 30 fps.

### Frames pares descartados del procesamiento

El bucle principal salta el procesamiento en frames pares, reutilizando el último frame anonimizado. Esto reduce a la mitad las llamadas a `anonymize`, `clean_mask` y `classify_motion` sin que sea perceptible visualmente a 25+ fps.

### Telegram como canal de alertas

La foto se envía desde el hilo principal para garantizar que corresponde exactamente al frame que disparó el evento; el clip se guarda y envía al terminar la grabación. Si `token.env` no existe el programa falla en el arranque — fallo rápido y visible antes de que el sistema quede corriendo sin alertas.

---

## Limitaciones { #limitaciones }

!!! warning "Limitaciones conocidas"
    - Clasificación heurística: personas agachadas o sentadas pueden clasificarse como "objeto".
    - Con modelo `hog`, las caras de perfil o parcialmente tapadas pueden quedar sin anonimizar.
    - Cualquier vibración de la cámara genera falsos positivos en todo el foreground.
    - Si `token.env` no existe o está mal configurado, el programa lanza excepción al arrancar.