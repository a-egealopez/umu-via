# Ej. 7 — Realidad Aumentada Gestual

!!! abstract "Enunciado"
    Crea un efecto de realidad aumentada en el que el usuario desplace objetos virtuales hacia posiciones marcadas con el ratón.

---

## Parámetros clave { #parametros }

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `_BASE_PX` | 80 px | Radio base del objeto en píxeles a escala 1.0 |
| `_SCALE_MIN` | 0.3 | Escala mínima del objeto AR (mano lejana) |
| `_SCALE_MAX` | 2.5 | Escala máxima del objeto AR (mano cerca) |
| `_XY_FRAC` | 0.60 | Fracción del ancho/alto del frame para el offset XY de palma |
| `_SMOOTH` | 0.25 | Coeficiente α del filtro exponencial (escala, yaw, XY) |
| `_SMOOTH_RY` | 0.18 | Coeficiente α del filtro exponencial para roll (más conservador) |
| `max_edges` | 4 000 | Límite de aristas al cargar un modelo `.obj` denso |

---

## Carga del modelo y fallback { #modelo }

`ARViewer` admite un fichero `.obj` generado por COLMAP o VGGT; si no se especifica `--model` o el fichero no existe, usa un **cubo unitario de referencia** para que el sistema sea ejecutable en cualquier momento.

El `load_obj` maneja dos casos:

- **Mallas con caras** (`.obj` con líneas `f`): extrae aristas únicas de cada polígono.
- **Nubes de puntos sin caras** (salida típica de COLMAP/VGGT exportada directamente): computa el `ConvexHull` de los vértices para derivar aristas y mostrar la envolvente del objeto.

```python title="extra_8_7_2/ar_viewer.py — load_obj() gestión de nube de puntos" linenums="1"
if not edges:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(arr)
    for simplex in hull.simplices:
        for k in range(len(simplex)):
            a, b = int(simplex[k]), int(simplex[(k + 1) % len(simplex)])
            edges.add((min(a, b), max(a, b)))
```

---

## Anclaje con ratón y desplazamiento gestual { #anclaje }

<figure markdown>
  ![AR cubo con anclaje de ratón](extra_ar_cube.png)
  <figcaption>Cubo de referencia proyectado sobre la imagen con el marcador de anclaje (cruz amarilla). Un clic izquierdo mueve el ancla a cualquier posición de la imagen.</figcaption>
</figure>

<figure markdown>
  ![AR con modelo .obj cargado](extra_ar_obj.png)
  <figcaption>Modelo 3D reconstruido cargado en el visor AR. La mano (visible en la esquina) controla la escala y rotación en tiempo real.</figcaption>
</figure>

El objeto se renderiza en `cx = anchor_x + ox`, `cy = anchor_y + oy`, donde `ox` y `oy` son offsets continuos derivados de la posición normalizada de la palma:

```python title="extra_8_7_2/ar_viewer.py — update()" linenums="1"
target_ox = (state.norm_x - 0.5) * w * self._XY_FRAC   # _XY_FRAC = 0.60
target_oy = (state.norm_y - 0.5) * h * self._XY_FRAC
```

Cuando la palma está centrada en el frame (`norm_x = norm_y = 0.5`), el offset es cero y el objeto queda exactamente sobre el ancla. Desplazando la palma se puede arrastrar el objeto hasta ±30 % del ancho/alto del frame a partir del ancla.

<figure markdown>
  ![Reposicionamiento con clic de ratón](extra_ar_anchor.png)
  <figcaption>Secuencia: clic sobre una nueva posición y el objeto salta al ancla marcada. La mano retoma el control de offset y rotación desde ese punto.</figcaption>
</figure>

---

## Renderizado en perspectiva simplificada { #renderizado }

El visor aplica rotaciones `Ry(ry) @ Rx(rx)` sobre los vértices y proyecta solo los dos primeros componentes al plano de imagen (proyección ortográfica). La profundidad del eje Z se usa únicamente para modular el color de cada arista:

```python title="extra_8_7_2/ar_viewer.py — draw()" linenums="1"
for i, j in self._edges:
    z = float(rot[i, 2] + rot[j, 2]) * 0.5
    g = int(np.clip((z + 1) * 0.5 * 155 + 100, 100, 255))
    cv.line(frame, tuple(ipts[i]), tuple(ipts[j]), (0, g, 255 - g // 2), 2, cv.LINE_AA)
```

Las aristas más alejadas de la cámara (z pequeno) son azul intenso; las más cercanas (z grande) viran al verde.

---

## Decisiones de diseno { #decisiones }

### Separación ancla / offset

El ratón fija el centro del objeto (ancla) y la mano solo aplica un offset relativo. Esto permite anclar el objeto sobre un elemento de la escena con precisión y luego moverlo con la mano sin tener que mantener la palma exactamente en ese punto.

### Suavizado exponencial asimétrico

Se usan dos valores de `α` distintos: `0.25` para escala, yaw y XY (respuesta ágil), y `0.18` para roll (más suave). El roll se estima a partir de la coordenada Z de MediaPipe, que es más ruidosa que la posición XY, por lo que necesita un filtro más conservador para evitar temblores visibles.

---

## Limitaciones { #limitaciones }

!!! warning "Limitaciones conocidas"
    - La proyección es **ortográfica**, no perspectiva: no hay corrección por posición de cámara ni escala dependiente de la profundidad. El efecto AR no es fotorrealista.
    - Con modelos complejos (muchos vértices), el `ConvexHull` para nubes sin caras puede ser muy lento o producir geometría incorrecta.
    - El límite de `max_edges=4000` puede simplificar en exceso modelos densos, perdiendo detalle geométrico.
