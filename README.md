# SWL-LSE Isolated Sign Recognition (PyTorch + TCN)

Proyecto completo para reconocimiento aislado de signos LSE con:
- Entrenamiento supervisado (300 clases) sobre PKLs de MediaPipe.
- Inspección automática obligatoria del dataset.
- Demo de inferencia en tiempo real con webcam.

## 1) Estructura esperada del dataset

```text
<DATA_ROOT>/
  MEDIAPIPE/
    1234567890123456.pkl
    ...
  splits/
    train.csv
    val.csv
    test.csv
  videos_ref_annotations.csv
```

- `train.csv`, `val.csv`, `test.csv` **sin cabecera**, columnas: `[FILENAME, CLASS_ID]`.
- `videos_ref_annotations.csv` con cabecera: `FILENAME, CLASS_ID, LABEL`.

## 2) Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Inspección automática (Paso 0)

Genera estadísticas del dataset y `artifacts/feature_spec.json`:

```bash
python tools/inspect_dataset.py \
  --train_csv <DATA_ROOT>/splits/train.csv \
  --mediapipe_dir <DATA_ROOT>/MEDIAPIPE \
  --output artifacts/feature_spec.json \
  --n_samples 20
```

Qué valida:
- distribución temporal `T` (frames por secuencia),
- porcentaje de frames sin manos,
- integridad de pose (`33` landmarks + attrs `x,y,z,visibility,presence`),
- decisión automática de `T_fixed` (percentil 95, cap 160).

## 4) Features por frame (D=291)

Orden fijo:
1. Pose (33 landmarks): `[x,y,z,visibility,presence]` => `33*5 = 165`
2. Mano izquierda (slot fijo): `21*[x,y,z] = 63`
3. Mano derecha (slot fijo): `21*[x,y,z] = 63`

Total: `165 + 63 + 63 = 291`

Normalización:
- Centro corporal = midpoint(LEFT_HIP, RIGHT_HIP) si visibles, si no midpoint(shoulders), si no origen.
- Escala = distancia entre hombros (fallback `1.0`).
- Pose y manos se transforman con el mismo centro/escala.

Secuencias:
- `T < T_fixed`: zero-padding al final.
- `T > T_fixed`: submuestreo uniforme a `T_fixed`.

Aumentaciones ligeras (train):
- temporal dropout,
- jitter gaussiano pequeño,
- landmark dropout de baja probabilidad.

## 5) Entrenamiento

```bash
python train.py \
  --mediapipe_dir <DATA_ROOT>/MEDIAPIPE \
  --splits_dir <DATA_ROOT>/splits \
  --annotations_csv <DATA_ROOT>/videos_ref_annotations.csv \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3
```

Salida:
- `checkpoints/best.pt`
- `artifacts/label_map.json`
- `artifacts/feature_spec.json`
- `artifacts/confusion_matrix.png`
- `logs/history.csv`

Métricas: top-1 accuracy, macro F1.

## 6) Demo webcam en tiempo real

```bash
python realtime_webcam.py \
  --checkpoint checkpoints/best.pt \
  --feature_spec artifacts/feature_spec.json \
  --label_map artifacts/label_map.json
```

Características:
- extracción de landmarks con MediaPipe Pose + Hands,
- buffer deslizante de tamaño `T_fixed`,
- inferencia cada `N` frames (`--infer_every`),
- suavizado por media de logits (`--smooth_k`),
- umbral de confianza para mostrar `Desconocido`.

## 7) Tests

```bash
pytest -q
```

Incluye test de:
- dimensión exacta `291`,
- determinismo del orden de features.
