# Basketball Tracker CV 🏀

Un sistema per il tracciamento e l'analisi di partite di basket in tempo reale, con visualizzazione tattica, basato su machine learning e computer vision.

![Basketball Tracker Demo](images/forREADME/demo.gif)


[English version below](#english-version)

## 📋 Indice
- [Caratteristiche](#caratteristiche)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Struttura del Progetto](#struttura-del-progetto)
- [Modelli](#modelli)
- [Output](#output)
- [Risoluzione Problemi](#risoluzione-problemi)
- [Licenza](#licenza)

## ✨ Caratteristiche

- **Rilevamento Giocatori e Pallone**: Utilizza modelli YOLO11 o RF-DETR per il rilevamento accurato di giocatori e pallone
- **Tracciamento Multi-Oggetto**: Implementa ByteTrack per tracciare i giocatori attraverso i frame anche con occlusioni
- **Rilevamento Keypoint del Campo**: Identifica automaticamente i punti chiave del campo da basket (linee, punti chiave, ecc.)
- **Vista Tattica**: Trasforma le posizioni dei giocatori in una vista tattica 2D del campo
- **Interpolazione Intelligente**: Riempie automaticamente i dati mancanti per posizioni di giocatori e pallone
- **Calcolo del Possesso**: Determina quale giocatore ha il possesso del pallone
- **Video Annotato**: Genera video di output con overlay delle rilevazioni e vista tattica

## 🔧 Requisiti

### Software
- Python 3.8+
- CUDA (opzionale, per accelerazione GPU)


## 📦 Installazione

1. **Clonare il repository**
```bash
git clone https://github.com/pgnsamu/basketballTrackerCV.git
cd basketballTrackerCV
```

2. **Creare un ambiente virtuale** (consigliato)
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

3. **Installare le dipendenze**
```bash
pip install -r requirements.txt
```

4. **Scaricare i modelli pre-addestrati**

Devi avere due modelli:
- Modello per il rilevamento dei keypoint del campo (es. `models/BEST2.pt`)
- Modello per il rilevamento di giocatori e pallone (es. `models/PlayerDet.pt`)

Posiziona i modelli nella cartella `models/`.

## 🚀 Utilizzo

### Comando Base
```bash
python main.py --keypoint-model models/BEST2.pt --player-model models/PlayerDet.pt
```

Purtroppo ancora non si è riusciti ad alleggerire i processi di inferenza in modo tale da permettere l'esecuzione in tempo reale su video a 30 fps e risoluzione 1920x1080.

### Parametri Disponibili

#### Video Input/Output
- `--video`: Path del video da processare (default: `input_video/video_1.mp4`)
- `--output-path`: Path del video di output (default: `outputVideo/output_video.mp4`)
- `--fps`: FPS del video di output (default: `30.0`)

#### Modelli
- `--keypoint-model`: Path del modello per rilevamento keypoints (richiesto)
- `--player-model`: Path del modello per rilevamento giocatori (richiesto)

#### Stub (Cache)
##### In qualunque caso ai path degli stubs viene concatenato alla fine il nome del video processato, per permettere di avere stubs separati per video diversi
- `--keypoint-stub`: Path dello stub per keypoints (default: `stubs/court_key_points_stub.pkl`)
- `--player-stub`: Path dello stub per posizioni giocatori (default: `stubs/players_positions_stub.pkl`)
- `--no-stub`: Disabilita lettura da stub e ricalcola tutto

#### Altri
- `--court-image`: Path immagine campo tattico (default: `images/basketball_court.png`)
- `--debug`: Abilita modalità debug per visualizzazione numero frame e salvataggio keypoints rilevati in un file di testo

### Esempi

**Elaborare un video specifico:**
```bash
python main.py \
  --video input_video/game1.mp4 \
  --output-path outputVideo/game1_analyzed.mp4 \
  --keypoint-model models/BEST2.pt \
  --player-model models/PlayerDet.pt
```

**Modalità debug (visualizza i frame durante l'elaborazione):**
```bash
python main.py \
  --video video_1.mp4 \
  --keypoint-model models/BEST2.pt \
  --player-model models/PlayerDet.pt \
  --debug
```

**Ricalcolare tutto senza usare la cache:**
```bash
python main.py \
  --video video_1.mp4 \
  --keypoint-model models/BEST2.pt \
  --player-model models/PlayerDet.pt \
  --no-stub
```

## 📁 Struttura del Progetto

```
basketballTrackerCV/
├── main.py                          # Entry point principale
├── requirements.txt                 # Dipendenze Python
├── README.md                        # Questo file
│
├── detectors/                       # Moduli di rilevamento
│   ├── keypoint_detector.py        # Rilevamento keypoint del campo
│   ├── player_ball_detector.py     # Rilevamento giocatori e pallone
│   └── player_tracker.py           # Tracciamento e interpolazione
│
├── homography/                      # Trasformazione prospettica
│   └── homography.py               # Calcolo omografia
│
├── tactical_view_converter/        # Vista tattica
│   └── tactical_view_converter.py  # Conversione coordinate a vista tattica
│
├── drawers/                         # Rendering e visualizzazione
│   ├── drawWindow.py               # Gestione finestra output
│   └── drawPoint.py                # Rendering elementi grafici
│
├── utils/                           # Utility varie
│   ├── video_utils.py              # Lettura/scrittura video
│   ├── detectedObject.py           # Classi oggetti rilevati
│   ├── bbox_utils.py               # Utilità bounding box
│   ├── stubs_utils.py              # Gestione cache
│   └── overlay.py                  # Overlay grafici
│
├── models/                          # Modelli di ML (non inclusi nel repo)
│   ├── BEST2.pt                    # Modello keypoint
│   └── PlayerDet.pt                # Modello giocatori/pallone
│
├── images/                          # Immagini risorse
│   └── basketball_court.png        # Immagine campo tattico
│
├── input_video/                     # Video di input
├── outputVideo/                     # Video elaborati
└── stubs/                           # File cache (generati automaticamente)
```

## 🤖 Modelli

Il progetto utilizza due modelli di deep learning:

### 1. Modello Keypoint del Campo
- **Tipo**: YOLO per keypoint detection
- **Scopo**: Rileva 18 keypoint del campo da basket
- **Keypoint rilevati**:
  - [0] Angolo alto (lato sinistro)
  - [1] Linea dei 3 punti dall'alto (lato sinistro)
  - [2] Angolo sinistro alto dell'area per il tiro libero (lato sinistro)
  - [3] Angolo sinistro basso dell'area per il tiro libero (lato sinistro)
  - [4] Linea dei 3 punti dal basso (lato sinistro)
  - [5] Angolo basso (lato sinistro)
  - [8] Angolo destro alto dell'area per il tiro libero (lato sinistro)
  - [9] Angolo destro basso dell'area per il tiro libero (lato sinistro)
  ---
  - [6] Linea di metà campo punto alto
  - [7] Linea di metà campo punto basso
  ---
  - [10] Angolo alto (lato destro)
  - [11] Linea dei 3 punti dall'alto (lato destro)
  - [12] Angolo destro alto dell'area per il tiro libero (lato destro)
  - [13] Angolo destro basso dell'area per il tiro libero (lato destro)
  - [14] Linea dei 3 punti dal basso (lato destro)
  - [15] Angolo basso (lato destro)
  - [16] Angolo sinistro alto dell'area per il tiro libero (lato destro)
  - [17] Angolo sinistro basso dell'area per il tiro libero (lato destro)

  ![Basketball Court Keypoints](images/forREADME/court_stylized.png)

  #### Metriche del Modello per il rilevamento del campo durante l'addestramento:
  ![Keypoint Model Metrics](images/forREADME/metricsmAP50-95(B).png)

  
### 2. Modello Giocatori e Pallone
- **Tipo**: YOLO11 o RF-DETR
- **Classi rilevate**:
  - Classe 0/1: Pallone
  - Classe 3/4: Giocatore
- **Features**:
  - Tracciamento multi-oggetto con ByteTrack
  - Rilevamento del possesso palla
  - Smoothing EMA per posizione pallone

  #### Metriche del Modello per il rilevamento di giocatori e pallone durante l'addestramento:
  ![Player and Ball Model Metrics](images/forREADME/results.png)


## 📊 Output

Il sistema genera:

1. **Video Annotato**: Video con overlay che include:
   - Bounding box dei giocatori
   - Posizione del pallone
   - Keypoint del campo visualizzati
   - Vista tattica sovrapposta
   - Indicatore possesso palla

2. **File Stub**: Tutti i risultati rilevati vengono salvati nei file di stub per consentire test più rapidi dello stesso video di input
   - `court_key_points_stub.pkl`: Keypoint del campo per frame
   - `players_positions_stub.pkl`: Posizioni giocatori per frame
   - `balls_positions_stub.pkl`: Posizioni pallone per frame

3. **File Debug** (modalità `--debug`):
   - `court_keypoints.txt`: Log dei keypoint rilevati

## 🔍 Pipeline di Elaborazione

1. **Caricamento Video**: Lettura dei frame dal video di input
2. **Rilevamento Keypoint**: Identificazione punti chiave del campo
3. **Rilevamento Oggetti**: Detection di giocatori e pallone
4. **Tracciamento**: Assegnazione ID consistenti ai giocatori
5. **Interpolazione**: Riempimento posizioni mancanti
6. **Validazione**: Controllo coerenza keypoint rilevati
7. **Trasformazione**: Conversione a coordinate vista tattica
8. **Rendering**: Generazione video finale con annotazioni

## Tracciamento dei giocatori
Il tracciamento dei giocatori è gestito da ByteTrack, che assegna ID univoci a ciascun giocatore rilevato e mantiene il tracciamento anche in caso di occlusioni temporanee.

```python
self.TRACKER = sv.ByteTrack(
    track_activation_threshold=0.35,
    lost_track_buffer=60,
    minimum_matching_threshold=0.95,
    frame_rate=30
)
```

## Interpolazione
 
### Palla
Le posizioni della palla possono essere assenti in alcuni frame (detection mancante).
Per ottenere una traiettoria continua, convertiamo i bounding box in una tabella (x1,y1,x2,y2) e applichiamo:
  -	interpolazione lineare sui frame mancanti
  -	backfill (bfill) per riempire eventuali buchi all’inizio della sequenza

### Giocatori
Per ridurre lo sfarfallio (box che “saltano” o spariscono per pochi frame), interpoliamo le coordinate separatamente per ogni track_id:
  1.	raccogliamo tutte le detection (frame, track_id, bbox, class_id)
  2.	per ogni track_id creiamo un range completo di frame tra prima e ultima apparizione
  3.	reindicizziamo inserendo i frame mancanti (NaN)
  4.	interpoliamo linearmente x1,y1,x2,y2 sui buchi
  5.	class_id viene propagato con forward fill / backfill
  6.	ricostruiamo list[list[Player]] per frame


## Validazione dei Keypoint Rilevati
il processo di validazione è stato implementato in maniera try and error testando su diverse combinazioni di video, cercando di rifinire il risultato finale.

#### Algoritmo finale di validazione:

<img src="images/forREADME/algo222.png" alt="drawing" width="200"/>

## Omografia

### Riferimenti teorici all'omografia

L'omografia è una trasformazione prospettica 3×3 che mappa punti da un piano a un altro:

$$\begin{equation*}
H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\\ h_{21} & h_{22} & h_{23} \\\ h_{31} & h_{32} & h_{33} \end{pmatrix}
\end{equation*}$$

This ensures the matrix renders as a block element rather than inline.

La trasformazione di un punto $(x, y)$ in coordinate omogenee è:

$$\begin{pmatrix} x' \\\ y' \\\ w' \end{pmatrix} = H \begin{pmatrix} x \\\ y \\\ 1 \end{pmatrix}$$

Il punto risultante in coordinate cartesiane è:

$$x_{risultato} = \frac{x'}{w'}, \quad y_{risultato} = \frac{y'}{w'}$$

Nel progetto, l'omografia viene calcolata utilizzando almeno 4 corrispondenze punto-a-punto tra i keypoint del campo rilevati e le coordinate note nel piano tattico, applicando l'algoritmo DLT (Direct Linear Transform) o RANSAC per robustezza.


## 🐛 Risoluzione Problemi

### Errore: "OMP: Error #15"
Questo errore è già gestito nel codice. Se persiste:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
set KMP_DUPLICATE_LIB_OK=TRUE     # Windows
```

### Performance lente
- Assicurati di avere CUDA installato per usare la GPU
- Usa file stub per evitare di rielaborare video già processati
- Riduci la risoluzione del video di input

### Keypoint non rilevati correttamente
- Verifica che il modello keypoint sia addestrato sui dati corretti
- Controlla l'illuminazione e la qualità del video
- Aumenta il `conf_threshold` nel CourtKeypointDetector

### Giocatori non tracciati
- Verifica le soglie di confidenza in `player_ball_detector.py`
- Controlla la dimensione di inferenza (attualmente 1280px)
- Verifica che il modello sia compatibile con la risoluzione del video


## Problemi noti
- Il sistema può avere difficoltà con occlusioni pesanti o movimenti molto rapidi, portando a una temporanea perdita di tracciamento. I miglioramenti futuri si concentreranno sull'aumentare la robustezza in questi scenari.
- Se i keypoint del campo sono sulla stessa linea orizzontale/verticale, il calcolo dell'omografia può diventare instabile.
- In caso di assenza di rilevamenti dei keypoint per un frame, la trasposizione dei giocatori non verrà eseguita.
- L'implementazione attuale non supporta ancora l'elaborazione in tempo reale a 30 fps e risoluzione 1920x1080 a causa di vincoli computazionali. Le ottimizzazioni future mireranno a migliorare le prestazioni.
- Il sistema può avere difficoltà a determinare con precisione il possesso della palla in scene affollate o quando la palla è occlusa.
- Se le canotte dei giocatori sono di colore simile a quello del campo, il modello potrebbe creare falsi positivi.

## Miglioramenti futuri
- Aggiungere il riconoscimento dei numeri di maglia basato su OCR per l'identificazione e il tracciamento dei giocatori
- Implementare un modello di segmentazione per migliorare la classificazione delle squadre dei giocatori (casa vs trasferta)
- Ottimizzare la velocità di inferenza per raggiungere l'elaborazione in tempo reale a 30 fps e risoluzione 1920x1080


## 📄 Licenza

La licenza per questo progetto non è ancora stata specificata.

Dataset di training giocatori e palla: [basketball-player-detection-3](https://universe.roboflow.com/projects-wh5rm/basketball-player-detection-3-ycjdo-cffrq)

Dataset di training keypoint del campo: [reloc2-den7l](https://universe.roboflow.com/fyp-3bwmg/reloc2-den7l)

---

## English Version

# Basketball Tracker CV 🏀

A system for real-time basketball game tracking and analysis with tactical visualization, based on machine learning and computer vision.

## 📋 Table of Contents
- [Features](#features-1)
- [Requirements](#requirements-1)
- [Installation](#installation-1)
- [Usage](#usage-1)
- [Project Structure](#project-structure-1)
- [Models](#models-1)
- [Output](#output-1)
- [Troubleshooting](#troubleshooting-1)
- [License](#license-1)

## ✨ Features

- **Player and Ball Detection**: Uses YOLO11 or RF-DETR models for accurate player and ball detection
- **Multi-Object Tracking**: Implements ByteTrack to track players across frames even with occlusions
- **Court Keypoint Detection**: Automatically identifies basketball court keypoints (lines, keypoints, etc.)
- **Tactical View**: Transforms player positions into a 2D tactical court view
- **Smart Interpolation**: Automatically fills missing data for player and ball positions
- **Possession Calculation**: Determines which player has ball possession
- **Annotated Video**: Generates output video with detection overlays and tactical view

## 🔧 Requirements

### Software
- Python 3.8+
- CUDA (optional, for GPU acceleration)


## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/pgnsamu/basketballTrackerCV.git
cd basketballTrackerCV
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models**

You need two models:
- Court keypoint detection model (e.g., `models/BEST2.pt`)
- Player and ball detection model (e.g., `models/PlayerDet.pt`)

Place the models in the `models/` folder.

## 🚀 Usage

### Basic Command
```bash
python main.py --keypoint-model models/BEST2.pt --player-model models/PlayerDet.pt
```
Unfortunately, the inference processes have not yet been optimized to allow real-time execution on videos at 30 fps and 1920x1080 resolution.

### Available Parameters

#### Video Input/Output
- `--video`: Path to video to process (default: `input_video/video_1.mp4`)
- `--output-path`: Output video path (default: `outputVideo/output_video.mp4`)
- `--fps`: Output video FPS (default: `30.0`)

#### Models
- `--keypoint-model`: Keypoint detection model path (required)
- `--player-model`: Player detection model path (required)

#### Stub (Cache)
##### In any case, the video name is appended to the end of the stub paths to allow separate stubs for different videos
- `--keypoint-stub`: Keypoint stub path (default: `stubs/court_key_points_stub.pkl`)
- `--player-stub`: Player positions stub path (default: `stubs/players_positions_stub.pkl`)
- `--no-stub`: Disable stub reading and recalculate everything

#### Other
- `--court-image`: Tactical court image path (default: `images/basketball_court.png`)
- `--debug`: Enable debug mode with frame visualization and saving detected keypoints to a text file

### Examples

**Process a specific video:**
```bash
python main.py \
  --video input_video/game1.mp4 \
  --output-path outputVideo/game1_analyzed.mp4 \
  --keypoint-model models/BEST2.pt \
  --player-model models/PlayerDet.pt
```

**Debug mode (displays frames during processing):**
```bash
python main.py \
  --video video_1.mp4 \
  --keypoint-model models/BEST2.pt \
  --player-model models/PlayerDet.pt \
  --debug
```

**Recalculate everything without using cache:**
```bash
python main.py \
  --video video_1.mp4 \
  --keypoint-model models/BEST2.pt \
  --player-model models/PlayerDet.pt \
  --no-stub
```

## 📁 Project Structure

```
basketballTrackerCV/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── detectors/                       # Detection modules
│   ├── keypoint_detector.py        # Court keypoint detection
│   ├── player_ball_detector.py     # Player and ball detection
│   └── player_tracker.py           # Tracking and interpolation
│
├── homography/                      # Perspective transformation
│   └── homography.py               # Homography calculation
│
├── tactical_view_converter/        # Tactical view
│   └── tactical_view_converter.py  # Coordinate conversion to tactical view
│
├── drawers/                         # Rendering and visualization
│   ├── drawWindow.py               # Output window management
│   └── drawPoint.py                # Graphic elements rendering
│
├── utils/                           # Various utilities
│   ├── video_utils.py              # Video read/write
│   ├── detectedObject.py           # Detected object classes
│   ├── bbox_utils.py               # Bounding box utilities
│   ├── stubs_utils.py              # Cache management
│   └── overlay.py                  # Graphic overlays
│
├── models/                          # ML models (not included in repo)
│   ├── BEST2.pt                    # Keypoint model
│   └── PlayerDet.pt                # Player/ball model
│
├── images/                          # Resource images
│   └── basketball_court.png        # Tactical court image
│
├── input_video/                     # Input videos
├── outputVideo/                     # Processed videos
└── stubs/                           # Cache files (auto-generated)
```

## 🤖 Models

The project uses two deep learning models:

### 1. Court Keypoint Model
- **Type**: YOLO for keypoint detection
- **Purpose**: Detects 18 basketball court keypoints
- **Detected keypoints**:
  - **Keypoint rilevati**:
  - [0] Top left corner (lato sinistro)
  - [1] Top 3-point line (lato sinistro)
  - [2] Top left corner of free throw area (lato sinistro)
  - [3] Bottom left corner of free throw area (lato sinistro)
  - [4] Bottom 3-point line (lato sinistro)
  - [5] Bottom left corner (lato sinistro)
  - [8] Top right corner of free throw area (lato sinistro)
  - [9] Bottom right corner of free throw area (lato sinistro)
  ---
  - [6] Top half-court line
  - [7] Bottom half-court line
  ---
  - [10] Top right corner (lato destro)
  - [11] Top 3-point line (lato destro)
  - [12] Top right corner of free throw area (lato destro)
  - [13] Bottom right corner of free throw area (lato destro)
  - [14] Bottom 3-point line (lato destro)
  - [15] Bottom right corner (lato destro)
  - [16] Top left corner of free throw area (lato destro)
  - [17] Bottom left corner of free throw area (lato destro)

  ![Basketball Court Keypoints](images/forREADME/court_stylized.png)

  #### Court Keypoint Model metrics during training:
  ![Keypoint Model Metrics](images/forREADME/metricsmAP50-95(B).png)

### 2. Player and Ball Model
- **Type**: YOLO11 or RF-DETR
- **Detected classes**:
  - Class 0/1: Ball
  - Class 3/4: Player
- **Features**:
  - Multi-object tracking with ByteTrack
  - Ball possession detection
  - EMA smoothing for ball position

  #### Player and Ball Model metrics during training:
  ![Player and Ball Model Metrics](images/forREADME/results.png)


## 📊 Output

The system generates:

1. **Annotated Video**: Video with overlays including:
   - Player bounding boxes
   - Ball position
   - Visualized court keypoints
   - Overlaid tactical view
   - Ball possession indicator

2. **Stub Files**: All of the detected results are saved in stubs file to allow faster testing of the same input video
   - `court_key_points_stub.pkl`: Court keypoints per frame
   - `players_positions_stub.pkl`: Player positions per frame
   - `balls_positions_stub.pkl`: Ball positions per frame

3. **Debug Files** (`--debug` mode):
   - `court_keypoints.txt`: Log of detected keypoints

## 🔍 Processing Pipeline

1. **Video Loading**: Read frames from input video
2. **Keypoint Detection**: Identify court keypoints
3. **Object Detection**: Detect players and ball
4. **Tracking**: Assign consistent IDs to players
5. **Interpolation**: Fill missing positions
6. **Validation**: Check consistency of detected keypoints
7. **Transformation**: Convert to tactical view coordinates
8. **Rendering**: Generate final video with annotations

## Players Tracking
The players are tracked using ByteTrack, which assigns unique IDs to each detected player and maintains tracking even in case of temporary occlusions.

```python
self.TRACKER = sv.ByteTrack(
    track_activation_threshold=0.35,
    lost_track_buffer=60,
    minimum_matching_threshold=0.95,
    frame_rate=30
)
```

## Interpolation
 
### Ball
Ball positions may be missing in some frames (detection missing).
To obtain a continuous trajectory, we convert the bounding boxes into a table (x1,y1,x2,y2) and apply:
  -	linear interpolation on missing frames
  -	backfill (bfill) to fill any gaps at the beginning of the sequence

### Players
To reduce flickering (boxes that "jump" or disappear for a few frames), we interpolate the coordinates separately for each track_id:
  1.	collect all detections (frame, track_id, bbox, class_id)
  2.	for each track_id, create a complete range of frames between the first and last appearance
  3.	reindex by inserting missing frames (NaN)
  4.	interpolate x1,y1,x2,y2 linearly over the gaps
  5.	class_id is propagated with forward fill / backfill
  6.	reconstruct list[list[Player]] per frame

## Validation of Detected Keypoints
The validation process has been implemented in a try and error way, testing on different combinations of videos, trying to refine the final result.

#### Pseudo-code for final validation:
<img src="images/forREADME/algo2_en22.png" alt="drawing" width="200"/>

## Homography 

### Theoretical references to homography

The homography is a 3×3 perspective transformation that maps points from one plane to another:

$$H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\\ h_{21} & h_{22} & h_{23} \\\ h_{31} & h_{32} & h_{33} \end{pmatrix}$$

The transformation of a point $(x, y)$ in homogeneous coordinates is:

$$\begin{pmatrix} x' \\\ y' \\\ w' \end{pmatrix} = H \begin{pmatrix} x \\\ y \\\ 1 \end{pmatrix}$$

The resulting point in Cartesian coordinates is:

$$x_{result} = \frac{x'}{w'}, \quad y_{result} = \frac{y'}{w'}$$

In the project, the homography is calculated using at least 4 point-to-point correspondences between the detected court keypoints and the known coordinates in the tactical plane, applying the DLT (Direct Linear Transform) or RANSAC algorithm for robustness.


## 🐛 Troubleshooting

### Error: "OMP: Error #15"
This error is already handled in the code. If it persists:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
set KMP_DUPLICATE_LIB_OK=TRUE     # Windows
```

### Slow performance
- Make sure CUDA is installed to use GPU
- Use stub files to avoid reprocessing already processed videos
- Reduce input video resolution

### Keypoints not detected correctly
- Verify the keypoint model is trained on correct data
- Check lighting and video quality
- Increase `conf_threshold` in CourtKeypointDetector

### Players not tracked
- Check confidence thresholds in `player_ball_detector.py`
- Check inference size (currently 1280px)
- Verify model is compatible with video resolution

## Known issues
- The system may struggle with heavy occlusions or very fast movements, leading to temporary loss of tracking. Future improvements will focus on enhancing robustness in these scenarios.
- In case the court keypoints are on the same horizontal/vertical line, the homography calculation may become unstable.
- In case of no detections of keypoints for a frame the transposition of the players won't be performed
- The current implementation does not yet support real-time processing at 30 fps and 1920x1080 resolution due to computational constraints. Future optimizations will aim to improve performance.
- The system may have difficulty accurately determining ball possession in crowded scenes or when the ball is occluded.
- if the players' jerseys are of a color similar to that of the court, the model may create false positives.

## Future Improvements
- Add OCR-based jersey number recognition for player identification and tracking
- Implement a segmentation model to improve the team classification of players (home vs away)
- Optimize inference speed to achieve real-time processing at 30 fps and 1920x1080 resolution


## 📄 License

The license for this project has not yet been specified.

Training dataset: [basketball-player-detection-3](https://universe.roboflow.com/projects-wh5rm/basketball-player-detection-3-ycjdo-cffrq) 

Dataset for court keypoints: [reloc2-den7l](https://universe.roboflow.com/fyp-3bwmg/reloc2-den7l)

---

## 👥 Authors

- [@pgnsamu](https://github.com/pgnsamu)
- [@AlessioCesarini](https://github.com/AlessioCesarini)
- [@Z0platen](https://github.com/Z0platen)

## 🙏 Acknowledgments

- YOLO by Ultralytics
- RF-DETR detection framework
- Supervision library for tracking
- Roboflow for the training dataset
- [Embiricos, Alexander, and Gabriel Poon. "Single-view 3D reconstruction of basketball scenes." Retreived on the 16th of September (2017).](https://arc.net/l/quote/npmaqijg)
