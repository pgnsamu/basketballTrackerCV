# Basketball Tracker CV 🏀

Un sistema avanzato di computer vision per il tracciamento e l'analisi di partite di basket in tempo reale, con visualizzazione tattica.

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
- [Contribuire](#contribuire)
- [Licenza](#licenza)

## ✨ Caratteristiche

- **Rilevamento Giocatori e Pallone**: Utilizza modelli YOLO11 o RF-DETR per il rilevamento accurato di giocatori e pallone
- **Tracciamento Multi-Oggetto**: Implementa ByteTrack per tracciare i giocatori attraverso i frame anche con occlusioni
- **Rilevamento Keypoint del Campo**: Identifica automaticamente i punti chiave del campo da basket (linee, canestri, etc.)
- **Vista Tattica**: Trasforma le posizioni dei giocatori in una vista tattica 2D del campo
- **Interpolazione Intelligente**: Riempie automaticamente i dati mancanti per posizioni di giocatori e pallone
- **Calcolo del Possesso**: Determina quale giocatore ha il possesso del pallone
- **Video Annotato**: Genera video di output con overlay delle rilevazioni e vista tattica

## 🔧 Requisiti

### Software
- Python 3.8+
- CUDA (opzionale, per accelerazione GPU)

### Dipendenze Python
```
ultralytics==8.3.67
rfdetr
opencv-python
numpy
torch
supervision
```

Installare tutte le dipendenze con:
```bash
pip install -r requirements.txt
```

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

### Parametri Disponibili

#### Video Input/Output
- `--video`: Path del video da processare (default: `video_1.mp4`)
- `--output-path`: Path del video di output (default: `outputVideo/output_video.mp4`)
- `--fps`: FPS del video di output (default: `30.0`)

#### Modelli
- `--keypoint-model`: Path del modello per rilevamento keypoints (richiesto)
- `--player-model`: Path del modello per rilevamento giocatori (richiesto)

#### Stub (Cache)
- `--keypoint-stub`: Path dello stub per keypoints (default: `stubs/court_key_points_stub.pkl`)
- `--player-stub`: Path dello stub per posizioni giocatori (default: `stubs/players_positions_stub.pkl`)
- `--no-stub`: Disabilita lettura da stub e ricalcola tutto

#### Altri
- `--court-image`: Path immagine campo tattico (default: `./images/basketball_court.png`)
- `--debug`: Abilita modalità debug con visualizzazione frame

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
  - Bordi sinistro e destro del campo
  - Linea di metà campo
  - Linee tiro libero
  - Angoli del campo
  - Altri punti caratteristici

### 2. Modello Giocatori e Pallone
- **Tipo**: YOLO11 o RF-DETR
- **Classi rilevate**:
  - Classe 0/1: Pallone
  - Classe 3/4: Giocatore
- **Features**:
  - Tracciamento multi-oggetto con ByteTrack
  - Rilevamento del possesso palla
  - Smoothing EMA per posizione pallone

## 📊 Output

Il sistema genera:

1. **Video Annotato**: Video con overlay che include:
   - Bounding box dei giocatori
   - Posizione del pallone
   - Keypoint del campo visualizzati
   - Vista tattica sovrapposta
   - Indicatore possesso palla

2. **File Stub**: Cache dei risultati per elaborazioni successive più veloci
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

## 🤝 Contribuire

Contributi, issues e feature requests sono benvenuti!

1. Fork del progetto
2. Crea il tuo Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al Branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📄 Licenza

Questo progetto è distribuito sotto licenza [specificare licenza].

Dataset di training: [basketball-player-detection-3](https://universe.roboflow.com/projects-wh5rm/basketball-player-detection-3-ycjdo-cffrq) - CC BY 4.0

---

## English Version

# Basketball Tracker CV 🏀

An advanced computer vision system for real-time basketball game tracking and analysis with tactical visualization.

## 📋 Table of Contents
- [Features](#features-1)
- [Requirements](#requirements-1)
- [Installation](#installation-1)
- [Usage](#usage-1)
- [Project Structure](#project-structure-1)
- [Models](#models-1)
- [Output](#output-1)
- [Troubleshooting](#troubleshooting-1)
- [Contributing](#contributing-1)
- [License](#license-1)

## ✨ Features

- **Player and Ball Detection**: Uses YOLO11 or RF-DETR models for accurate player and ball detection
- **Multi-Object Tracking**: Implements ByteTrack to track players across frames even with occlusions
- **Court Keypoint Detection**: Automatically identifies basketball court keypoints (lines, hoops, etc.)
- **Tactical View**: Transforms player positions into a 2D tactical court view
- **Smart Interpolation**: Automatically fills missing data for player and ball positions
- **Possession Calculation**: Determines which player has ball possession
- **Annotated Video**: Generates output video with detection overlays and tactical view

## 🔧 Requirements

### Software
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Python Dependencies
```
ultralytics==8.3.67
rfdetr
opencv-python
numpy
torch
supervision
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

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

### Available Parameters

#### Video Input/Output
- `--video`: Path to video to process (default: `video_1.mp4`)
- `--output-path`: Output video path (default: `outputVideo/output_video.mp4`)
- `--fps`: Output video FPS (default: `30.0`)

#### Models
- `--keypoint-model`: Keypoint detection model path (required)
- `--player-model`: Player detection model path (required)

#### Stub (Cache)
- `--keypoint-stub`: Keypoint stub path (default: `stubs/court_key_points_stub.pkl`)
- `--player-stub`: Player positions stub path (default: `stubs/players_positions_stub.pkl`)
- `--no-stub`: Disable stub reading and recalculate everything

#### Other
- `--court-image`: Tactical court image path (default: `./images/basketball_court.png`)
- `--debug`: Enable debug mode with frame visualization

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
  - Left and right court edges
  - Half-court line
  - Free throw lines
  - Court corners
  - Other characteristic points

### 2. Player and Ball Model
- **Type**: YOLO11 or RF-DETR
- **Detected classes**:
  - Class 0/1: Ball
  - Class 3/4: Player
- **Features**:
  - Multi-object tracking with ByteTrack
  - Ball possession detection
  - EMA smoothing for ball position

## 📊 Output

The system generates:

1. **Annotated Video**: Video with overlays including:
   - Player bounding boxes
   - Ball position
   - Visualized court keypoints
   - Overlaid tactical view
   - Ball possession indicator

2. **Stub Files**: Cached results for faster subsequent processing
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

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is distributed under [specify license].

Training dataset: [basketball-player-detection-3](https://universe.roboflow.com/projects-wh5rm/basketball-player-detection-3-ycjdo-cffrq) - CC BY 4.0

---

## 👥 Authors

- [@pgnsamu](https://github.com/pgnsamu)

## 🙏 Acknowledgments

- YOLO by Ultralytics
- RF-DETR detection framework
- Supervision library for tracking
- Roboflow for the training dataset
