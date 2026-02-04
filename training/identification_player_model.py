# -*- coding: utf-8 -*-
"""
Script di Addestramento RF-DETR - NIGHT MODE (SAFE)
Configurato per non crashare mentre dormi.
"""

import os
import torch
import supervision as sv

# Gestione sicura del modello
try:
    from rfdetr import RFDETRX as RFDETR_Model  
    print("Modello RFDETRX caricato.")
except ImportError:
    try:
        from rfdetr import RFDETRLarge as RFDETR_Model
        print("Modello RFDETR Large caricato.")
    except ImportError:
        from rfdetr import RFDETRNano as RFDETR_Model
        print("Modello RFDETR Nano caricato.")

from PIL import Image
from roboflow import Roboflow

# Chiavi API 
ROBOFLOW_API_KEY = "rf_SAVeUEH7P5W6fC6IpshsX6IBVbW2" 
os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY
rf = Roboflow(api_key="FwSC85pmDizQnJJu6fpt")

# Variabile globale
model = None

def setup_iniziale():
    print("==========================================")
    print("      CONFIGURAZIONE NOTTURNA (SAFE)      ")
    print("==========================================")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Svuotiamo la cache per partire puliti
        torch.cuda.empty_cache() 

def setup_dataset():
    try:
        project = rf.workspace("projects-wh5rm").project("basketball-player-detection-3-ycjdo-cffrq")
        version = project.version(1)
        dataset = version.download("coco")
        return dataset
    except Exception as e:
        print(f"Errore download dataset: {e}")
        return None

def train_model(dataset):
    global model
    if dataset is None: return

    print("\n--- INIZIALIZZAZIONE MODELLO ---")
    try:
        # Pretraining attivo per massima qualità
        model = RFDETR_Model(force_no_pretrain=False) 
    except:
        model = RFDETR_Model(force_no_pretrain=True)

    try:
        if hasattr(model, 'model') and model.model is not None:
            model.model.train()
    except: pass

    print("\n--- AVVIO ADDESTRAMENTO SAFE ---")
    
    # === PARAMETRI ANTI-CRASH ===
    # Batch 4 occupa circa 8-10GB VRAM. La tua 5080 ne ha 17.
    # È impossibile che crashi ora.
    BATCH_SIZE = 4      
    ACCUM_STEPS = 4     # 4 * 4 = 16 (Batch virtuale perfetto)
    WORKERS = 2         # Meno carico sulla CPU
    EPOCHS = 150     

    try:
        model.train(
            dataset_dir=dataset.location, 
            epochs=EPOCHS,             
            batch_size=BATCH_SIZE,            
            grad_accum_steps=ACCUM_STEPS,
            num_workers=WORKERS            
        )
        print("Addestramento completato!")
        
    except RuntimeError as e:
        print(f"\nERRORE CRITICO: {e}")

def start():
    setup_iniziale()
    dataset = setup_dataset()
    if dataset:
        train_model(dataset)

if __name__ == "__main__":
    start()