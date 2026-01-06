import importlib.metadata

# Lista delle librerie da controllare
libraries = [
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "pandas",
    "Pillow",
    "roboflow",
    "supervision",
    "torch",
    "transformers",
    "ultralytics"
]

print("Versioni delle librerie installate:\n")
print("-" * 50)

for lib in libraries:
    try:
        version = importlib.metadata.version(lib)
        print(f"{lib}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{lib}: NON INSTALLATA")

print("-" * 50)


'''
.venv  (funzionante)                             |       .venv2 (non funzionante si porta in giro i punti)
------------------------------------------------- -----------------------------------------------------------
numpy: 1.24.4                                       numpy: 2.2.6
opencv-python: 4.10.0.84                            opencv-python: 4.12.0.88                                    (*)
opencv-python-headless: 4.10.0.84                   opencv-python-headless: 4.10.0.84                           (*)     
pandas: 2.0.3                                       pandas: NON INSTALLATA
Pillow: 11.1.0                                      Pillow: 12.1.0
roboflow: 1.1.51                                    roboflow: 1.2.11
supervision: 0.25.1                                 supervision: 0.27.0                                         (*)
torch: 2.2.0                                        torch: 2.9.1
transformers: 4.46.3                                transformers: NON INSTALLATA
ultralytics: 8.3.67                                 ultralytics: 8.3.248                                        (*)                 
-------------------------------------------------- ----------------------------------------------------------
(*) librerie effettivamente utilizzate nel tracking dei keypoints


'''