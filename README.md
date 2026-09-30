# Reconocimiento Facial con Deep Learning (face_recognition)

Este proyecto es una implementación de un sistema de control de acceso biométrico basado en redes neuronales convolucionales. Desarrollado como parte de una evaluación de seguridad y rendimiento de algoritmos para el programa MEXMACS (UNIR).

El sistema extrae 128 vectores geométricos del rostro humano para autenticación en tiempo real, superando las vulnerabilidades de falsos positivos por consanguinidad presentes en modelos 2D tradicionales (como LBPH).

## Tecnologías Utilizadas
* **Python 3.14**
* **OpenCV (`opencv-python`)**: Captura y procesamiento de fotogramas en tiempo real.
* **dlib**: Motor matemático en C++ para análisis biométrico.
* **face_recognition**: Abstracción de red neuronal tipo ResNet para la extracción de los 128 encodings faciales.
* **Tkinter & Pillow**: Interfaz gráfica de usuario (GUI).

## Requisitos Previos
Debido a que `dlib` requiere compilación en entornos Windows, es indispensable tener instalado:
* [Visual Studio Build Tools](https://visualstudio.microsoft.com/es/visual-cpp-build-tools/) con la carga de trabajo "Desarrollo para el escritorio con C++".
* Git para Windows.

## Instalación

1. Clonar el repositorio:
```bash
git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
cd TU_REPOSITORIO

instalar las dependencias

pip install opencv-python numpy Pillow face_recognition
python -m pip install "setuptools<70.0.0"

ejecutar el script
python deep_face_ui.py

nota: Al iniciar, una ventana emergente solicitará seleccionar un directorio. Este directorio debe contener imágenes claras del rostro a autorizar (se recomiendan 30 imágenes para las pruebas de evaluación). El sistema procesará las firmas y encenderá la cámara web.