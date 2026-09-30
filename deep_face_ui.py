import cv2
import face_recognition
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import sys

class DeepFaceModel:
    """Motor de Deep Learning basado en redes neuronales convolucionales."""
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.known_encodings = []
        self.known_names = []

    def load_and_encode(self) -> None:
        """Extrae los vectores matemáticos de 128 dimensiones de las imágenes."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"La ruta no existe: {self.dataset_path}")

        valid_extensions = {'.jpg', '.jpeg', '.png'}
        processed_count = 0

        for file_name in os.listdir(self.dataset_path):
            if os.path.splitext(file_name)[1].lower() not in valid_extensions:
                continue

            image_path = os.path.join(self.dataset_path, file_name)
            try:
                # face_recognition maneja automáticamente la lectura de archivos
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append("Autorizado")
                    processed_count += 1
            except Exception as e:
                print(f"Error procesando {file_name}: {e}", file=sys.stderr)
        
        if processed_count == 0:
            raise ValueError("No se encontraron rostros válidos para codificar en el directorio.")
        
        print(f"Modelo DL cargado. Se procesaron {processed_count} firmas faciales.")

    def recognize_frame(self, frame):
        """Compara vectores faciales en tiempo real."""
        # Se reduce el tamaño del frame a 1/4 para acelerar el procesamiento de la IA
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # OpenCV usa BGR, face_recognition requiere RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 1. Localiza todos los rostros en la pantalla
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # 2. Extrae los 128 vectores de cada rostro encontrado
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_distances_list = []

        for face_encoding in face_encodings:
            # Tolerancia de 0.5 (Más estricto que el 0.6 por defecto) para evitar falsos positivos familiares
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
            name = "Desconocido"
            
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
            
            min_distance = 1.0
            if best_match_index != -1 and matches[best_match_index]:
                name = self.known_names[best_match_index]
                min_distance = face_distances[best_match_index]
            elif best_match_index != -1:
                min_distance = face_distances[best_match_index]

            face_names.append(name)
            face_distances_list.append(min_distance)

        # Renderizado gráfico
        for (top, right, bottom, left), name, dist in zip(face_locations, face_names, face_distances_list):
            # Escala nuevamente a la resolución original del video
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "Autorizado":
                color = (0, 255, 0)
                text = f"{name} ({dist:.2f})"
            else:
                color = (0, 0, 255)
                text = f"{name} ({dist:.2f})"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

class FaceAppUI:
    """Gestiona la interfaz gráfica y los hilos de cámara."""
    def __init__(self, window: tk.Tk):
        self.window = window
        self.window.title("Módulo de Identificación DL")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.window.withdraw()
        
        messagebox.showinfo("Configuración", "Selecciona el dataset de entrenamiento.")
        selected_directory = filedialog.askdirectory(title="Dataset")
        
        if not selected_directory:
            sys.exit(0)

        self.window.deiconify()

        try:
            self.model = DeepFaceModel(dataset_path=selected_directory)
            self.model.load_and_encode()
        except Exception as e:
            messagebox.showerror("Error del Motor DL", str(e))
            sys.exit(1)

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Cámara web no detectada.")
            sys.exit(1)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(padx=10, pady=10)

        self.update_video()

    def update_video(self) -> None:
        success, frame = self.video_capture.read()
        if success:
            processed_frame = self.model.recognize_frame(frame)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(15, self.update_video)

    def on_close(self) -> None:
        if hasattr(self, 'video_capture'):
            self.video_capture.release()
        self.window.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAppUI(root)
    root.mainloop()