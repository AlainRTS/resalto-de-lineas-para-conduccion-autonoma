# Importamos las librerías necesarias
import numpy as np  # Para manejo de arreglos numéricos
import matplotlib.pyplot as plt  # Para visualización de gráficos (histograma)
from matplotlib.widgets import Slider, Button  # No utilizado en este código, pero se importa por error
import cv2  # Librería de OpenCV para manipulación de imágenes y video
import os
# --- FUNCIONES AUXILIARES ---

def calculadora_histograma(frame, histograma_total):
    """
    Convierte el cuadro a escala de grises y calcula su histograma,
    sumándolo al histograma total acumulado.

    Args:
        frame (ndarray): Cuadro (frame) del video en color.
        histograma_total (ndarray): Histograma acumulado de los cuadros anteriores.

    Returns:
        ndarray: Histograma acumulado actualizado.
    """
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversión a escala de grises
    histograma_frame = cv2.calcHist([frame_gris], [0], None, [256], [0, 256])  # Histograma de 256 niveles
    histograma_total += histograma_frame  # Suma al histograma acumulado
    return histograma_total

def recorte(frame):
    """
    Recorta una región específica del cuadro.

    Args:
        frame (ndarray): Cuadro del video.

    Returns:
        ndarray: Cuadro recortado.
    """
    x1, y1 = 280, 400  # Coordenada superior izquierda del recorte
    x2, y2 = 1280, 720  # Coordenada inferior derecha del recorte
    return frame[y1:y2, x1:x2]  # Devolvemos la sección recortada

def union(frame, frame_recortado):
    """
    Inserta el cuadro recortado en un fondo negro del mismo tamaño que el frame original.

    Args:
        frame (ndarray): Cuadro original del video.
        frame_recortado (ndarray): Cuadro recortado.

    Returns:
        ndarray: Cuadro con el recorte insertado sobre un fondo negro.
    """
    x1, y1 = 280, 400  # Coordenada superior izquierda
    x2, y2 = 1280, 720  # Coordenada inferior derecha

    # Redimensionamos el recorte para encajar en la región de destino
    width = x2 - x1
    height = y2 - y1
    frame_recortado = cv2.resize(frame_recortado, (width, height))

    # Creamos un fondo negro del mismo tamaño que el frame original
    frame_negro = np.zeros_like(frame)
    
    # Colocamos el recorte en la región especificada
    frame_negro[y1:y2, x1:x2] = frame_recortado
    return frame_negro

def cargar_video(ruta_video, ruta_salida, procesar_frame):
    """
    Procesa un video cuadro por cuadro, aplicando una transformación específica
    y generando un nuevo video procesado.

    Args:
        ruta_video (str): Ruta del video de entrada.
        ruta_salida (str): Ruta donde se guardará el video procesado.
        procesar_frame (func): Función que procesa cada cuadro del video.

    Returns:
        ndarray: Histograma acumulado de todo el video.
    """
    cap = cv2.VideoCapture(ruta_video)  # Cargamos el video
    histograma_total = np.zeros((256, 1))  # Inicializamos el histograma acumulado

    # Obtenemos propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Configuramos el video de salida con el mismo tamaño y fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Códec de video
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))

    # Procesamos cada cuadro mientras el video esté abierto
    while cap.isOpened():
        ret, frame = cap.read()  # Leemos un cuadro
        if not ret:  # Si no hay más cuadros, salimos del bucle
            break

        # Aplicamos las transformaciones al cuadro
        frame_recortado = recorte(frame)
        frame_procesado = procesar_frame(frame_recortado)
        frame_unido = union(frame, frame_procesado)

        # Mostramos el cuadro procesado en una ventana
        cv2.imshow('Frame procesado', frame_unido)

        # Actualizamos el histograma acumulado y guardamos el cuadro procesado en el nuevo video
        histograma_total = calculadora_histograma(frame, histograma_total)
        out.write(frame_unido)

        # Salimos si el usuario presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberamos los recursos y cerramos ventanas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return histograma_total  # Devolvemos el histograma acumulado

def filtros(frame):
    """
    Aplica una corrección de gamma para ajustar el brillo del cuadro.

    Args:
        frame (ndarray): Cuadro del video.

    Returns:
        ndarray: Cuadro con corrección de gamma aplicada.
    """
    gamma = 2  # Valor de gamma
    return np.array(255 * (frame / 255) ** gamma, dtype="uint8")  # Ajuste de gamma

def detectar_lineas(frame_procesado):
    """
    Detecta líneas amarillas y blancas en el cuadro procesado.

    Args:
        frame_procesado (ndarray): Cuadro procesado.

    Returns:
        ndarray: Máscara con las líneas detectadas.
    """
    hsv = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2HSV)  # Conversión a HSV
    # Rango de colores para amarillo y blanco
    lower_yellow = np.array([15, 125, 140])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Creamos máscaras para los colores
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    return cv2.bitwise_or(mask_yellow, mask_white)  # Combinamos las máscaras

def aplicar_fondo_negro(frame_procesado, mask_lineas):
    """
    Resalta las líneas detectadas sobre un fondo negro.

    Args:
        frame_procesado (ndarray): Cuadro procesado.
        mask_lineas (ndarray): Máscara con las líneas detectadas.

    Returns:
        ndarray: Cuadro con las líneas resaltadas.
    """
    resultado = cv2.bitwise_and(frame_procesado, frame_procesado, mask=mask_lineas)  # Aplicamos la máscara
    fondo_negro = np.zeros_like(frame_procesado)  # Creamos un fondo negro
    return cv2.add(fondo_negro, resultado)  # Combinamos las líneas con el fondo negro

def procesar_frame(frame):
    """
    Aplica todos los filtros y procesamiento al cuadro.

    Args:
        frame (ndarray): Cuadro del video.

    Returns:
        ndarray: Cuadro con las líneas resaltadas.
    """
    frame_procesado = filtros(frame)
    mask_lineas = detectar_lineas(frame_procesado)
    return aplicar_fondo_negro(frame_procesado, mask_lineas)

def mostrar_histograma(histograma_total):
    """
    Muestra el histograma acumulado del video.

    Args:
        histograma_total (ndarray): Histograma acumulado de todos los cuadros.
    """
    plt.figure()
    plt.title('Histograma de todo el video')
    plt.xlabel('Intensidad de píxeles')
    plt.ylabel('Frecuencia')
    plt.plot(histograma_total, color='black')
    plt.xlim([0, 256])
    plt.show()

# --- BLOQUE PRINCIPAL ---

if __name__ == "__main__":
    ruta_video = os.path.join(os.path.dirname(__file__), 'lineas.mp4')  # Ruta del video de entrada
    ruta_salida = 'lineas_procesado.mp4'  # Ruta del video de salida

    # Procesamos el video y mostramos el histograma si fue exitoso
    histograma_total = cargar_video(ruta_video, ruta_salida, procesar_frame)
    if histograma_total is not None:
        mostrar_histograma(histograma_total)
    else:
        print("No se pudo procesar el video.")
