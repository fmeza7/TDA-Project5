from __future__ import annotations

import numpy as np
from scipy import stats

def smooth_predictions(predictions: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Aplica un filtro de moda (majority vote) sobre una ventana deslizante
    para eliminar el 'jitter' o ruido temporal en la segmentación de acciones.
    
    Args:
        predictions: Arreglo 1D de etiquetas predichas por frame.
        window_size: Tamaño de la ventana de suavizado (debe ser impar).
        
    Returns:
        Arreglo 1D con las etiquetas suavizadas.
    """
    if len(predictions) < window_size:
        return predictions
        
    # Asegurar que la ventana sea impar para tener un centro claro
    if window_size % 2 == 0:
        window_size += 1
        
    pad_width = window_size // 2
    # Rellenar los bordes repitiendo el primer y último valor
    padded = np.pad(predictions, (pad_width, pad_width), mode='edge')
    
    smoothed = np.zeros_like(predictions)
    
    for i in range(len(predictions)):
        window = padded[i : i + window_size]
        # Calcular la moda en la ventana actual
        mode_val = stats.mode(window, keepdims=False)[0]
        smoothed[i] = mode_val
        
    return smoothed

def filter_short_segments(predictions: np.ndarray, min_frames: int = 10) -> np.ndarray:
    """
    Filtra segmentos de acciones que son demasiado cortos para ser reales,
    reemplazándolos por la clase del segmento adyacente más largo.
    """
    if len(predictions) == 0:
        return predictions

    filtered = np.copy(predictions)
    
    # Encontrar los índices donde cambia la clase
    changes = np.where(predictions[:-1] != predictions[1:])[0] + 1
    splits = np.split(predictions, changes)
    
    current_idx = 0
    for i, segment in enumerate(splits):
        length = len(segment)
        if length < min_frames:
            # Si es el primer segmento, usar la clase del siguiente
            if i == 0 and len(splits) > 1:
                replacement_class = splits[i+1][0]
            # Si es el último, usar la del anterior
            elif i == len(splits) - 1 and i > 0:
                replacement_class = splits[i-1][0]
            # Para los del medio, usar la clase del segmento vecino más largo
            elif i > 0 and i < len(splits) - 1:
                prev_len = len(splits[i-1])
                next_len = len(splits[i+1])
                if prev_len > next_len:
                    replacement_class = splits[i-1][0]
                else:
                    replacement_class = splits[i+1][0]
            else:
                replacement_class = segment[0]
                
            filtered[current_idx : current_idx + length] = replacement_class
            
        current_idx += length
        
    return filtered