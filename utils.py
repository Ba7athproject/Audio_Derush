def is_librosa_available():
    try:
        import librosa
        return True
    except ImportError:
        return False

def is_soundfile_available():
    try:
        import soundfile
        return True
    except ImportError:
        return False

def is_numpy_array(obj):
    try:
        import numpy as np
        return isinstance(obj, np.ndarray)
    except ImportError:
        return False

def is_torch_tensor(obj):
    try:
        import torch
        return isinstance(obj, torch.Tensor)
    except ImportError:
        return False

def requires_backends(func, backends):
    # Stub : cette fonction ne fait rien mais assure la compatibilité avec les checks du code principal
    pass
