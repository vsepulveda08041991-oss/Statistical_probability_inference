import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.special import logsumexp
from scipy.fft import fft
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Intentar importar TensorFlow (opcional)
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    HAS_TENSORFLOW = True
    print("TensorFlow disponible")
except ImportError:
    print("TensorFlow NO disponible")

# Para validación cruzada temporal y tuning de hiperparámetros
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, ParameterSampler
from sklearn.metrics import accuracy_score
import time

print("Iniciando prueba simple...")

# Crear datos de prueba simples
np.random.seed(42)
n_samples = 100
data = {
    'sorteo': range(1, n_samples+1),
    'n1': np.random.randint(1, 43, n_samples),
    'n2': np.random.randint(1, 43, n_samples),
    'n3': np.random.randint(1, 43, n_samples),
    'n4': np.random.randint(1, 43, n_samples),
    'n5': np.random.randint(1, 43, n_samples),
    'balota': np.random.randint(1, 17, n_samples)
}
df = pd.DataFrame(data)
print(f"Datos creados: {df.shape}")

# Probar la función de carga de datos simple
print("Prueba completada exitosamente")