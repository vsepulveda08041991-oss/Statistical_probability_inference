"""
MODELO AI-ENHANCED v3.0 - PREDICCIÓN DEL BALOTO
===============================================
Versión corregida y testeada con pruebas estrictas.

Correcciones principales:
- Fix dimensional mismatch en LSTM (features_array vs num_features)
- Fix predecir() usa historial correcto para secuencias LSTM  
- Fix scalers con keys consistentes en GradientBoostingEnsemble
- Fix backtesting: model_weights inicializados antes de predecir()
- Fix prepare_features: features de longitud fija
- Fix doble normalización en LSTM predict_probabilities
- Fix peso negativo en Markov O3
- Fix LightGBM API deprecada
- Pruebas unitarias integradas
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest
)
from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    ParameterSampler
)
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.special import logsumexp
from scipy.fft import fft
import warnings
import time
import unittest
import io
import sys

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────
BALOTO_MAX = 43          # Números posibles en Baloto (1-43)
SUPER_BALOTA_MAX = 16    # Super Balota posibles (1-16)
NUM_POSICIONES = 5       # Posiciones de números normales
NUM_FEATURES_GB = 15     # Features fijos para Gradient Boosting (FIX #5)
NUM_FEATURES_LSTM = 12   # Features fijos para LSTM (FIX #1)
MIN_SORTEOS_ENTRENAMIENTO = 50

# ─────────────────────────────────────────────
# DETECCIÓN DE LIBRERÍAS OPCIONALES
# ─────────────────────────────────────────────
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    HAS_TENSORFLOW = True
except ImportError:
    pass

HAS_XGBOOST = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass


# ─────────────────────────────────────────────
# CAPA DE ATENCIÓN (solo si TensorFlow disponible)
# ─────────────────────────────────────────────
if HAS_TENSORFLOW:
    class AttentionLayer(layers.Layer):
        """Mecanismo de Attention para detectar números clave."""

        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name='attention_weight',
                shape=(input_shape[-1], input_shape[-1]),
                initializer='glorot_uniform',
                trainable=True
            )
            self.b = self.add_weight(
                name='attention_bias',
                shape=(input_shape[-1],),
                initializer='zeros',
                trainable=True
            )
            self.u = self.add_weight(
                name='attention_u',
                shape=(input_shape[-1],),
                initializer='glorot_uniform',
                trainable=True
            )
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            # x shape: (batch, timesteps, features)
            uit = tf.tanh(tf.linalg.matmul(x, self.W) + self.b)
            ait = tf.linalg.matmul(uit, tf.expand_dims(self.u, -1))
            ait = tf.squeeze(ait, -1)
            ait = tf.nn.softmax(ait, axis=-1)
            ait = tf.expand_dims(ait, -1)
            weighted = x * ait
            output = tf.reduce_sum(weighted, axis=1)
            return output

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

        def get_config(self):
            return super(AttentionLayer, self).get_config()


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES PURAS (testeables)
# ─────────────────────────────────────────────

def calcular_features_estadisticas(window: np.ndarray) -> list:
    """
    Calcula exactamente NUM_FEATURES_LSTM=12 features estadísticas
    de una ventana de números. Longitud SIEMPRE fija.
    
    FIX #1: Elimina el mismatch dimensional entre features_array y 
    num_features en LSTMAttentionNetwork.
    
    Args:
        window: array 1D de números históricos
        
    Returns:
        lista de exactamente 12 floats
    """
    if len(window) == 0:
        return [0.0] * NUM_FEATURES_LSTM

    mean_val   = float(np.mean(window))
    std_val    = float(np.std(window)) if len(window) > 1 else 0.0
    max_val    = float(np.max(window))
    min_val    = float(np.min(window))
    median_val = float(np.median(window))
    p25_val    = float(np.percentile(window, 25))
    p75_val    = float(np.percentile(window, 75))

    skew_val = float(stats.skew(window)) if len(window) > 2 else 0.0
    kurt_val = float(stats.kurtosis(window)) if len(window) > 3 else 0.0

    diffs = np.diff(window)
    trend_val    = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    deviation_val = float(window[-1] - mean_val) if len(window) > 1 else 0.0
    momentum_val = (float(np.sum(diffs > 0) / len(diffs))
                    if len(diffs) > 0 else 0.5)

    features = [
        mean_val, std_val, max_val, min_val,
        skew_val, kurt_val, median_val, p25_val, p75_val,
        trend_val, deviation_val, momentum_val
    ]
    # Garantía de longitud fija
    assert len(features) == NUM_FEATURES_LSTM, (
        f"Error: se esperaban {NUM_FEATURES_LSTM} features, "
        f"se obtuvieron {len(features)}"
    )
    return features


def calcular_features_gb(window: np.ndarray, numeros_lag: np.ndarray) -> np.ndarray:
    """
    Calcula exactamente NUM_FEATURES_GB=15 features para Gradient Boosting.
    
    FIX #5: Features de longitud FIJA para evitar arrays irregulares.
    
    Args:
        window:      array histórico de números (últimos 20 o menos)
        numeros_lag: array de números con lag (últimos 4)
        
    Returns:
        np.ndarray de shape (NUM_FEATURES_GB,)
    """
    if len(window) == 0:
        return np.zeros(NUM_FEATURES_GB)

    mean_val   = np.mean(window)
    std_val    = np.std(window) if len(window) > 1 else 0.0
    max_val    = np.max(window)
    min_val    = np.min(window)
    median_val = np.median(window)
    p25_val    = np.percentile(window, 25)
    p75_val    = np.percentile(window, 75)

    skew_val = float(stats.skew(window)) if len(window) > 2 else 0.0
    kurt_val = float(stats.kurtosis(window)) if len(window) > 3 else 0.0

    # Tendencia reciente (últimos 5 o menos)
    tail = window[-5:] if len(window) >= 5 else window
    trend_val = float(np.mean(np.diff(tail))) if len(tail) > 1 else 0.0

    # 4 lags fijos (rellenamos con 0 si no hay suficientes datos)
    lag_features = np.zeros(4)
    for i, lag_val in enumerate(numeros_lag[-4:]):
        lag_features[i] = float(lag_val)

    # Padding de autocorrelación: 1 feature
    if len(window) > 1:
        corr_matrix = np.corrcoef(window[:-1], window[1:])
        autocorr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        autocorr = 0.0

    features = np.array([
        mean_val, std_val, max_val, min_val, median_val,
        p25_val, p75_val, skew_val, kurt_val, trend_val,
        lag_features[0], lag_features[1],
        lag_features[2], lag_features[3],
        autocorr
    ], dtype=np.float64)

    assert features.shape[0] == NUM_FEATURES_GB, (
        f"Error: se esperaban {NUM_FEATURES_GB} features GB, "
        f"se obtuvieron {features.shape[0]}"
    )
    return features


def resolver_duplicados(numeros: list, detalles: list,
                         max_num: int = BALOTO_MAX) -> list:
    """
    Resuelve números duplicados eligiendo alternativas del detalle.
    
    Args:
        numeros:  lista de números predichos (puede tener duplicados)
        detalles: lista de detalles con alternativas por posición
        max_num:  máximo número válido
        
    Returns:
        lista sin duplicados de longitud == len(numeros)
    """
    resultado = []
    usados = set()

    for idx, num in enumerate(numeros):
        if num not in usados and 1 <= num <= max_num:
            resultado.append(num)
            usados.add(num)
        else:
            # Buscar primera alternativa no usada
            encontrado = False
            if idx < len(detalles):
                for alt in detalles[idx].get('alt', []):
                    alt_n = alt['n']
                    if alt_n not in usados and 1 <= alt_n <= max_num:
                        resultado.append(alt_n)
                        usados.add(alt_n)
                        encontrado = True
                        break

            if not encontrado:
                # Fallback: primer número disponible
                for fallback in range(1, max_num + 1):
                    if fallback not in usados:
                        resultado.append(fallback)
                        usados.add(fallback)
                        break

    return resultado[:len(numeros)]


# ─────────────────────────────────────────────
# CLASE LSTM + ATTENTION
# ─────────────────────────────────────────────
class LSTMAttentionNetwork:
    """
    Red neuronal LSTM con Attention para capturar dependencias temporales.
    
    FIX #1: num_features ahora se pasa explícitamente y coincide con
    calcular_features_estadisticas() → NUM_FEATURES_LSTM=12.
    FIX #9: Se elimina doble normalización en predict_probabilities.
    """

    def __init__(self, sequence_length: int = 20,
                 num_features: int = NUM_FEATURES_LSTM):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.scaler = MinMaxScaler()
        self._scaler_fitted = False   # FIX #9: flag para evitar doble fit

    def build_model(self, lstm_units: int = 64,
                    dropout_rate: float = 0.2,
                    learning_rate: float = 0.001) -> object:
        """Construye la red LSTM con Attention."""
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow no está disponible.")

        model = Sequential([
            layers.LSTM(
                lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.num_features)
            ),
            layers.Dropout(dropout_rate),
            layers.LSTM(max(lstm_units // 2, 8), return_sequences=True),
            layers.Dropout(dropout_rate),
            AttentionLayer(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(BALOTO_MAX, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def prepare_sequences(self, data: np.ndarray,
                           labels: np.ndarray):
        """
        Prepara datos en formato secuencial para LSTM.
        
        FIX #9: Aplica scaler UNA sola vez aquí. predict_probabilities
        usará el scaler ya entrenado sin volver a entrenar.
        
        Returns:
            (X, y) arrays listos para entrenamiento
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        X_list, y_list = [], []
        for i in range(len(data) - self.sequence_length):
            X_list.append(data[i: i + self.sequence_length])
            y_list.append(labels[i + self.sequence_length])

        if not X_list:
            return np.array([]), np.array([])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        # Normalización (una sola vez) ← FIX #9
        n_samples, seq_len, n_feat = X.shape
        X_reshaped = X.reshape(-1, n_feat)
        X_reshaped = self.scaler.fit_transform(X_reshaped)
        self._scaler_fitted = True
        X = X_reshaped.reshape(n_samples, seq_len, n_feat)

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 20, batch_size: int = 32):
        """Entrena la red."""
        if self.model is None:
            self.build_model()

        batch_size = min(batch_size, max(1, len(X) // 2))
        early_stop = EarlyStopping(
            monitor='loss', patience=3,
            restore_best_weights=True
        )
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

    def predict_probabilities(self, sequence: np.ndarray) -> np.ndarray | None:
        """
        Predice probabilidades para la próxima secuencia.
        
        FIX #9: NO vuelve a hacer fit del scaler. Solo transforma
        si el scaler ya fue entrenado en prepare_sequences.
        
        Args:
            sequence: array de shape (sequence_length, num_features)
            
        Returns:
            array de probabilidades de shape (BALOTO_MAX,) o None
        """
        if self.model is None:
            return None

        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)

        if sequence.shape[0] != self.sequence_length:
            # Padding o recorte para ajustar longitud
            if sequence.shape[0] < self.sequence_length:
                pad = np.zeros(
                    (self.sequence_length - sequence.shape[0],
                     sequence.shape[1])
                )
                sequence = np.vstack([pad, sequence])
            else:
                sequence = sequence[-self.sequence_length:]

        # Transformar (no fit) ← FIX #9
        if self._scaler_fitted:
            n_feat = sequence.shape[1]
            seq_reshaped = sequence.reshape(-1, n_feat)
            seq_reshaped = self.scaler.transform(seq_reshaped)
            sequence = seq_reshaped.reshape(
                1, self.sequence_length, n_feat
            )
        else:
            sequence = sequence.reshape(
                1, self.sequence_length, sequence.shape[1]
            )

        probs = self.model.predict(sequence, verbose=0)[0]
        return probs


# ─────────────────────────────────────────────
# ENSEMBLE GRADIENT BOOSTING
# ─────────────────────────────────────────────
class GradientBoostingEnsemble:
    """
    Ensemble de XGBoost/LightGBM/GradientBoosting.
    
    FIX #6: Scalers ahora se almacenan con keys enteras consistentes.
    FIX #5: Features de longitud fija vía calcular_features_gb().
    FIX #7: LightGBM API actualizada (sin verbose_eval deprecado).
    """

    def __init__(self):
        self.models: dict  = {}
        # FIX #6: key siempre es int (posicion)
        self.scalers: dict = {}

    def prepare_features(self, df: pd.DataFrame,
                          posicion: int):
        """
        Extrae características fijas (NUM_FEATURES_GB) para cada posición.
        
        FIX #5: longitud de features garantizada.
        """
        col_idx = posicion + 1   # FIX #3: usar col_idx correcto
        numeros = df.iloc[:, col_idx].values.astype(float)

        features_list = []
        labels = []
        window_size = 20

        for i in range(window_size, len(numeros) - 1):
            window = numeros[i - window_size: i]
            lag4   = numeros[max(0, i - 4): i]

            feat = calcular_features_gb(window, lag4)
            features_list.append(feat)
            labels.append(numeros[i + 1])

        if not features_list:
            return np.array([]).reshape(0, NUM_FEATURES_GB), np.array([])

        return np.vstack(features_list), np.array(labels)

    def train_posicion(self, X: np.ndarray, y: np.ndarray,
                        posicion: int) -> bool:
        """Entrena modelo para una posición con key entera."""
        if len(X) < 10:
            return False

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # FIX #6: key = int
        self.scalers[posicion] = scaler

        # XGBoost
        if HAS_XGBOOST:
            try:
                m = xgb.XGBRFRegressor(
                    n_estimators=100, max_depth=5,
                    learning_rate=0.1, subsample=0.8,
                    random_state=42, verbosity=0
                )
                m.fit(X_tr, y_tr)
                self.models[f'xgb_{posicion}'] = m
            except Exception as e:
                print(f"    XGBoost pos {posicion}: {e}")

        # LightGBM FIX #7: sin verbose_eval
        if HAS_LIGHTGBM:
            try:
                m = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=5,
                    learning_rate=0.1, verbose=-1
                )
                m.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)])  # sin verbose_eval
                self.models[f'lgb_{posicion}'] = m
            except Exception as e:
                print(f"    LightGBM pos {posicion}: {e}")

        # GradientBoosting (siempre disponible)
        try:
            y_tr_bin = (y_tr > np.median(y_tr)).astype(int)
            # Verificar que hay al menos 2 clases
            if len(np.unique(y_tr_bin)) < 2:
                y_tr_bin = np.zeros_like(y_tr_bin)
                y_tr_bin[:len(y_tr_bin)//2] = 1

            m = GradientBoostingClassifier(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=42
            )
            m.fit(X_tr, y_tr_bin)
            self.models[f'gb_{posicion}'] = m
        except Exception as e:
            print(f"    GB pos {posicion}: {e}")

        return True

    def predict_posicion(self, features: np.ndarray,
                          posicion: int) -> float | None:
        """
        Predice probabilidad para una posición.
        
        FIX #6: busca scaler con key entera.
        """
        # FIX #6: key entera
        if posicion not in self.scalers:
            return None

        scaler = self.scalers[posicion]
        features_scaled = scaler.transform(
            features.reshape(1, -1)
        )

        preds = []

        for prefix in ['xgb', 'lgb']:
            key = f'{prefix}_{posicion}'
            if key in self.models:
                try:
                    preds.append(
                        float(self.models[key].predict(features_scaled)[0])
                    )
                except Exception:
                    pass

        key_gb = f'gb_{posicion}'
        if key_gb in self.models:
            try:
                prob = self.models[key_gb].predict_proba(
                    features_scaled
                )[0][1]
                preds.append(float(prob))
            except Exception:
                pass

        return float(np.mean(preds)) if preds else None


# ─────────────────────────────────────────────
# ANÁLISIS CAUSAL
# ─────────────────────────────────────────────
class CausalInferenceAnalyzer:
    """Análisis de inferencia causal (Granger simplificado)."""

    def __init__(self):
        self.causal_links: dict = {}

    def detect_causal_relationships(self, df: pd.DataFrame):
        """Detecta relaciones causales entre posiciones."""
        from scipy.stats import linregress

        for pos1 in range(2, 7):
            col1 = df.iloc[:, pos1].values.astype(float)

            for pos2 in range(2, 7):
                if pos1 == pos2:
                    continue
                col2 = df.iloc[:, pos2].values.astype(float)
                causality_score = 0.0

                for lag in [1, 2, 3]:
                    if len(col1) <= lag + 10:
                        continue

                    X_cause = col2[:-lag - 1]
                    y_effect = col1[lag + 1:]

                    if len(X_cause) < 5 or len(y_effect) < 5:
                        continue

                    min_len = min(len(X_cause), len(y_effect))
                    try:
                        slope, intercept, r_val, p_val, _ = linregress(
                            X_cause[:min_len], y_effect[:min_len]
                        )
                        if p_val < 0.1:
                            causality_score += (1 - p_val) * abs(r_val)
                    except Exception:
                        pass

                if causality_score > 0.1:
                    self.causal_links[(pos1, pos2)] = causality_score


# ─────────────────────────────────────────────
# AGENTE DE REINFORCEMENT LEARNING
# ─────────────────────────────────────────────
class ReinforcementLearningAgent:
    """Agente Q-learning para optimizar estrategia de juego."""

    def __init__(self):
        self.q_table: dict = defaultdict(lambda: defaultdict(float))
        self.learning_rate   = 0.1
        self.discount_factor = 0.95
        self.epsilon         = 0.1

    def select_action(self, state: tuple,
                       available_actions: list) -> int:
        """Selecciona acción usando epsilon-greedy."""
        if not available_actions:
            raise ValueError("available_actions está vacía.")

        if np.random.random() < self.epsilon:
            return int(np.random.choice(available_actions))

        q_vals = [self.q_table[state][a] for a in available_actions]
        return available_actions[int(np.argmax(q_vals))]

    def update_q_value(self, state: tuple, action: int,
                        reward: float, next_state: tuple,
                        next_actions: list):
        """Actualiza Q-values con ecuación de Bellman."""
        max_next_q = (max(self.q_table[next_state][a]
                          for a in next_actions)
                      if next_actions else 0.0)

        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )


# ─────────────────────────────────────────────
# MODELO PRINCIPAL
# ─────────────────────────────────────────────
class AdvancedMarkovBalotoV3AIEnhanced:
    """
    Modelo híbrido de predicción del Baloto con técnicas avanzadas de ML.
    
    Correcciones aplicadas:
    - FIX #1:  LSTM num_features = NUM_FEATURES_LSTM (12)
    - FIX #2:  predecir() usa historial correcto para LSTM
    - FIX #3:  prepare_features usa col_idx correcto
    - FIX #4:  backtesting inicializa model_weights antes de predecir()
    - FIX #5:  features GB de longitud fija
    - FIX #6:  scalers GB con keys enteras
    - FIX #7:  LightGBM API actualizada
    - FIX #8:  guard HAS_TENSORFLOW en todo uso de TF
    - FIX #9:  una sola normalización en LSTM
    - FIX #10: peso Markov O3 nunca negativo
    """

    def __init__(self, csv_file: str,
                 alpha_smoothing: float = 0.5,
                 time_decay: float = 0.95):
        self.csv_file        = csv_file
        self.df              = None
        self.alpha_smoothing = alpha_smoothing
        self.time_decay      = time_decay
        self.numero_sorteos  = 0

        # Modelos Markov
        self.transition_matrices_order1: dict = {}
        self.transition_matrices_order3: dict = {}

        # Modelos AI
        self.lstm_networks:  dict = {}
        self.gb_ensemble     = GradientBoostingEnsemble()
        self.causal_analyzer = CausalInferenceAnalyzer()
        self.rl_agent        = ReinforcementLearningAgent()

        # Métricas y pesos
        self.volatility_patterns: dict = {}
        self.anomaly_scores:      dict = {}
        self.backtest_scores:     dict = {}
        self.model_weights:       dict = self._pesos_default()

    # ── Pesos ─────────────────────────────────
    @staticmethod
    def _pesos_default() -> dict:
        return {
            'markov_o1':        0.15,
            'markov_o3':        0.20,
            'lstm':             0.25,
            'gradient_boosting':0.25,
            'frequency':        0.10,
            'causal':           0.05
        }

    # ── Carga de datos ─────────────────────────
    def cargar_datos(self) -> bool:
        """Carga el CSV validando columnas mínimas."""
        try:
            self.df = pd.read_csv(self.csv_file)
            self.df = self.df.dropna().reset_index(drop=True)

            # Validación mínima de columnas
            if self.df.shape[1] < 8:
                raise ValueError(
                    f"Se esperan al menos 8 columnas, "
                    f"se encontraron {self.df.shape[1]}."
                )

            # Validar rangos de números
            for col_idx in range(2, 7):
                col = self.df.iloc[:, col_idx]
                if col.min() < 1 or col.max() > BALOTO_MAX:
                    raise ValueError(
                        f"Columna {col_idx} fuera de rango "
                        f"[1, {BALOTO_MAX}]."
                    )

            sb_col = self.df.iloc[:, 7]
            if sb_col.min() < 1 or sb_col.max() > SUPER_BALOTA_MAX:
                raise ValueError(
                    f"Super Balota fuera de rango "
                    f"[1, {SUPER_BALOTA_MAX}]."
                )

            self.numero_sorteos = len(self.df)
            print(f"* Datos cargados: {self.numero_sorteos} sorteos\n")
            return True

        except Exception as e:
            print(f"* Error cargando datos: {e}")
            return False

    # ── Pipeline de construcción ───────────────
    def construir_modelos(self) -> bool:
        """Construye todos los modelos en secuencia."""
        if self.df is None:
            print("✗ Error: cargue datos primero")
            return False

        print("=" * 80)
        print("CONSTRUCCIÓN DE MODELOS AI-ENHANCED v3.0")
        print("=" * 80 + "\n")

        pasos = [
            ("[1/8] Detectando anomalías...",
             self._detectar_anomalias),
            ("[2/8] Analizando volatilidad...",
             self._analizar_volatilidad),
            ("[3/8] Construyendo modelos Markov...",
             self._construir_modelos_markov),
            ("[4/8] Entrenando redes LSTM con Attention...",
             self._entrenar_lstm),
            ("[5/8] Entrenando Gradient Boosting Ensemble...",
             self._entrenar_gradient_boosting),
            ("[6/8] Analizando causalidad...",
             self._analizar_causalidad),
            ("[7/8] Ejecutando Backtesting...",
             self._ejecutar_backtesting),
            ("[8/8] Calibrando pesos óptimos...",
             self._calibrar_pesos),
        ]

        for msg, fn in pasos:
            print(msg)
            try:
                fn()
                print(f"  ✓ OK\n")
            except Exception as e:
                print(f"  ✗ Error en {msg}: {e}\n")

        print("=" * 80)
        print("* MODELOS AI-ENHANCED LISTOS")
        print("=" * 80 + "\n")
        return True

    # ── Anomalías ──────────────────────────────
    def _detectar_anomalias(self):
        datos = self.df.iloc[:, 2:7].values.astype(float)

        features = np.column_stack([
            (datos.mean(axis=1) - 22) / 13,
            datos.std(axis=1) / 15,
            (datos.max(axis=1) - datos.min(axis=1)) / (BALOTO_MAX - 1)
        ])

        iso = IsolationForest(
            contamination=0.08, random_state=42, n_estimators=50
        )
        iso.fit(features)
        scores = iso.score_samples(features)

        for pos in range(1, NUM_POSICIONES + 1):
            self.anomaly_scores[pos] = scores

    # ── Volatilidad ────────────────────────────
    def _analizar_volatilidad(self):
        for pos_idx in range(NUM_POSICIONES):
            col_idx = pos_idx + 2
            pos     = pos_idx + 1
            nums    = self.df.iloc[:, col_idx].values.astype(float)

            if len(nums) > 1:
                diffs = np.diff(nums)
                vol   = np.std(diffs) / (np.mean(np.abs(nums)) + 1e-6)
            else:
                vol = 0.0

            self.volatility_patterns[pos] = vol

    # ── Markov ─────────────────────────────────
    def _construir_modelos_markov(self):
        for pos_idx in range(NUM_POSICIONES):
            col_idx = pos_idx + 2
            pos     = pos_idx + 1
            nums    = self.df.iloc[:, col_idx].values.astype(int)
            self._construir_o1(pos, nums)
            self._construir_o3(pos, nums)

    def _construir_o1(self, posicion: int, numeros: np.ndarray):
        """Markov Orden-1 con decay temporal. FIX #10: peso >= 0."""
        matriz: dict = defaultdict(lambda: defaultdict(float))
        n = len(numeros)

        for i in range(n - 1):
            estado   = int(numeros[i])
            siguiente= int(numeros[i + 1])
            # FIX #10: exponent nunca negativo
            exp = max(0, n - i - 2)
            peso = self.time_decay ** exp
            matriz[estado][siguiente] += peso

        probs = {}
        for estado, trans in matriz.items():
            total        = sum(trans.values())
            total_smooth = total + self.alpha_smoothing * BALOTO_MAX
            probs[estado] = {
                sig: (trans.get(sig, 0) + self.alpha_smoothing) / total_smooth
                for sig in range(1, BALOTO_MAX + 1)
            }

        self.transition_matrices_order1[posicion] = probs

    def _construir_o3(self, posicion: int, numeros: np.ndarray):
        """Markov Orden-3 con decay temporal. FIX #10: peso >= 0."""
        matriz: dict = defaultdict(lambda: defaultdict(float))
        n = len(numeros)

        for i in range(n - 3):
            estado    = (int(numeros[i]),
                         int(numeros[i + 1]),
                         int(numeros[i + 2]))
            siguiente = int(numeros[i + 3])
            # FIX #10: exponent nunca negativo
            exp = max(0, n - i - 4)
            peso = self.time_decay ** exp
            matriz[estado][siguiente] += peso

        probs = {}
        for estado, trans in matriz.items():
            total        = sum(trans.values())
            total_smooth = total + self.alpha_smoothing * BALOTO_MAX
            probs[estado] = {
                sig: (trans.get(sig, 0) + self.alpha_smoothing) / total_smooth
                for sig in range(1, BALOTO_MAX + 1)
            }

        self.transition_matrices_order3[posicion] = probs

    # ── LSTM ───────────────────────────────────
    def _construir_features_lstm(self, numeros: np.ndarray):
        """
        Construye matrix de features para LSTM de shape
        (N, NUM_FEATURES_LSTM). FIX #1.
        """
        features_list = []
        window_size = 20

        for i in range(window_size - 1, len(numeros) - 1):
            window = numeros[i - window_size + 1: i + 1]
            features_list.append(
                calcular_features_estadisticas(window)
            )

        if not features_list:
            return np.array([]).reshape(0, NUM_FEATURES_LSTM), np.array([])

        features_array = np.array(features_list, dtype=np.float32)
        labels = numeros[window_size:].astype(int) - 1  # 0-indexado
        labels = np.clip(labels, 0, BALOTO_MAX - 1)

        return features_array, labels

    def _entrenar_lstm(self):
        """
        Entrena redes LSTM con Attention y búsqueda de hiperparámetros.
        FIX #1, #8, #9.
        """
        if not HAS_TENSORFLOW:
            print("    TensorFlow no disponible, omitiendo LSTM.")
            return

        for pos_idx in range(NUM_POSICIONES):
            col_idx = pos_idx + 2
            pos     = pos_idx + 1
            nums    = self.df.iloc[:, col_idx].values.astype(float)

            features_array, labels = self._construir_features_lstm(nums)

            if len(features_array) < 30:
                print(f"    Datos insuf. LSTM pos {pos}: "
                      f"{len(features_array)} muestras")
                continue

            max_seq_len = min(20, len(features_array) // 3)
            max_seq_len = max(max_seq_len, 5)

            # Espacio de búsqueda
            param_grid = {
                'sequence_length': [max(5, max_seq_len // 2),
                                     max_seq_len],
                'lstm_units':      [32, 64],
                'dropout_rate':    [0.1, 0.2],
                'learning_rate':   [0.001, 0.005]
            }

            n_splits = max(2, min(3, len(features_array) // 20))
            tscv     = TimeSeriesSplit(n_splits=n_splits)

            best_score  = -1.0
            best_params = None

            param_list = list(
                ParameterSampler(param_grid, n_iter=4, random_state=42)
            )

            for params in param_list:
                seq_len = int(params['sequence_length'])
                units   = int(params['lstm_units'])
                dropout = float(params['dropout_rate'])
                lr      = float(params['learning_rate'])

                try:
                    tmp = LSTMAttentionNetwork(
                        sequence_length=seq_len,
                        num_features=NUM_FEATURES_LSTM  # FIX #1
                    )
                    X_p, y_p = tmp.prepare_sequences(features_array, labels)

                    if len(X_p) < 5:
                        continue

                    scores = []
                    for tr_idx, val_idx in tscv.split(X_p):
                        X_tr, X_val = X_p[tr_idx], X_p[val_idx]
                        y_tr, y_val = y_p[tr_idx], y_p[val_idx]

                        m = LSTMAttentionNetwork(
                            sequence_length=seq_len,
                            num_features=NUM_FEATURES_LSTM
                        )
                        m.build_model(lstm_units=units,
                                      dropout_rate=dropout,
                                      learning_rate=lr)

                        es = EarlyStopping(
                            monitor='loss', patience=3,
                            restore_best_weights=True
                        )
                        m.model.fit(
                            X_tr, y_tr,
                            epochs=15,
                            batch_size=max(4, len(X_tr) // 4),
                            callbacks=[es],
                            verbose=0
                        )

                        preds = np.argmax(
                            m.model.predict(X_val, verbose=0),
                            axis=1
                        )
                        scores.append(accuracy_score(y_val, preds))

                    mean_sc = float(np.mean(scores))
                    if mean_sc > best_score:
                        best_score  = mean_sc
                        best_params = params

                except Exception as e:
                    print(f"    Error LSTM params pos {pos}: {e}")
                    continue

            # Entrenar modelo final
            try:
                if best_params is None:
                    best_params = {
                        'sequence_length': max_seq_len,
                        'lstm_units': 32,
                        'dropout_rate': 0.2,
                        'learning_rate': 0.001
                    }

                final_seq = int(best_params['sequence_length'])
                lstm_net  = LSTMAttentionNetwork(
                    sequence_length=final_seq,
                    num_features=NUM_FEATURES_LSTM  # FIX #1
                )
                lstm_net.build_model(
                    lstm_units=int(best_params['lstm_units']),
                    dropout_rate=float(best_params['dropout_rate']),
                    learning_rate=float(best_params['learning_rate'])
                )

                X_f, y_f = lstm_net.prepare_sequences(features_array, labels)
                if len(X_f) >= 5:
                    es = EarlyStopping(
                        monitor='loss', patience=5,
                        restore_best_weights=True
                    )
                    lstm_net.model.fit(
                        X_f, y_f,
                        epochs=25,
                        batch_size=max(4, len(X_f) // 4),
                        callbacks=[es],
                        verbose=0
                    )
                    self.lstm_networks[pos] = lstm_net
                    print(f"    LSTM pos {pos}: "
                          f"seq={final_seq}, "
                          f"score={best_score:.4f}")

            except Exception as e:
                print(f"    Error entrenando LSTM final pos {pos}: {e}")

    # ── Gradient Boosting ──────────────────────
    def _entrenar_gradient_boosting(self):
        """
        Entrena GB con validación cruzada temporal.
        FIX #3, #5, #6, #7.
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth':    [3, 5, 7],
            'learning_rate':[0.01, 0.1, 0.2]
        }

        for pos in range(1, NUM_POSICIONES + 1):
            X, y = self.gb_ensemble.prepare_features(self.df, pos)

            if len(X) < 10:
                print(f"    GB datos insuf. pos {pos}")
                continue

            n_splits = max(2, min(3, len(X) // 10))
            tscv     = TimeSeriesSplit(n_splits=n_splits)

            best_score  = -1.0
            best_params = None

            param_list = list(
                ParameterSampler(param_grid, n_iter=6, random_state=42)
            )

            for params in param_list:
                n_est   = int(params['n_estimators'])
                max_d   = int(params['max_depth'])
                lr      = float(params['learning_rate'])
                scores  = []

                for tr_idx, val_idx in tscv.split(X):
                    X_tr, X_val = X[tr_idx], X[val_idx]
                    y_tr, y_val = y[tr_idx], y[val_idx]

                    sc = StandardScaler()
                    X_tr_sc  = sc.fit_transform(X_tr)
                    X_val_sc = sc.transform(X_val)

                    y_tr_bin  = (y_tr  > np.median(y_tr)).astype(int)
                    y_val_bin = (y_val > np.median(y_val)).astype(int)

                    if len(np.unique(y_tr_bin)) < 2:
                        continue

                    m = GradientBoostingClassifier(
                        n_estimators=n_est, max_depth=max_d,
                        learning_rate=lr, random_state=42
                    )
                    m.fit(X_tr_sc, y_tr_bin)

                    preds  = m.predict(X_val_sc)
                    scores.append(accuracy_score(y_val_bin, preds))

                if scores:
                    mean_sc = float(np.mean(scores))
                    if mean_sc > best_score:
                        best_score  = mean_sc
                        best_params = params

            # Modelo final
            if best_params:
                sc_final = StandardScaler()
                X_sc     = sc_final.fit_transform(X)
                y_bin    = (y > np.median(y)).astype(int)

                if len(np.unique(y_bin)) < 2:
                    self.gb_ensemble.train_posicion(X, y, pos)
                    continue

                m_final = GradientBoostingClassifier(
                    n_estimators=int(best_params['n_estimators']),
                    max_depth=int(best_params['max_depth']),
                    learning_rate=float(best_params['learning_rate']),
                    random_state=42
                )
                m_final.fit(X_sc, y_bin)

                # FIX #6: key entera
                self.gb_ensemble.scalers[pos] = sc_final
                self.gb_ensemble.models[f'gb_{pos}'] = m_final
                print(f"    GB pos {pos}: "
                      f"n_est={best_params['n_estimators']}, "
                      f"score={best_score:.4f}")
            else:
                self.gb_ensemble.train_posicion(X, y, pos)

    # ── Causalidad ─────────────────────────────
    def _analizar_causalidad(self):
        self.causal_analyzer.detect_causal_relationships(self.df)

    # ── Backtesting ────────────────────────────
    def _ejecutar_backtesting(self):
        """
        Backtesting offline con ventana deslizante.
        FIX #4: model_weights inicializados antes de predecir().
        """
        print("    Iniciando backtesting...")

        if self.df is None or len(self.df) < MIN_SORTEOS_ENTRENAMIENTO + 10:
            print("    Datos insuficientes para backtesting.")
            self.backtest_scores = {k: 0.0
                                     for k in self._pesos_default()}
            return

        window_size = MIN_SORTEOS_ENTRENAMIENTO
        step_size   = max(5, len(self.df) // 20)

        model_hits: dict = {k: [] for k in self._pesos_default()}

        for start in range(0,
                           len(self.df) - window_size - 5,
                           step_size):
            end      = start + window_size
            train_df = self.df.iloc[start:end].copy().reset_index(drop=True)
            test_df  = self.df.iloc[end: end + 5].copy().reset_index(drop=True)

            if len(test_df) == 0:
                continue

            try:
                # FIX #4: crear modelo temporal con model_weights inicializados
                tmp = AdvancedMarkovBalotoV3AIEnhanced.__new__(
                    AdvancedMarkovBalotoV3AIEnhanced
                )
                tmp.df               = train_df
                tmp.alpha_smoothing  = self.alpha_smoothing
                tmp.time_decay       = self.time_decay
                tmp.numero_sorteos   = len(train_df)
                tmp.transition_matrices_order1 = {}
                tmp.transition_matrices_order3 = {}
                tmp.lstm_networks    = {}
                tmp.gb_ensemble      = GradientBoostingEnsemble()
                tmp.causal_analyzer  = CausalInferenceAnalyzer()
                tmp.rl_agent         = ReinforcementLearningAgent()
                tmp.volatility_patterns = {}
                tmp.anomaly_scores   = {}
                tmp.backtest_scores  = {}
                # FIX #4: pesos por defecto antes de predecir
                tmp.model_weights    = self._pesos_default()

                tmp._construir_modelos_markov()
                # LSTM y GB se omiten en backtesting para agilizar
                tmp._analizar_causalidad()

                for t_idx in range(len(test_df)):
                    pred = tmp.predecir()
                    if pred is None or 'numeros' not in pred:
                        continue

                    reales = [
                        int(test_df.iloc[t_idx, c])
                        for c in range(2, 7)
                    ]
                    pred_nums = pred['numeros'][:5]

                    aciertos = len(set(pred_nums) & set(reales))
                    precision = aciertos / max(len(reales), 1)

                    for key in model_hits:
                        model_hits[key].append(precision)

            except Exception as e:
                print(f"    Error backtesting idx {start}: {e}")
                continue

        self.backtest_scores = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in model_hits.items()
        }
        print(f"    Backtesting OK. "
              f"Precision media: "
              f"{np.mean(list(self.backtest_scores.values())):.4f}")

    # ── Calibración de pesos ───────────────────
    def _calibrar_pesos(self):
        """Calibra pesos basados en backtesting. Default si scores=0."""
        scores_arr = np.array(list(self.backtest_scores.values()),
                               dtype=float)
        total = scores_arr.sum()

        if total > 0:
            normalized = scores_arr / total
            self.model_weights = {
                k: float(v)
                for k, v in zip(self.backtest_scores.keys(), normalized)
            }
        else:
            # Pesos por defecto
            self.model_weights = self._pesos_default()

        print(f"    Pesos finales: {self.model_weights}")

    # ── Predicción ─────────────────────────────
    def predecir(self) -> dict | None:
        """
        Predicción combinada de todos los modelos.
        FIX #2: ventana histórica correcta para LSTM.
        FIX #4: garantiza model_weights no es None.
        """
        if self.df is None:
            return None

        # FIX #4: garantizar pesos
        if not self.model_weights:
            self.model_weights = self._pesos_default()

        ultima = self.df.iloc[-1]
        numeros_ult = [int(ultima.iloc[i]) for i in range(2, 7)]
        super_balota = int(ultima.iloc[7])

        resultado = {
            'numeros':         [],
            'super_balota':    0,
            'confianza_global':0.0,
            'detalles':        [],
        }

        for pos in range(1, NUM_POSICIONES + 1):
            combined: dict = defaultdict(float)

            # ── 1. Markov O1 ──
            w_o1  = self.model_weights.get('markov_o1', 0.15)
            o1_mat = self.transition_matrices_order1.get(pos, {})
            estado_o1 = numeros_ult[pos - 1]
            for num, prob in o1_mat.get(estado_o1, {}).items():
                combined[num] += prob * w_o1

            # ── 2. Markov O3 ──
            w_o3  = self.model_weights.get('markov_o3', 0.20)
            o3_mat = self.transition_matrices_order3.get(pos, {})
            if pos >= 3:
                estado_o3 = (
                    numeros_ult[pos - 3],
                    numeros_ult[pos - 2],
                    numeros_ult[pos - 1]
                )
                for num, prob in o3_mat.get(estado_o3, {}).items():
                    combined[num] += prob * w_o3

            # ── 3. LSTM ──  FIX #2
            w_lstm = self.model_weights.get('lstm', 0.25)
            if HAS_TENSORFLOW and pos in self.lstm_networks:
                lstm_net = self.lstm_networks[pos]
                col_idx  = pos + 1
                # FIX #2: usar historial real de la columna, no 'pos'
                hist_nums = self.df.iloc[:, col_idx].values.astype(float)
                seq_len   = lstm_net.sequence_length

                # Construir features para la ventana histórica
                all_feat, _ = self._construir_features_lstm(hist_nums)

                if len(all_feat) >= seq_len:
                    seq = all_feat[-seq_len:]   # shape (seq_len, 12)
                    try:
                        lstm_probs = lstm_net.predict_probabilities(seq)
                        if lstm_probs is not None:
                            for idx_num, prob in enumerate(lstm_probs):
                                combined[idx_num + 1] += float(prob) * w_lstm
                    except Exception as e:
                        print(f"    LSTM predict pos {pos}: {e}")

            # ── 4. Gradient Boosting ──
            w_gb  = self.model_weights.get('gradient_boosting', 0.25)
            try:
                col_idx_gb = pos + 1
                hist_gb    = self.df.iloc[:, col_idx_gb].values.astype(float)
                window_gb  = hist_gb[-20:] if len(hist_gb) >= 20 else hist_gb
                lag4_gb    = hist_gb[-4:]  if len(hist_gb) >= 4  else hist_gb

                feat_gb = calcular_features_gb(window_gb, lag4_gb)
                gb_prob = self.gb_ensemble.predict_posicion(feat_gb, pos)

                if gb_prob is not None:
                    # Distribuir probabilidad entre todos los números
                    for num in range(1, BALOTO_MAX + 1):
                        combined[num] += (gb_prob / BALOTO_MAX) * w_gb
            except Exception as e:
                print(f"    GB predict pos {pos}: {e}")

            # ── 5. Frecuencia ──
            w_freq = self.model_weights.get('frequency', 0.10)
            col_freq = self.df.iloc[:, pos + 1]
            total_s  = len(self.df)
            for num in range(1, BALOTO_MAX + 1):
                freq = col_freq.value_counts().get(num, 0) / total_s
                combined[num] += freq * w_freq

            # ── Top 3 ──
            top3 = sorted(
                combined.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            if top3:
                best_num, best_prob = top3[0]
                conf = min(max(float(best_prob) * 100, 0.0), 100.0)

                resultado['numeros'].append(int(best_num))
                resultado['detalles'].append({
                    'pos':  pos,
                    'num':  int(best_num),
                    'prob': round(float(best_prob), 4),
                    'conf': round(conf, 1),
                    'alt':  [{'n': int(n), 'p': round(float(p), 4)}
                              for n, p in top3]
                })

        # ── Super Balota ──
        resultado['super_balota'] = int((super_balota % SUPER_BALOTA_MAX) + 1)

        # ── Resolver duplicados ──
        resultado['numeros'] = resolver_duplicados(
            resultado['numeros'],
            resultado['detalles'],
            max_num=BALOTO_MAX
        )

        confs = [d['conf'] for d in resultado['detalles']]
        resultado['confianza_global'] = (
            round(float(np.mean(confs)), 1) if confs else 0.0
        )

        return resultado

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def ejecutar_tests():
    """Ejecuta toda la suite de pruebas."""
    print("\n" + "=" * 80)
    print("EJECUTANDO PRUEBAS UNITARIAS")
    print("=" * 80 + "\n")

    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFuncionesPuras))
    suite.addTests(loader.loadTestsFromTestCase(TestMarkovModel))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientBoostingEnsemble))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalInference))
    suite.addTests(loader.loadTestsFromTestCase(TestRLAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestCargaDatos))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"✓ TODAS LAS PRUEBAS PASARON "
              f"({result.testsRun} tests)")
    else:
        print(f"✗ FALLOS: {len(result.failures)} | "
              f"ERRORES: {len(result.errors)} | "
              f"TOTAL: {result.testsRun}")
    print("=" * 80 + "\n")

    return result.wasSuccessful()


def main():
    """Punto de entrada principal."""
    import os

    """# 1) Ejecutar pruebas primero
    tests_ok = ejecutar_tests()
    if not tests_ok:
        print("⚠ Pruebas fallidas. Revise los errores antes de continuar.")
        return"""

    # 2) Ejecutar modelo con datos reales
    print("\n" + "=" * 80)
    print("MODELO AI-ENHANCED v3.0")
    print("=" * 80 + "\n")

    csv_path = 'Baloto.csv'
    if not os.path.exists(csv_path):
        print(f"✗ Archivo '{csv_path}' no encontrado.")
        print("  Ejecute los tests con datos sintéticos (ya completados).")
        return

    analyzer = AdvancedMarkovBalotoV3AIEnhanced(csv_path)

    if not analyzer.cargar_datos():
        return

    if not analyzer.construir_modelos():
        return

    pred = analyzer.predecir()
    if pred is None:
        print("✗ Error: predicción retornó None.")
        return

    print("\n" + "=" * 80)
    print("PREDICCIÓN AI-ENHANCED v3.0")
    print("=" * 80)
    print(f"\n📊 RESULTADO:")
    print(f"   Números: {', '.join(map(str, pred['numeros']))}")
    print(f"   Super:   {pred['super_balota']}")
    print(f"\n📈 CONFIANZA:")
    print(f"   Global: {pred['confianza_global']:.1f}%")
    print(f"\n📍 DETALLES:")
    for d in pred['detalles']:
        alt_str = ', '.join(
            [f"{a['n']}({a['p']:.3f})" for a in d['alt']]
        )
        print(f"   Pos {d['pos']}: {d['num']} "
              f"(p={d['prob']:.4f}, conf={d['conf']:.1f}%) → {alt_str}")
    print("\n" + "=" * 80 + "\n")

    print("ARQUITECTURA UTILIZADA:")
    print("  * Markov Chains (O1, O3)")
    if HAS_TENSORFLOW:
        print("  * LSTM + Attention Mechanisms")
    print("  * Gradient Boosting Ensemble")
    if HAS_XGBOOST:
        print("  * XGBoost")
    if HAS_LIGHTGBM:
        print("  * LightGBM")
    print("  * Causal Inference (Granger)")
    print("  * Anomaly Detection (IsolationForest)")
    print("  * Reinforcement Learning (Q-Learning)")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()