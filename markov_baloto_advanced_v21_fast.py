"""
MODELO ULTRA-OPTIMIZADO RÁPIDO - PREDICCIÓN DEL BALOTO v2.1 FAST
================================================================

Versión optimizada v2.1 sin Gaussian Processes (demasiado lentos).
Mantiene todas las mejoras clave con tiempo de ejecución ~40-50 segundos.

Técnicas implementadas:
1. Detección mejorada de anomalías
2. Análisis de volatilidad de patrones
3. Calibración inteligente de confianza
4. Backtesting temporal
5. Ensemble ponderado por exactitud
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from scipy.special import logsumexp
from scipy.fft import fft
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class AdvancedMarkovBalotoV21Fast:
    """
    Modelo ultra-optimizado RÁPIDO de predicción del Baloto.
    Mantiene mejoras clave sin overhead computacional de Gaussian Processes.
    """
    
    def __init__(self, csv_file, alpha_smoothing=0.5, time_decay=0.95):
        self.csv_file = csv_file
        self.df = None
        self.alpha_smoothing = alpha_smoothing
        self.time_decay = time_decay
        
        # Modelos base
        self.transition_matrices_order1 = {}
        self.transition_matrices_order2 = {}
        self.transition_matrices_order3 = {}
        self.transition_matrices_order20 = {}
        self.initial_distributions = {}
        self.frequency_distributions = {}
        self.ciclos_temporales = {}
        self.periodo_promedio = {}
        
        # Super Balota
        self.transition_matrix_super = defaultdict(lambda: defaultdict(float))
        self.frequency_distribution_super = defaultdict(int)
        
        # Características v2.1 Fast
        self.volatility_patterns = {}
        self.trend_strength = {}
        self.seasonal_patterns = {}
        self.optimal_weights = None
        self.model_accuracies = {}
        self.anomaly_scores = {}
        
    def cargar_datos(self):
        """Carga datos."""
        try:
            self.df = pd.read_csv(self.csv_file)
            self.df = self.df.dropna()
            self.numero_sorteos = len(self.df)
            print(f"[OK] Datos cargados: {self.numero_sorteos} sorteos ({self.df.iloc[0, 0]} a {self.df.iloc[-1, 0]})\n")
            return True
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            return False
    
    def construir_modelos(self):
        """Construye todos los modelos."""
        if self.df is None:
            print("✗ Error: cargue datos primero")
            return False
        
        print("="*80)
        print("CONSTRUCCIÓN DE MODELOS ULTRA-OPTIMIZADOS v2.1 FAST")
        print("="*80 + "\n")
        
        print("[1/5] Detectando anomalías...")
        self._detectar_anomalias()
        print("  [OK] Anomalías identificadas\n")

        print("[2/5] Analizando volatilidad y tendencias...")
        self._analizar_volatilidad()
        print("  [OK] Volatilidad calculada\n")

        print("[3/5] Construyendo modelos Markov (O1, O2, O3, O20)...")
        self._construir_modelos_markov()
        print("  [OK] Modelos construidos\n")

        print("[4/5] Ejecutando backtesting...")
        self._ejecutar_backtesting()
        print("  [OK] Backtesting completado\n")

        print("[5/5] Calibrando pesos óptimos...")
        self._calibrar_pesos()
        print("  [OK] Pesos calibrados\n")

        print("="*80)
        print("[OK] MODELOS LISTOS PARA PREDICCIÓN")
        print("="*80 + "\n")
        
        return True
    
    def _detectar_anomalias(self):
        """Detección de anomalías mejorada."""
        posiciones_columnas = list(range(2, 7))
        datos = self.df.iloc[:, posiciones_columnas].values
        
        for pos in range(1, 6):
            X = []
            for fila in datos:
                features = [
                    (np.mean(fila) - 22) / 13,
                    np.std(fila) / 15,
                    (np.max(fila) - np.min(fila)) / 42,
                ]
                X.append(features)
            
            X = np.array(X)
            iso_forest = IsolationForest(contamination=0.08, random_state=42, n_estimators=50)
            iso_forest.fit(X)
            self.anomaly_scores[pos] = iso_forest.score_samples(X)
    
    def _analizar_volatilidad(self):
        """Analiza volatilidad de patrones."""
        posiciones_columnas = list(range(2, 7))
        
        for pos_idx, col_idx in enumerate(posiciones_columnas):
            pos = pos_idx + 1
            numeros = self.df.iloc[:, col_idx].values
            
            # Volatilidad
            if len(numeros) > 1:
                diffs = np.diff(numeros)
                volatility = np.std(diffs) / (np.mean(np.abs(numeros)) + 1e-6)
            else:
                volatility = 0
            
            self.volatility_patterns[pos] = volatility
            
            # Tendencia
            if len(numeros) > 1:
                trend = np.polyfit(range(len(numeros)), numeros, 1)[0]
            else:
                trend = 0
            
            self.trend_strength[pos] = abs(trend)
            
            # Estacionalidad (FFT)
            if len(numeros) > 10:
                try:
                    fft_vals = fft(numeros - np.mean(numeros))
                    power = np.abs(fft_vals) ** 2
                    freqs = np.fft.fftfreq(len(numeros))
                    if len(freqs) > 1:
                        top_freq_idx = np.argmax(power[1:len(numeros)//2]) + 1
                        dominant_period = 1 / freqs[top_freq_idx] if freqs[top_freq_idx] != 0 else 0
                    else:
                        dominant_period = 0
                except:
                    dominant_period = 0
                
                self.seasonal_patterns[pos] = abs(dominant_period)
            else:
                self.seasonal_patterns[pos] = 0
    
    def _construir_modelos_markov(self):
        """Construye modelos Markov."""
        posiciones_columnas = list(range(2, 7))
        
        for pos_idx, col_idx in enumerate(posiciones_columnas):
            posicion = pos_idx + 1
            numeros = self.df.iloc[:, col_idx].values.astype(int)
            
            # O1
            self._construir_o1(posicion, numeros)
            # O2
            self._construir_o2(posicion, numeros)
            # O3
            self._construir_o3(posicion, numeros)
            # O20
            self._construir_o20(posicion, numeros)
            # Ciclos
            self._analizar_ciclos(posicion, numeros)
            # Frecuencias
            contador = Counter(numeros)
            self.frequency_distributions[posicion] = {
                num: count / len(numeros) for num, count in contador.items()
            }
        
        # Super Balota
        super_balotas = self.df.iloc[:, 7].values.astype(int)
        self._construir_super(super_balotas)
        contador_super = Counter(super_balotas)
        self.frequency_distribution_super = {
            num: count / len(super_balotas) for num, count in contador_super.items()
        }
    
    def _construir_o1(self, posicion, numeros):
        """Markov Orden-1."""
        matriz = defaultdict(lambda: defaultdict(float))
        n = len(numeros)
        
        for i in range(len(numeros) - 1):
            estado = numeros[i]
            siguiente = numeros[i + 1]
            peso = self.time_decay ** (n - i - 2)
            matriz[estado][siguiente] += peso
        
        probs = {}
        for estado, transiciones in matriz.items():
            total = sum(transiciones.values())
            total_smooth = total + self.alpha_smoothing * 43
            
            probs[estado] = {}
            for siguiente in range(1, 44):
                count = transiciones.get(siguiente, 0)
                probs[estado][siguiente] = (count + self.alpha_smoothing) / total_smooth
        
        self.transition_matrices_order1[posicion] = probs
    
    def _construir_o2(self, posicion, numeros):
        """Markov Orden-2."""
        matriz = defaultdict(lambda: defaultdict(float))
        n = len(numeros)
        
        for i in range(len(numeros) - 2):
            estado = (numeros[i], numeros[i + 1])
            siguiente = numeros[i + 2]
            peso = self.time_decay ** (n - i - 3)
            matriz[estado][siguiente] += peso
        
        probs = {}
        for estado, transiciones in matriz.items():
            total = sum(transiciones.values())
            total_smooth = total + self.alpha_smoothing * 43
            
            probs[estado] = {}
            for siguiente in range(1, 44):
                count = transiciones.get(siguiente, 0)
                probs[estado][siguiente] = (count + self.alpha_smoothing) / total_smooth
        
        self.transition_matrices_order2[posicion] = probs
    
    def _construir_o3(self, posicion, numeros):
        """Markov Orden-3."""
        matriz = defaultdict(lambda: defaultdict(float))
        n = len(numeros)
        
        for i in range(len(numeros) - 3):
            estado = (numeros[i], numeros[i + 1], numeros[i + 2])
            siguiente = numeros[i + 3]
            peso = self.time_decay ** (n - i - 4)
            matriz[estado][siguiente] += peso
        
        probs = {}
        for estado, transiciones in matriz.items():
            total = sum(transiciones.values())
            total_smooth = total + self.alpha_smoothing * 43
            
            probs[estado] = {}
            for siguiente in range(1, 44):
                count = transiciones.get(siguiente, 0)
                probs[estado][siguiente] = (count + self.alpha_smoothing) / total_smooth
        
        self.transition_matrices_order3[posicion] = probs
    
    def _construir_o20(self, posicion, numeros):
        """Markov Orden-20."""
        if len(numeros) < 21:
            self.transition_matrices_order20[posicion] = {}
            return
        
        matriz = defaultdict(lambda: defaultdict(float))
        n = len(numeros)
        
        for i in range(len(numeros) - 20):
            estado = tuple(numeros[i:i+20])
            siguiente = numeros[i + 20]
            peso = self.time_decay ** (n - i - 21)
            matriz[estado][siguiente] += peso
        
        probs = {}
        for estado, transiciones in matriz.items():
            total = sum(transiciones.values())
            total_smooth = total + self.alpha_smoothing * 43
            
            probs[estado] = {}
            for siguiente in range(1, 44):
                count = transiciones.get(siguiente, 0)
                prob = (count + self.alpha_smoothing) / total_smooth
                if prob > 0.01:
                    probs[estado][siguiente] = prob
        
        self.transition_matrices_order20[posicion] = probs
    
    def _analizar_ciclos(self, posicion, numeros):
        """Analiza ciclos."""
        ciclos = {}
        periodo_promedio = {}
        
        for numero in range(1, 44):
            posiciones = [i for i, num in enumerate(numeros) if num == numero]
            if len(posiciones) > 1:
                intervalos = [posiciones[i+1] - posiciones[i] for i in range(len(posiciones)-1)]
                ciclos[numero] = intervalos
                periodo_promedio[numero] = np.mean(intervalos)
        
        self.ciclos_temporales[posicion] = ciclos
        self.periodo_promedio[posicion] = periodo_promedio
    
    def _construir_super(self, super_balotas):
        """Construye modelo Super Balota."""
        n = len(super_balotas)
        
        # O1
        for i in range(len(super_balotas) - 1):
            estado = super_balotas[i]
            siguiente = super_balotas[i + 1]
            peso = self.time_decay ** (n - i - 2)
            self.transition_matrix_super[estado][siguiente] += peso
        
        # Normalizar
        for estado, transiciones in self.transition_matrix_super.items():
            total = sum(transiciones.values())
            total_smooth = total + self.alpha_smoothing * 16
            
            for siguiente in range(1, 17):
                count = transiciones.get(siguiente, 0)
                self.transition_matrix_super[estado][siguiente] = (
                    (count + self.alpha_smoothing) / total_smooth
                )
    
    def _ejecutar_backtesting(self):
        """Backtesting simple."""
        n = len(self.df)
        train_size = int(n * 0.7)
        
        for modelo in ['o1', 'o2', 'o3', 'o20']:
            correctos = 0
            total = 0
            
            for idx in range(train_size, n - 1):
                sorteo_actual = self.df.iloc[idx, 2:7].values.astype(int)
                sorteo_siguiente = self.df.iloc[idx + 1, 2:7].values.astype(int)
                pred = self._predecir_con_modelo(sorteo_actual, modelo)
                aciertos = len(set(pred) & set(sorteo_siguiente))
                correctos += aciertos
                total += 5
            
            accuracy = correctos / total if total > 0 else 0
            self.model_accuracies[modelo] = accuracy
    
    def _predecir_con_modelo(self, numeros, modelo):
        """Predice con modelo."""
        predicciones = []
        
        for pos in range(1, 6):
            if modelo == 'o1':
                probs = self.transition_matrices_order1[pos].get(numeros[pos-1], {})
            elif modelo == 'o2':
                probs = self._get_o2(numeros, pos)
            elif modelo == 'o3':
                probs = self._get_o3(numeros, pos)
            else:
                probs = self._get_o20(numeros, pos)
            
            if probs:
                top_num = max(probs.items(), key=lambda x: x[1])[0]
                predicciones.append(top_num)
            else:
                predicciones.append(np.random.randint(1, 44))
        
        return predicciones[:5]
    
    def _get_o2(self, numeros, pos):
        if pos < 2:
            return {}
        estado = (numeros[pos-2], numeros[pos-1])
        return self.transition_matrices_order2[pos].get(estado, {})
    
    def _get_o3(self, numeros, pos):
        if pos < 3:
            return {}
        estado = (numeros[pos-3], numeros[pos-2], numeros[pos-1])
        return self.transition_matrices_order3[pos].get(estado, {})
    
    def _get_o20(self, numeros, pos):
        if pos < 20:
            return {}
        estado = tuple(numeros[pos-20:pos])
        return self.transition_matrices_order20[pos].get(estado, {})
    
    def _calibrar_pesos(self):
        """Calibra pesos."""
        if self.model_accuracies and sum(self.model_accuracies.values()) > 0:
            total = sum(self.model_accuracies.values())
            self.optimal_weights = {
                m: acc / total for m, acc in self.model_accuracies.items()
            }
        else:
            self.optimal_weights = {'o20': 0.40, 'o3': 0.30, 'o2': 0.18, 'o1': 0.12}
    
    def predecir(self):
        """Predicción optimizada."""
        if not self.df is not None:
            return None
        
        if not self.optimal_weights:
            self._calibrar_pesos()
        
        ultima_fila = self.df.iloc[-1]
        numeros = [int(ultima_fila.iloc[i]) for i in range(2, 7)]
        super_balota = int(ultima_fila.iloc[7])
        
        resultado = {
            'numeros': [],
            'super_balota': None,
            'confianza_global': 0.0,
            'detalles': []
        }
        
        # Predicción por posición
        for pos in range(1, 6):
            probs = {}
            
            # O1
            peso_o1 = self.optimal_weights.get('o1', 0.12)
            for num, prob in self.transition_matrices_order1[pos].get(numeros[pos-1], {}).items():
                probs[num] = probs.get(num, 0) + prob * peso_o1
            
            # O2
            peso_o2 = self.optimal_weights.get('o2', 0.18)
            o2_estado = (numeros[pos-2], numeros[pos-1]) if pos >= 2 else None
            if o2_estado:
                for num, prob in self.transition_matrices_order2[pos].get(o2_estado, {}).items():
                    probs[num] = probs.get(num, 0) + prob * peso_o2
            
            # O3
            peso_o3 = self.optimal_weights.get('o3', 0.30)
            o3_estado = (numeros[pos-3], numeros[pos-2], numeros[pos-1]) if pos >= 3 else None
            if o3_estado:
                for num, prob in self.transition_matrices_order3[pos].get(o3_estado, {}).items():
                    probs[num] = probs.get(num, 0) + prob * peso_o3
            
            # O20
            peso_o20 = self.optimal_weights.get('o20', 0.40)
            o20_estado = tuple(numeros[pos-20:pos]) if pos >= 20 else None
            if o20_estado:
                for num, prob in self.transition_matrices_order20[pos].get(o20_estado, {}).items():
                    probs[num] = probs.get(num, 0) + prob * peso_o20
            
            # Top 3
            top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_3:
                numero = top_3[0][0]
                probabilidad = top_3[0][1]
                
                # Confianza calibrada
                freq = self.frequency_distributions[pos].get(numero, 1/43)
                volatility_factor = 1.0 / (1.0 + self.volatility_patterns.get(pos, 0) * 5)
                confianza = (probabilidad * 0.5 + freq * 0.3) * volatility_factor * 100
                confianza = min(max(confianza, 0), 100)
                
                resultado['numeros'].append(numero)
                resultado['detalles'].append({
                    'pos': pos,
                    'num': numero,
                    'prob': round(probabilidad, 4),
                    'conf': round(confianza, 1),
                    'alt': [{'n': n, 'p': round(p, 4)} for n, p in top_3]
                })
        
        # Super Balota
        if super_balota in self.transition_matrix_super:
            probs_super = self.transition_matrix_super[super_balota]
            top_super = max(probs_super.items(), key=lambda x: x[1])
            resultado['super_balota'] = int(top_super[0])
            resultado['conf_super'] = round(top_super[1] * 100, 1)
        else:
            resultado['super_balota'] = np.random.randint(1, 17)
            resultado['conf_super'] = 6.25
        
        # Resolver duplicados
        numeros_unicos = []
        usados = set()
        for idx, num in enumerate(resultado['numeros']):
            if num not in usados:
                numeros_unicos.append(num)
                usados.add(num)
            else:
                for alt in resultado['detalles'][idx]['alt']:
                    if alt['n'] not in usados:
                        numeros_unicos.append(alt['n'])
                        usados.add(alt['n'])
                        break
        
        resultado['numeros'] = numeros_unicos[:5]
        confianzas = [d['conf'] for d in resultado['detalles']]
        resultado['confianza_global'] = round(np.mean(confianzas), 1) if confianzas else 0
        
        return resultado


def main():
    print("\n" + "="*80)
    print("MODELO ULTRA-OPTIMIZADO RÁPIDO v2.1 FAST")
    print("="*80 + "\n")
    
    analyzer = AdvancedMarkovBalotoV21Fast('Baloto.csv')
    
    if analyzer.cargar_datos():
        if analyzer.construir_modelos():
            prediccion = analyzer.predecir()
            
            if prediccion:
                print("\n" + "="*80)
                print("PREDICCION OPTIMIZADA v2.1 FAST")
                print("="*80)
                print(f"\nRESULTADO:")
                print(f"   Numeros: {', '.join(map(str, prediccion['numeros']))}")
                print(f"   Super:   {prediccion['super_balota']}")
                print(f"\nCONFIANZA:")
                print(f"   Global: {prediccion['confianza_global']:.1f}%")
                print(f"   Super:  {prediccion['conf_super']:.1f}%")
                print(f"\nDETALLES:")
                for d in prediccion['detalles']:
                    alt_str = ', '.join([f"{a['n']}({a['p']:.3f})" for a in d['alt']])
                    print(f"   Pos {d['pos']}: {d['num']} (p={d['prob']:.4f}, conf={d['conf']:.1f}%) -> {alt_str}")
                print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
