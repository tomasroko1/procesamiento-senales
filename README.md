# Procesamiento Avanzado de Señales y Minería de Series Temporales
## Decodificación de Dinámicas Cerebrales en EEG (Efecto Berger, PCA Espacial, STFT y K-Means)

Repositorio con el código, análisis experimental, gráficos científicos, presentación HTML interactiva y guion de orador para la **Evaluación Final Integradora (Exposición de 10 Minutos)**.

---

## 🎯 Resumen Ejecutivo y Articulación Curricular

El proyecto implementa un pipeline computacional *end-to-end* en Python que evalúa y conecta los tres ejes temáticos de la materia a partir de registros electroencefalográficos reales de [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/):

1. **Unidad 1 (Preprocesamiento, Dominios Tiempo/Frecuencia, PCA Espacial y STFT):**
   - Filtrado digital pasa-banda FIR (1.0 – 40.0 Hz) para remoción de deriva basal e interferencias electromagnéticas.
   - Selección de electrodos occipitales ($O_1, O_z, O_2$) y reducción de dimensionalidad espacial mediante **Análisis de Componentes Principales (PCA)** para mitigar la conducción de volumen.
   - Estimación espectral consistente mediante el método de **Welch PSD**.
   - Análisis tiempo-frecuencia dinámico mediante la **Transformada de Fourier de Tiempo Reducido (STFT)**.

2. **Unidad 2 (Minería de Series Temporales y Aprendizaje No Supervisado):**
   - Segmentación de la señal en ventanas disjuntas de corta duración ($2.0\text{ s}$, 320 muestras).
   - Ingeniería y extracción de atributos espectrales: $\log_{10}(\text{Potencia Absoluta Alfa})$ y Potencia Relativa Alfa.
   - Agrupamiento no supervisado mediante **K-Means ($k=2$)** y evaluación geométrica con *Silhouette Score* y Matriz de Confusión.

3. **Unidad 3 (Aplicaciones Biomédicas y Neurotecnología):**
   - Caracterización neurofisiológica del **Efecto Berger (1929)**: desincronización cortical con ojos abiertos vs. sincronización masiva en banda Alfa ($8\text{--}12\text{ Hz}$) en reposo con ojos cerrados.
   - Proyección de la metodología a sistemas embebidos de *Brain-Computer Interfaces* (BCI) y monitores de somnolencia/atención en tiempo real.

---

## 📊 Resultados Principales

| Métrica / Etapa | Concepto del Programa | Resultado Obtenido |
| :--- | :--- | :--- |
| **Varianza Explicada por $PC_1$** | Unidad 1: PCA Espacial | **`91.05%`** (los 3 canales convergen a 1 serie unificada) |
| **Incremento Espectral (Ratio Berger)** | Unidades 1 y 3: Biomarcador Alfa | **`16.40x`** más potencia en $8\text{--}12\text{ Hz}$ con ojos cerrados |
| **Ventanas Analizadas ($2.0\text{ s}$)** | Unidad 2: Minería Temporal | **`60 ventanas`** disjuntas ($F_s = 160\text{ Hz}$) |
| **Silhouette Score del Clustering** | Unidad 2: Calidad de Agrupamiento | **`0.781`** (conglomerados compactos y convexos) |
| **Exactitud vs. Ground Truth** | Unidad 2: Validación No Supervisada | **`100.00%`** (Matriz de confusión: 30/30 abiertos, 30/30 cerrados) |

---

## 🖼️ Figuras Científicas Generadas

1. **`fig1_pca_preprocesamiento.png`**: Comparación de canales crudos occipitales vs. primera componente principal ($PC_1$) en ambas condiciones.
2. **`fig2_psd_espectro_alfa.png`**: Densidad Espectral de Potencia (Welch) con realce de la banda Alfa ($8\text{--}12\text{ Hz}$) y pico resonante en $10.0\text{ Hz}$.
3. **`fig3_espectrograma_stft.png`**: Espectrograma continuo que evidencia la activación súbita y sostenida de la banda Alfa a partir del segundo 60.
4. **`fig4_clustering_kmeans.png`**: Proyección en el espacio de atributos bidimensional, fronteras de K-Means y matriz de confusión.

---

## 🖥️ Presentación Interactiva (`presentacion.html`)

El archivo **`presentacion.html`** es una presentación de diapositivas auto-contenida con diseño formal académico y controles integrados:
- **`▶` / `◀` o `Espacio`**: Navegación entre diapositivas.
- **`N`**: Desplegar / ocultar las **Notas de Orador** (guion guiado de 10 minutos con tiempos calibrados).
- **`F`**: Modo **Pantalla Completa**.

---

## 🚀 Instalación y Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tomasroko1/procesamiento-senales.git
   cd procesamiento-senales
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el pipeline:**
   ```bash
   python analisis_eeg_presentacion.py
   ```

4. **Abrir la presentación:**
   Abre el archivo `presentacion.html` en tu navegador favorito.
