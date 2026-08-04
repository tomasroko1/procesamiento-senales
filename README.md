# Procesamiento Avanzado de Señales y Minería de Series Temporales
## Decodificación de Dinámicas Cerebrales en EEG (Efecto Berger, Reducción Espacial por PCA, STFT y K-Means)

Repositorio con el código, análisis experimental, gráficos científicos, presentación HTML interactiva y guion de orador para la **Evaluación Final Integradora (Exposición de 10 Minutos)**.

---

## 🎯 Resumen Ejecutivo y Articulación Curricular

El proyecto implementa un pipeline computacional *end-to-end* en Python que evalúa y conecta los tres ejes temáticos de la materia a partir de registros electroencefalográficos reales de [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/):

1. **Unidad 1 (Preprocesamiento, Dominios Tiempo/Frecuencia, Reducción Espacial por PCA y STFT):**
   - Carga estándar mediante `mne.datasets.eegbci`.
   - Filtrado digital pasa-banda FIR de **fase cero** bidireccional no causal (`phase='zero'`, 1.0 – 40.0 Hz) con recorte dinámico de transitorios de borde ($3.0\text{ s}$, acorde a la respuesta impulsional del pasa-alto de 1 Hz).
   - Selección de electrodos occipitales ($O_1, O_z, O_2$) y reducción de dimensionalidad espacial mediante **Análisis de Componentes Principales (PCA)** para proyectar la fuente oscilatoria dominante del dipolo visual.
   - Estimación espectral consistente mediante el método de **Welch PSD** ($\Delta f = 0.5\text{ Hz}$, ventana Hann de $2.0\text{ s}$, 50% solapamiento).
   - Análisis tiempo-frecuencia dinámico mediante la **Transformada de Fourier de Tiempo Reducido (STFT)** con paneles comparativos independientes.

2. **Unidad 2 (Minería de Series Temporales y Aprendizaje No Supervisado):**
   - Segmentación de la señal en épocas disjuntas de corta duración ($2.0\text{ s}$, 320 muestras).
   - Estimación espectral por época mediante **Periodograma Modificado con ventana Hann** ($\Delta f = 0.5\text{ Hz}$).
   - Extracción de atributos de **Potencia Relativa** de banda ($P_{\text{banda}} / P_{1-40\text{Hz}}$), garantizando invariancia ante variaciones en la impedancia de los electrodos y grosor craneal.
   - Agrupamiento no supervisado mediante **K-Means ($k=2$)** y evaluación formal con *Silhouette Score*, *Adjusted Rand Index (ARI)* y Matriz de Confusión.

3. **Unidad 3 (Aplicaciones Biomédicas y Neurotecnología):**
   - Caracterización neurofisiológica del **Efecto Berger (1929)**: desincronización cortical con ojos abiertos vs. sincronización masiva en banda Alfa ($8\text{--}12\text{ Hz}$) en reposo con ojos cerrados.
   - Discusión crítica y proyección de la metodología a sistemas de *Brain-Computer Interfaces* (BCI) y monitoreo en tiempo real, evaluando la variabilidad inter-sujeto en la Frecuencia Alfa Individual (IAF) y esquemas *Leave-One-Subject-Out* (LOSO).

---

## 📊 Resultados Principales

| Métrica / Etapa | Concepto del Programa | Resultado Obtenido |
| :--- | :--- | :--- |
| **Varianza Explicada por $PC_1$** | Unidad 1: Reducción Espacial PCA | **`90.61%`** (Loadings: $O_1=0.585, O_z=0.568, O_2=0.579$) |
| **Incremento Espectral (Ratio Berger)** | Unidades 1 y 3: Biomarcador Alfa | **`16.00x`** más potencia en $8\text{--}12\text{ Hz}$ ($553.58\ \mu\text{V}^2$ vs $8873.06\ \mu\text{V}^2$) |
| **Ventanas Analizadas ($2.0\text{ s}$ netas)** | Unidad 2: Minería Temporal | **`54 épocas`** disjuntas ($27$ Abiertos, $27$ Cerrados) |
| **Espacio de Atributos 2D** | Unidad 2: Invarianza de Impedancia | **Potencia Relativa** $(\text{Alfa } [8\text{-}12\text{Hz}] / [1\text{-}40\text{Hz}], \text{Beta } [13\text{-}30\text{Hz}] / [1\text{-}40\text{Hz}])$ |
| **Adjusted Rand Index (ARI)** | Unidad 2: Calidad de Partición | **`0.786`** (fuerte concordancia no supervisada) |
| **Silhouette Score del Clustering** | Unidad 2: Compacidad Geométrica | **`0.487`** (separabilidad limpia de conglomerados) |
| **Concordancia Semántica vs. Ground Truth** | Unidad 2: Prueba Intra-Sujeto | **`94.44%`** (Matriz de confusión: 27/27 abiertos, 24/27 cerrados) |

---

## 🖼️ Figuras Científicas Generadas

1. **`fig1_pca_preprocesamiento.png`**: Comparación de canales crudos occipitales vs. primera componente principal ($PC_1$) en microvoltios ($\mu\text{V}$).
2. **`fig2_psd_espectro_alfa.png`**: Densidad Espectral de Potencia (Welch) en $\mu\text{V}^2/\text{Hz}$ con realce de la banda Alfa ($8\text{--}12\text{ Hz}$) y pico resonante en $10.0\text{ Hz}$.
3. **`fig3_espectrograma_stft.png`**: Espectrogramas comparativos continuos que evidencian la activación sostenida de la banda Alfa durante el reposo con ojos cerrados.
4. **`fig4_clustering_kmeans.png`**: Proyección en el espacio 2D de Potencia Relativa ($\%\text{Alfa}$ vs. $\%\text{Beta}$), fronteras de K-Means y matriz de confusión.

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

3. **Ejecutar el pipeline y regenerar presentación:**
   ```bash
   python analisis_eeg_presentacion.py
   ```

4. **Abrir la presentación:**
   Abre el archivo `presentacion.html` en tu navegador.
