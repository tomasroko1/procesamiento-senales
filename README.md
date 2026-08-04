# Caracterización Espectral y Agrupamiento No Supervisado del Ritmo Alfa Occipital en EEG (Estudio de Caso Intra-Sujeto)

Proyecto integrador para la materia **Procesamiento Avanzado de Señales y Minería de Series Temporales**. Implementa un pipeline computacional reproducible para el preprocesamiento, combinación espacial por PCA, análisis espectral Welch/STFT y particionamiento no supervisado mediante K-Means de bioseñales de EEG (PhysioNet EEGMMIDB, Sujeto S001).

---

## 🏛️ Estructura del Repositorio

*   **`analisis_eeg_presentacion.py`**: Pipeline integral en Python (MNE, SciPy, Scikit-Learn, Matplotlib) que descarga los registros EDF, ejecuta el filtrado FIR de fase cero, reducción espacial por PCA, análisis espectral Welch/STFT, extracción de Potencia Relativa en 54 épocas disjuntas de 2.0 s, clustering K-Means, y exporta automáticamente las 4 figuras científicas y la presentación web.
*   **`presentacion.html`**: Presentación académica interactiva (HTML5/CSS3/Vanilla JS) estructurada para 6 diapositivas con notas de orador integradas (tecla `N`) y soporte para pantalla completa (tecla `F`).
*   **`guion_presentacion.md`**: Guion de exposición oral calibrado para una defensa estricta de 9 minutos con tiempos asignados por diapositiva.
*   **`fig1_pca_preprocesamiento.png`**: Trazas temporales occipitales (O1, Oz, O2) con espaciado vertical no saturado, proyección espacial PC1 en escala vertical simétrica unificada e inset con esquema topográfico 10-20 del polo occipital.
*   **`fig2_psd_espectro_alfa.png`**: Densidad Espectral de Potencia (Welch) en escala lineal ($\mu\text{V}^2/\text{Hz}$) y semilogarítmica ($\text{dB}$) distinguiendo el Efecto Berger ($f_{\text{IAF}} = 10.0\text{ Hz}$, Ratio: 16.00x) del fondo aperiódico $1/f$.
*   **`fig3_espectrograma_stft.png`**: Representación tiempo-frecuencia continua de Ojos Abiertos vs. Ojos Cerrados con rango dinámico calibrado (36 dB), $\Delta t_{\text{hop}} = 0.25\text{ s}$ y marcado de fluctuaciones atencionales transitorias.
*   **`fig4_clustering_kmeans.png`**: Espacio composicional 2D de Potencia Relativa (Alfa vs. Beta, $r = -0.472$), frontera de decisión Voronoi de K-Means, centroides $\mu_0, \mu_1$ y matriz de contingencia post-hoc ($\text{ARI} = 0.786$, $\text{Silhouette} = 0.487$, concordancia semántica $94.44\%$).

---

## 🚀 Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el pipeline completo y regenerar figuras y presentación
python analisis_eeg_presentacion.py

# 3. Abrir la presentación en el navegador
# Doble clic en presentacion.html o mediante servidor local
```

### Controles de la Presentación Web:
*   `[◀]` / `[▶]` o `[Espacio]`: Navegar entre diapositivas.
*   `[N]`: Alternar panel inferior de notas de orador (teleprompter).
*   `[F]`: Alternar pantalla completa.
