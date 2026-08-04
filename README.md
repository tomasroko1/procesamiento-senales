# Decodificación de Dinámicas Cerebrales en EEG mediante Reducción Espacial por PCA, STFT y Clustering K-Means

Proyecto integrador para la materia **Procesamiento Avanzado de Señales y Minería de Series Temporales**. Implementa un pipeline computacional end-to-end para el análisis, procesamiento espectral y particionamiento no supervisado de bioseñales de EEG (PhysioNet EEGMMIDB, Sujeto S001).

---

## 🏛️ Estructura del Repositorio

*   **`analisis_eeg_presentacion.py`**: Pipeline integral en Python (MNE, SciPy, Scikit-Learn, Matplotlib) que descarga los registros EDF, ejecuta el filtrado FIR de fase cero, reducción espacial por PCA, análisis espectral Welch/STFT, extracción de Potencia Relativa en épocas disjuntas de 2.0 s, clustering K-Means, y exporta automáticamente las 4 figuras y la presentación web.
*   **`presentacion.html`**: Presentación académica interactiva (HTML5/CSS3/Vanilla JS) estructurada para 6 diapositivas con notas de orador integradas (tecla `N`) y soporte para pantalla completa (tecla `F`).
*   **`guion_presentacion.md`**: Guion de exposición oral calibrado para una defensa estricta de 9 minutos con tiempos asignados por diapositiva.
*   **`defensa_tribunal_faq.md`**: Guía de blindaje metodológico con respuestas académicas rigurosas y fundamentación matemática para las 8 preguntas más complejas de un tribunal evaluador.
*   **`fig1_pca_preprocesamiento.png`**: Trazas temporales occipitales (O1, Oz, O2) con espaciado no saturado y proyección espacial PC1 en escala vertical simétrica unificada.
*   **`fig2_psd_espectro_alfa.png`**: Densidad Espectral de Potencia (Welch) en escala lineal ($\mu\text{V}^2/\text{Hz}$) y semilogarítmica ($\text{dB}$) exhibiendo el Efecto Berger y la dinámica $1/f$.
*   **`fig3_espectrograma_stft.png`**: Representación tiempo-frecuencia continua de Ojos Abiertos vs. Ojos Cerrados (banda 8–12 Hz sostenida).
*   **`fig4_clustering_kmeans.png`**: Espacio 2D de Potencia Relativa (Alfa vs. Beta), frontera Voronoi de K-Means, centroides $\mu_0, \mu_1$ y matriz de confusión ($94.44\%$ de concordancia, $\text{ARI} = 0.786$, $\text{Silhouette} = 0.487$).

---

## 🚀 Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el pipeline completo
python analisis_eeg_presentacion.py

# 3. Abrir la presentación en el navegador
# Doble clic en presentacion.html o mediante servidor local
```

### Controles de la Presentación Web:
*   `[◀]` / `[▶]` o `[Espacio]`: Navegar entre diapositivas.
*   `[N]`: Alternar panel inferior de notas de orador (teleprompter).
*   `[F]`: Alternar pantalla completa.
