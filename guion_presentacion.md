# Guion de Exposición Oral (Defensa Final de 10 Minutos)

**Materia:** Procesamiento Avanzado de Señales y Minería de Series Temporales  
**Tema:** Decodificación de Dinámicas Cerebrales en EEG mediante PCA, STFT y K-Means  
**Duración Total:** 10:00 minutos (~1.300 palabras a ~130 palabras/minuto)

---

### Diapositiva 1: Portada y Marco Metodológico
* **Tiempo asignado:** `0:00 - 1:30` (1 minuto y 30 segundos)
* **Texto a exponer:**
> "Buenos días, profesores y miembros del tribunal evaluador. En esta presentación final expongo el desarrollo de un pipeline computacional end-to-end diseñado para decodificar dinámicas cerebrales a partir de señales de electroencefalografía real.
>
> El trabajo articula de forma directa los tres ejes temáticos de la asignatura:
> - En la **Unidad 1**, abordamos el acondicionamiento de señales, el filtrado digital, la reducción de dimensionalidad espacial mediante Análisis de Componentes Principales (PCA) y el análisis tiempo-frecuencia mediante la Transformada de Fourier de Tiempo Reducido (STFT).
> - En la **Unidad 2**, aplicamos minería de series temporales, segmentando la señal en épocas homogéneas para extraer atributos espectrales y alimentar algoritmos de agrupamiento no supervisado.
> - En la **Unidad 3**, contextualizamos estos métodos en una aplicación neurofisiológica clásica: la detección del ritmo alfa occipital y el fenómeno de sincronización de Berger.
>
> Para validar el pipeline, utilizamos los registros basales de 64 canales del Sujeto 1 del dataset PhysioNet EEGMMIDB, muestreados a 160 Hz, evaluando dos estados de un minuto cada uno: ojos abiertos y ojos cerrados."

---

### Diapositiva 2: Preprocesamiento y Reducción Espacial por PCA
* **Tiempo asignado:** `1:30 - 3:15` (1 minuto y 45 segundos)
* **Texto a exponer:**
> "Iniciamos con la etapa de acondicionamiento de la señal correspondiente a la **Unidad 1**.
>
> Las señales de EEG presentan amplitudes en el orden de los microvoltios y están severamente expuestas a interferencias. Aplicamos un filtro pasa-banda digital entre 1 y 40 Hz para eliminar la deriva de línea base, artefactos respiratorios y ruidos de alta frecuencia.
>
> Posteriormente, nos concentramos en los canales occipitales O1, Oz y O2, ubicados sobre la corteza visual primaria. Debido a la **conducción de volumen** a través del cráneo y cuero cabelludo, estos electrodos registran señales fuertemente correlacionadas. En lugar de seleccionar un canal de manera arbitraria perdiendo información espacial, aplicamos **PCA Espacial**.
>
> Calculamos la matriz de covarianza de los canales y proyectamos las series sobre el autovector dominante. Como se observa en la figura, la primera componente principal logra sintetizar el **91.05% de la varianza total**.
>
> En el panel inferior derecho apreciamos el impacto: mientras que con ojos abiertos la señal PC1 exhibe baja amplitud y comportamiento cuasi-estocástico, con ojos cerrados la componente rescata una oscilación sinusoidal nítida a 10 Hz, incrementando drásticamente la relación señal/ruido."

---

### Diapositiva 3: Densidad Espectral de Potencia (PSD) y Efecto Berger
* **Tiempo asignado:** `3:15 - 5:00` (1 minuto y 45 segundos)
* **Texto a exponer:**
> "Continuando en el dominio de la frecuencia (**Unidad 1**) y su aplicación neurofisiológica (**Unidad 3**), analizamos la Densidad Espectral de Potencia.
>
> Para estimar el espectro con baja varianza y evitar las fluctuaciones espurias del periodograma estándar, empleamos el **método de Welch** con ventanas Hanning de 2 segundos y 50% de solapamiento. En la Figura 2 comparamos el espectro de la componente PC1 en ambas condiciones.
>
> Este resultado cuantifica con precisión el histórico **Efecto Berger**. Cuando el sujeto mantiene los ojos abiertos, la aferencia de estímulos lumínicos a la retina provoca una **desincronización neuronal** en la corteza visual, manteniendo la energía en banda alfa en niveles basales.
>
> Al cerrar los ojos, cesa la entrada visual y las neuronas tálamo-corticales entran en un estado de **sincronización rítmica masiva**. La potencia integrada en la banda alfa (8 a 12 Hz) se multiplica por un factor de **16.40 veces**, generando un pico prominente exactamente en 10.0 Hz. Esto demuestra que la densidad de potencia alfa es un biomarcador altamente informativo para discriminar estados cerebrales."

---

### Diapositiva 4: Análisis Tiempo-Frecuencia Dinámico (STFT)
* **Tiempo asignado:** `5:00 - 6:45` (1 minuto y 45 segundos)
* **Texto a exponer:**
> "Dado que las bioseñales son inherentemente dinámicas y **no estacionarias**, el análisis de Fourier global resulta insuficiente al promediar todo el registro temporal. Por ello, recurrimos al análisis tiempo-frecuencia mediante la **Transformada de Fourier de Tiempo Reducido (STFT)**.
>
> La STFT balancea el límite de incertidumbre de Gabor-Heisenberg desplazando una ventana temporal sobre la señal para determinar cuándo y en qué frecuencias se concentran los cambios energéticos.
>
> En la Figura 3 presentamos la serie temporal concatenada en el panel superior y el espectrograma en el inferior. Durante los primeros 60 segundos (ojos abiertos), la energía en el rango de 8 a 12 Hz es baja y homogénea.
>
> Exactamente a los 60 segundos, marcado por la línea discontinua, se aprecia la transición abrupta: la banda alfa se enciende de forma sostenida e intensa en tonos amarillos brillantes hasta el final de la prueba. El espectrograma demuestra visualmente la estabilidad temporal y la reversibilidad del patrón neuroeléctrico."

---

### Diapositiva 5: Minería de Series Temporales y Clustering K-Means
* **Tiempo asignado:** `6:45 - 8:30` (1 minuto y 45 segundos)
* **Texto a exponer:**
> "En la **Unidad 2** entramos en la minería de series temporales y el aprendizaje de patrones. El desafío planteado fue: ¿puede un algoritmo no supervisado descubrir por sí solo los estados neurofisiológicos sin disponer de etiquetas previas?
>
> Para responderlo, segmentamos la serie temporal PC1 en **60 ventanas disjuntas de 2.0 segundos** (320 muestras por ventana) y extrajimos dos características espectrales por ventana: el logaritmo de la potencia alfa absoluta y la potencia alfa relativa normalizada respecto a la banda total de 1 a 30 Hz.
>
> Estandarizamos las variables y entrenamos el algoritmo **K-Means con k=2**.
>
> En la Figura 4 observamos la proyección de las muestras en el espacio de atributos: los datos forman dos conglomerados geométricamente compactos y linealmente separables.
>
> K-Means descubrió la separación con un **Silhouette Score de 0.781** y, al cotejarlo con la condición real, alcanzó una **exactitud del 100.0%** (30 ventanas de ojos abiertos y 30 de ojos cerrados perfectamente asignadas), demostrando la eficacia del proceso de ingeniería de atributos."

---

### Diapositiva 6: Conclusiones, Síntesis e Impacto en Neurotecnología
* **Tiempo asignado:** `8:30 - 10:00` (1 minuto y 30 segundos)
* **Texto a exponer:**
> "Para concluir, este trabajo demuestra de forma experimental que las técnicas del curso no son módulos aislados, sino componentes sinérgicos de un flujo de procesamiento integral:
> - El preprocesamiento y el **PCA espacial de la Unidad 1** resolvieron el problema físico de conducción de volumen y elevaron la relación señal/ruido reteniendo más del 91% de la varianza.
> - La caracterización espectral y la STFT permitieron aislar el **biomarcador del ritmo alfa de la Unidad 3**.
> - Y la minería temporal de la **Unidad 2** demostró que, con una adecuada extracción de atributos en ventanas cortas, un clasificador simple y no supervisado puede decodificar estados cognitivos con máxima precisión.
>
> Al operar sobre ventanas de solo 2 segundos y requerir una baja carga computacional, este pipeline es directamente exportable a **sistemas embebidos y de tiempo real**, tales como interfaces cerebro-computadora pasivas, detectores automáticos de somnolencia al volante o monitores de nivel de atención.
>
> Agradezco su atención y quedo a disposición para responder las preguntas del tribunal."
