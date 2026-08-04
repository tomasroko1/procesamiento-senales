# Guion de Exposición Oral (Video Explicativo de 9 Minutos)

**Materia:** Procesamiento Avanzado de Señales y Minería de Series Temporales (UBA)  
**Tema:** Caracterización Espectral y Agrupamiento No Supervisado del Ritmo Alfa Occipital en EEG (Estudio de Caso Intra-Sujeto)  
**Calibración de Tiempo:** 9 minutos netos para video explicativo asincrónico  

---

### Diapositiva 1: Introducción, Metodología y Objetivo

**Tiempo asignado:** 0:00 - 1:20

**Texto a exponer:**

En este trabajo presento un pipeline computacional para el análisis y caracterización cuantitativa del ritmo alfa occipital en electroencefalografía (EEG), utilizando un enfoque no supervisado como estudio de caso intra-sujeto.

La metodología se estructura en tres etapas funcionales integradas:

Primero, abordamos el **preprocesamiento y reducción espacial**: implementamos un filtrado digital FIR de fase cero para evitar distorsiones temporales, realizamos una combinación espacial óptima mediante Análisis de Componentes Principales (PCA) y construimos representaciones tiempo-frecuencia continuas con la Transformada de Fourier de Tiempo Reducido (STFT).

Segundo, aplicamos técnicas de **minería de series temporales**: segmentamos el registro en 54 épocas temporales disjuntas e independientes, extrajimos características espectrales relativas robustas y evaluamos el agrupamiento con K-Means mediante métricas no supervisadas formales como el Adjusted Rand Index y el Silhouette Score.

Tercero, **contextualizamos los resultados biofísicamente**: caracterizando cuantitativamente el ritmo alfa y el Efecto Berger sobre registros basales del sujeto S001 de PhysioNet EEGMMIDB, adquiridos a 160 Hz en condiciones de ojos abiertos y ojos cerrados.

---

### Diapositiva 2: Preprocesamiento y Combinación Espacial mediante PCA

**Tiempo asignado:** 1:20 - 2:50

**Texto a exponer:**

La primera etapa del pipeline consiste en el preprocesamiento de la señal.

Las señales de EEG tienen amplitudes muy pequeñas, del orden de los microvoltios, por lo que son especialmente sensibles al ruido y a diferentes tipos de interferencias. Para acondicionarlas se aplicó un filtro FIR pasa-banda entre 1 y 40 Hz de fase cero bidireccional, preservando la fase original de las oscilaciones neuronales.

Para evitar los transitorios de respuesta impulsional en los extremos del registro generados por el pasa-alto de 1 Hz, se recortaron tres segundos al inicio y al final de cada archivo.

A continuación, seleccionamos los canales occipitales O1, Oz y O2, ubicados anatómicamente sobre el lóbulo occipital y la corteza visual primaria, como se ilustra en el esquema topográfico 10-20.

Sobre estos tres sensores adyacentes aplicamos PCA para sintetizar la actividad coherente. La primera componente principal explicó el **90,61 %** de la varianza total.

Matemáticamente, los coeficientes obtenidos, alrededor de 0,58 para cada canal, convergen de forma natural al vector unitario balanceado uno sobre raíz de tres. Esto es clave: significa que el algoritmo no supervisado descubrió por sí solo que el promedio espacial simple de los tres electrodos era la combinación óptima para maximizar la varianza común compartida por conducción de volumen.

En las trazas temporales de la figura se aprecia con claridad la transición: un trazado de baja amplitud en ojos abiertos versus una oscilación rítmica periódica de gran amplitud en ojos cerrados.

---

### Diapositiva 3: Análisis Espectral y Cuantificación del Efecto Berger

**Tiempo asignado:** 2:50 - 4:20

**Texto a exponer:**

Una vez obtenida la componente principal, caracterizamos su contenido frecuencial mediante la Densidad Espectral de Potencia.

Para lograr una estimación espectral consistente y de baja varianza utilizamos el método de Welch, empleando ventanas de Hann de dos segundos con un 50 % de solapamiento. Esto establece un espaciado entre muestras de la DFT de 0,5 Hz, con una resolución física de Rayleigh aproximada de 0,75 Hz acorde al ancho del lóbulo principal de la ventana.

En el panel lineal de la Figura 2 observamos el contraste clásico del Efecto Berger: en la condición de ojos abiertos predomina una desincronización cortical con baja potencia espectral, mientras que al cerrar los ojos emerge un pico oscilatorio resonante muy marcado centrado en 10,0 Hz.

La potencia integrada en la banda alfa de 8 a 12 Hz se incrementa **16 veces (alrededor de 12 dB)** respecto a la condición basal.

Complementariamente, en el panel semilogarítmico en decibelios vemos que el pico oscilatorio de 10 Hz emerge casi **20 dB por encima del piso aperiódico 1/f**, característico de la actividad electrofisiológica cerebral asincrónica.

---

### Diapositiva 4: Análisis Tiempo-Frecuencia Dinámico mediante STFT

**Tiempo asignado:** 4:20 - 5:50

**Texto a exponer:**

Dado que las bioseñales electroencefalográficas son procesos no estacionarios, complementamos el análisis espectral con la Transformada de Fourier de Tiempo Reducido, o STFT discreta.

Bajo el principio de incertidumbre de Gabor-Heisenberg, la resolución temporal física está gobernada por la duración de la ventana de análisis de dos segundos (320 muestras). Para mapear la evolución de forma continua y suave, desplazamos la ventana con un paso temporal o hop size de 0,25 segundos (40 muestras), lo que equivale a un solapamiento del 87,5 %.

En la Figura 3 calibramos el rango dinámico en 36 decibelios para visualizar tanto la actividad de fondo como las oscilaciones dominantes en una escala cuantitativa unificada.

En el panel superior de ojos abiertos se aprecia un espectro desincronizado y homogéneo a lo largo de los 55 segundos.

En el panel inferior de ojos cerrados se observa una banda prominente y continua en torno a los 10 Hz. Si bien el alto solapamiento produce un suavizado visual, es posible detectar sutiles atenuaciones transitorias de potencia (claramente señaladas con marcadores rojos en los segundos 12, 26 y 46), vinculadas a variaciones dinámicas del reposo que analizaremos en la etapa de clustering.

---

### Diapositiva 5: Minería de Series Temporales y Agrupamiento con K-Means

**Tiempo asignado:** 5:50 - 7:40

**Texto a exponer:**

En la etapa de minería de series temporales, el objetivo fue comprobar si un algoritmo no supervisado es capaz de particionar automáticamente los estados neurofisiológicos sin disponer de etiquetas previas durante el entrenamiento.

Para evitar muestras redundantes y respetar la independencia estadística requerida por el clustering, dividimos la componente principal en **54 épocas temporales disjuntas de dos segundos** (27 épocas por condición).

Sobre cada época calculamos la potencia relativa en la banda alfa y en la banda beta normalizadas por la energía total de 1 a 40 Hz. Esta normalización composicional otorga robustez frente a posibles derivas lentas de impedancia del electrodo, operando la banda alfa como el eje primario de discriminación.

Tras estandarizar las características para operar en una escala homogénea, aplicamos K-Means con k=2. Evaluamos la estructura del agrupamiento mediante métricas formales no supervisadas: [pausa breve] obtuvimos un **Adjusted Rand Index de 0,786** y un **Silhouette Score de 0,487**.

Al contrastar los clústeres asignados con las condiciones experimentales mediante un mapeo semántico post-hoc, la concordancia alcanza el **94,44 %**, separando perfectamente todas las épocas de ojos abiertos (27/27) y 24 de 27 de ojos cerrados.

Las 3 épocas de ojos cerrados asignadas al otro grupo (resaltadas en naranja) ocurrieron en los segundos 12, 26 y 46 del registro, coincidiendo con las breves desincronizaciones de alfa visibles en el espectrograma, atribuibles a micro-modulaciones atencionales o pequeñas fluctuaciones del reposo.

---

### Diapositiva 6: Conclusiones y Perspectivas

**Tiempo asignado:** 7:40 - 9:00

**Texto a exponer:**

En conclusión, este trabajo muestra cómo articular las herramientas de procesamiento de señales y minería de series temporales en un pipeline computacional integrado para bioseñales de EEG.

El preprocesamiento digital y la combinación espacial por PCA permitieron sintetizar la actividad del polo occipital maximizando la varianza común sin introducir distorsión de fase.

El análisis espectral de Welch y la STFT caracterizaron con precisión la dinámica periódica del Efecto Berger respecto al fondo aperiódico, revelando una amplificación de 16 veces (12 dB) en la potencia alfa centrada en 10 Hz.

Finalmente, las técnicas de minería de series temporales en un espacio composicional de épocas disjuntas permitieron validar la separabilidad no supervisada de los estados basales con un Adjusted Rand Index de 0,786 y 94,44 % de concordancia.

Como consideraciones metodológicas para futuras extensiones, se destacan: evaluar esquemas inter-sujeto (como Leave-One-Subject-Out), calibrar la frecuencia individual alfa (IAF), incorporar módulos de limpieza de artefactos por ICA y migrar a filtros digitales causales sobre buffers deslizantes para aplicaciones en tiempo real.

Muchas gracias por su atención.
