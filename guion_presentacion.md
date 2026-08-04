# Guion de Presentación (9 minutos) - Versión Blindada y Rigurosa

**Materia:** Procesamiento Avanzado de Señales y Minería de Series Temporales  
**Tema:** Decodificación de Dinámicas Cerebrales en EEG mediante Reducción Espacial por PCA, STFT y Clustering K-Means  

---

### Diapositiva 1: Introducción y objetivo del trabajo

**Tiempo asignado:** 0:00 - 1:20

**Texto a exponer:**

En este trabajo presento un pipeline computacional para analizar señales reales de electroencefalografía, o EEG, con el objetivo de identificar y cuantificar cambios en la actividad cerebral asociados al ritmo alfa occipital.

El proyecto articula de forma directa los contenidos desarrollados a lo largo de la materia.

En la **Unidad 1**, se implementó el preprocesamiento de señales: filtrado digital pasa-banda de fase cero, combinación espacial mediante Análisis de Componentes Principales, o PCA, y representaciones tiempo-frecuencia con la Transformada de Fourier de Tiempo Reducido, o STFT.

En la **Unidad 2**, se aplicaron técnicas de minería de series temporales, segmentando la señal en épocas temporales disjuntas y extrayendo atributos de potencia espectral relativa para clustering no supervisado.

En la **Unidad 3**, estos métodos se aplicaron a un biomarcador neurofisiológico clásico: la modulación del ritmo alfa occipital y el efecto Berger.

Como prueba de concepto metodológica, se analizaron registros del dataset público PhysioNet EEGMMIDB, correspondientes al sujeto S001, muestreados a 160 Hz durante dos condiciones de reposo continuo: ojos abiertos y ojos cerrados.

---

### Diapositiva 2: Preprocesamiento y reducción espacial mediante PCA

**Tiempo asignado:** 1:20 - 2:50

**Texto a exponer:**

La primera etapa del pipeline consiste en el preprocesamiento de la señal.

Las señales de EEG presentan amplitudes del orden de los microvoltios y son muy vulnerables al ruido de baja frecuencia y a interferencias de línea. Para acondicionarlas, se aplicó un filtro FIR pasa-banda entre 1 y 40 Hz mediante filtrado de fase cero bidireccional, garantizando distorsión de fase nula.

Para eliminar los transitorios de borde característicos del filtro pasa-alto, se descartaron los primeros y últimos tres segundos de cada registro.

Posteriormente, se seleccionaron los tres canales occipitales contiguos: O1, Oz y O2, ubicados sobre la corteza visual primaria.

Sobre estos sensores se aplicó PCA. La primera componente explicó el 90,61 % de la varianza total con coeficientes prácticamente balanceados entre 0,57 y 0,58.

Desde el punto de vista biofísico, debido a la fuerte conducción de volumen entre electrodos adyacentes, esta primera componente opera como un filtro espacial de promedio óptimo que realza la señal cerebral común y atenúa el ruido incoherente de los sensores.

En la figura temporal se observa con claridad cómo, mientras en ojos abiertos predomina una señal desincronizada de baja amplitud, en ojos cerrados emerge una oscilación periódica prominente cercana a los 10 Hz.

---

### Diapositiva 3: Análisis espectral y efecto Berger

**Tiempo asignado:** 2:50 - 4:20

**Texto a exponer:**

Una vez obtenida la componente principal espacial, se analizó su contenido en frecuencia a través de la Densidad Espectral de Potencia.

Para estimar el espectro se utilizó el método de Welch con ventanas Hann de dos segundos y un 50 % de solapamiento, logrando un espaciado entre bins discretos de 0,5 Hz.

En la figura se contrastan los espectros de ambas condiciones experimentales.

Con los ojos abiertos, la potencia en la banda alfa permanece atenuada debido a la desincronización cortical continua inducida por la estimulación visual.

Al cerrar los ojos, al cesar la entrada visual, los circuitos tálamo-corticales occipitales entran en un régimen de sincronización masiva en reposo, generando un pico resonante en 10 Hz conocido clásicamente como Efecto Berger.

La potencia integrada en la banda alfa se incrementa en aproximadamente 16 veces respecto a la condición de ojos abiertos. El panel logarítmico permite verificar además la dinámica aperiódica de fondo tipo uno sobre efe.

---

### Diapositiva 4: Análisis tiempo-frecuencia mediante STFT

**Tiempo asignado:** 4:20 - 5:50

**Texto a exponer:**

Hasta este punto, el análisis espectral de Welch proporcionó una caracterización promedio global en frecuencia.

Sin embargo, para verificar si la oscilación observada corresponde a un fenómeno biológico sostenido o a descargas transitorias aisladas, se aplicó la Transformada de Fourier de Tiempo Reducido, o STFT.

Se utilizó la formulación discreta con ventanas Hann de dos segundos y un solapamiento del 87,5 %, generando una resolución temporal de 0,25 segundos.

En el espectrograma de ojos abiertos, la energía en la banda alfa permanece uniformemente baja a lo largo de todo el tramo temporal.

En la condición de ojos cerrados, en cambio, se aprecia una banda de alta potencia continua y estable centrada en los 10 Hz durante prácticamente la totalidad de los 55 segundos de registro.

Esta representación confirma empíricamente la estabilidad temporal y persistencia del ritmo alfa occipital en este sujeto.

---

### Diapositiva 5: Minería de series temporales y clustering

**Tiempo asignado:** 5:50 - 7:40

**Texto a exponer:**

La última etapa del trabajo corresponde a la minería de series temporales.

El objetivo fue evaluar si un algoritmo de agrupamiento no supervisado es capaz de particionar automáticamente los estados cerebrales sin emplear etiquetas durante el ajuste.

Para ello, la componente principal se segmentó en 54 épocas temporales disjuntas de dos segundos: 27 de ojos abiertos y 27 de ojos cerrados.

En cada época se extrajeron dos características normalizadas: la potencia relativa en alfa y la potencia relativa en beta, divididas por la energía de banda ancha entre 1 y 40 Hz.

Esta normalización relativa reduce la variabilidad por impedancia de contacto de los sensores. Es importante notar que la correlación negativa observada en el plano 2D responde en gran parte a la restricción composicional del denominador total ante la subida del pico alfa.

Tras estandarizar las características, se aplicó K-Means con k igual a 2. El algoritmo logró una estructura de partición muy definida, con un Adjusted Rand Index de 0,786 y un Silhouette Score de 0,487.

Al asociar el clúster de mayor potencia alfa a la condición de ojos cerrados según la hipótesis biofísica de Berger, se obtiene una concordancia del 94,44 % con el ground truth.

Las tres ventanas discrepantes corresponden a épocas de baja sincronización transitoria, artefactos residuales o ruido en esa ventana específica.

---

### Diapositiva 6: Conclusiones

**Tiempo asignado:** 7:40 - 9:00

**Texto a exponer:**

Como conclusión, este trabajo demuestra cómo las herramientas de procesamiento de señales y minería temporal vistas a lo largo del curso se integran en un pipeline coherente y reproducible para el análisis de EEG.

El preprocesamiento FIR de fase cero y la combinación espacial por PCA permitieron sintetizar la actividad de la corteza visual occipital maximizando la energía compartida y mejorando la relación señal-ruido.

El análisis espectral de Welch y la STFT caracterizaron con precisión el ritmo alfa a 10 Hz y confirmaron su persistencia temporal continua durante el reposo.

En la etapa de minería, el espacio 2D de potencia espectral relativa permitió a un algoritmo no supervisado como K-Means particionar los estados cerebrales con un alto índice de Rand ajustado.

Reconociendo que este análisis se realizó sobre un único sujeto en bloques continuos como prueba de concepto metodológica, la proyección natural hacia interfaces cerebro-computadora requerirá validaciones inter-sujeto con validación cruzada Leave-One-Subject-Out, calibración de frecuencias individuales alfa y técnicas de desmezcla ciega de artefactos por ICA.

Muchas gracias. Quedo a disposición de las preguntas del tribunal.

