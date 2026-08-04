# Guion de Presentación (9 minutos)

**Materia:** Procesamiento Avanzado de Señales y Minería de Series Temporales  
**Tema:** Decodificación de Dinámicas Cerebrales en EEG mediante Reducción Espacial por PCA, STFT y Clustering K-Means  

---

### Diapositiva 1: Introducción y objetivo del trabajo

**Tiempo asignado:** 0:00 - 1:20

**Texto a exponer:**

En este trabajo presento un pipeline computacional para analizar señales reales de electroencefalografía, o EEG, con el objetivo de identificar cambios en la actividad cerebral asociados al ritmo alfa.

El proyecto integra los principales contenidos vistos durante la materia.

En la Unidad 1, se trabajó sobre el preprocesamiento de señales: filtrado digital, reducción de dimensionalidad mediante Análisis de Componentes Principales, o PCA, y representaciones tiempo-frecuencia utilizando la Transformada de Fourier de Tiempo Reducido, conocida como STFT.

En la Unidad 2, se aplicaron técnicas de minería de series temporales. Para eso se segmentó la señal en ventanas temporales y se extrajeron características espectrales que luego fueron utilizadas por un algoritmo de clustering no supervisado.

Finalmente, en la Unidad 3, todos estos métodos se aplicaron a un problema clásico de neurofisiología: la detección del ritmo alfa occipital y el efecto Berger.

Como conjunto de datos se utilizaron registros públicos del dataset PhysioNet EEGMMIDB, específicamente el sujeto S001, con señales de 64 canales muestreadas a 160 Hz en dos condiciones de reposo: ojos abiertos y ojos cerrados.

---

### Diapositiva 2: Preprocesamiento y reducción espacial mediante PCA

**Tiempo asignado:** 1:20 - 2:50

**Texto a exponer:**

La primera etapa del pipeline consiste en el preprocesamiento de la señal.

Las señales de EEG tienen amplitudes muy pequeñas, del orden de los microvoltios, por lo que son especialmente sensibles al ruido y a diferentes tipos de interferencias. Para reducir estos efectos se aplicó un filtro FIR pasa-banda entre 1 y 40 Hz utilizando filtrado de fase cero bidireccional. Esta técnica elimina el ruido sin introducir desplazamientos temporales en la señal.

Como este tipo de filtro genera transitorios al comienzo y al final del registro, se descartaron tres segundos en ambos extremos para evitar que esos efectos afectaran el análisis.

Luego se seleccionaron únicamente los canales occipitales O1, Oz y O2, ya que son los electrodos donde el ritmo alfa suele observarse con mayor intensidad.

Sobre estos tres canales se aplicó PCA, una técnica de reducción de dimensionalidad que permite sintetizar la información común en una única componente principal.

La primera componente explicó el 90,61 % de la varianza total, indicando que resume prácticamente toda la información compartida entre los tres electrodos.

En la figura puede verse que, mientras con los ojos abiertos la señal presenta baja amplitud y un comportamiento desorganizado, con los ojos cerrados aparece una oscilación periódica muy clara alrededor de los 10 Hz, correspondiente al ritmo alfa.

---

### Diapositiva 3: Análisis espectral y efecto Berger

**Tiempo asignado:** 2:50 - 4:20

**Texto a exponer:**

Una vez obtenida la componente principal, se analizó su contenido en frecuencia mediante la Densidad Espectral de Potencia.

Para estimar el espectro se utilizó el método de Welch con ventanas Hann de dos segundos y un 50 % de solapamiento, logrando una resolución de 0,5 Hz.

En la figura se comparan los espectros correspondientes a las dos condiciones experimentales.

Con los ojos abiertos, la potencia en la banda alfa permanece baja debido a la estimulación visual constante.

En cambio, al cerrar los ojos desaparece la entrada de información visual y las neuronas de la corteza occipital comienzan a sincronizar su actividad. Como consecuencia, aparece un pico muy marcado alrededor de los 10 Hz, fenómeno conocido como efecto Berger.

La potencia integrada en la banda alfa aumenta aproximadamente 16 veces respecto de la condición de ojos abiertos, mostrando una separación muy clara entre ambos estados.

---

### Diapositiva 4: Análisis tiempo-frecuencia mediante STFT

**Tiempo asignado:** 4:20 - 5:50

**Texto a exponer:**

Hasta este punto el análisis fue realizado únicamente en frecuencia.

Sin embargo, las señales biológicas son procesos no estacionarios, por lo que también resulta importante conocer cómo evoluciona el contenido espectral a lo largo del tiempo.

Para eso se utilizó la Transformada de Fourier de Tiempo Reducido, o STFT.

Esta técnica desplaza una ventana temporal sobre la señal y calcula el espectro en cada posición, generando un espectrograma donde puede observarse simultáneamente la evolución temporal y la distribución en frecuencia.

En el espectrograma correspondiente a ojos abiertos, la energía en la banda alfa es baja durante prácticamente todo el registro.

En cambio, en la condición de ojos cerrados aparece una banda brillante y estable alrededor de los 10 Hz, mostrando que la actividad alfa permanece sostenida durante toda la adquisición.

Esto confirma visualmente lo observado previamente en el análisis espectral.

---

### Diapositiva 5: Minería de series temporales y clustering

**Tiempo asignado:** 5:50 - 7:40

**Texto a exponer:**

La última etapa del trabajo corresponde a la minería de series temporales.

El objetivo fue evaluar si un algoritmo de aprendizaje no supervisado podía distinguir automáticamente entre las condiciones de ojos abiertos y ojos cerrados sin utilizar etiquetas.

Para ello, la componente principal se dividió en 54 ventanas independientes de dos segundos.

Sobre cada ventana se calculó la potencia espectral y se construyeron dos atributos:

la potencia relativa en la banda alfa,  
y la potencia relativa en la banda beta,  

ambas normalizadas respecto de la potencia total entre 1 y 40 Hz.

Esta normalización reduce la influencia de factores externos, como diferencias de amplitud producidas por la impedancia de los electrodos.

Luego de estandarizar las variables, se aplicó K-Means con k igual a 2.

El resultado muestra que el algoritmo logró separar correctamente la mayoría de las ventanas, obteniendo un Adjusted Rand Index de 0,786, un Silhouette Score de 0,487 y una concordancia del 94,44 % respecto de las etiquetas reales.

Las pocas ventanas clasificadas de forma diferente probablemente reflejan variaciones transitorias en la actividad cerebral durante el registro.

---

### Diapositiva 6: Conclusiones

**Tiempo asignado:** 7:40 - 9:00

**Texto a exponer:**

Como conclusión, este trabajo muestra cómo las distintas herramientas vistas durante la materia pueden integrarse en un único pipeline de análisis de señales de EEG.

El preprocesamiento y la reducción espacial mediante PCA permitieron sintetizar la actividad de la corteza occipital en una única señal representativa.

El análisis espectral y la STFT permitieron identificar y caracterizar el ritmo alfa tanto en frecuencia como en el tiempo.

Finalmente, las técnicas de minería de series temporales mostraron que un algoritmo no supervisado puede diferenciar automáticamente los estados de ojos abiertos y ojos cerrados utilizando únicamente características espectrales.

Si bien este estudio se realizó sobre un único sujeto como prueba de concepto, el pipeline podría extenderse incorporando técnicas de eliminación de artefactos, evaluaciones sobre múltiples sujetos y ventanas temporales solapadas para aplicaciones en interfaces cerebro-computadora.

En conjunto, los resultados muestran que las herramientas desarrolladas durante la materia permiten construir un flujo completo para el procesamiento y análisis de señales EEG, desde el preprocesamiento hasta la extracción automática de patrones neurofisiológicos.
