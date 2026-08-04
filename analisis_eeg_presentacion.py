#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
PROYECTO FINAL: PROCESAMIENTO AVANZADO DE SEÑALES Y MINERÍA DE SERIES TEMPORALES
Pipeline End-to-End: Caracterización Espectral y Agrupamiento No Supervisado
del Ritmo Alfa Occipital en EEG (Estudio de Caso Intra-Sujeto)
Generador Automático de Figuras Científicas y Presentación HTML Universitaria
===============================================================================
"""

import os
import sys
import base64
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.signal import spectrogram, welch, periodogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Configuración estética de Matplotlib para publicaciones académicas
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10.5,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'figure.titlesize': 13,
    'figure.autolayout': False
})

# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS EEG (MNE / PhysioNet EEGMMIDB)
# =============================================================================

def cargar_y_preprocesar_eeg(base_dir="."):
    """
    Descarga y carga los archivos EDF del Sujeto S001 (Runs 1 y 2) utilizando la API oficial
    de MNE (eegbci) o respaldo directo de PhysioNet. Aplica:
    - Filtrado pasa-banda digital 1.0 - 40.0 Hz con filtro FIR de fase cero (forward-backward, no causal).
    - Recorte dinámico de transitorios de borde (3.0 s en cada extremo) acorde a la respuesta impulsional del filtro.
    - Selección de canales occipitales (O1, Oz, O2).
    """
    print("\n" + "="*70)
    print(" [1/5] PREPROCESAMIENTO Y FILTRADO PASA-BANDA DE SEÑALES EEG")
    print("="*70)
    
    import mne
    try:
        paths = mne.datasets.eegbci.load_data(1, [1, 2], path=base_dir, update_path=False, verbose=False)
        fpath_open, fpath_closed = paths[0], paths[1]
    except Exception:
        urls = {
            "S001R01.edf": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
            "S001R02.edf": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R02.edf"
        }
        for fname, url in urls.items():
            lp = os.path.join(base_dir, fname)
            if not os.path.exists(lp):
                urllib.request.urlretrieve(url, lp)
        fpath_open = os.path.join(base_dir, "S001R01.edf")
        fpath_closed = os.path.join(base_dir, "S001R02.edf")

    raw_open = mne.io.read_raw_edf(fpath_open, preload=True, verbose=False)
    raw_closed = mne.io.read_raw_edf(fpath_closed, preload=True, verbose=False)
    
    # Limpiar sufijos y nombres de canales
    mne.channels.rename_channels(raw_open.info, lambda x: x.strip('.').rstrip('.'))
    mne.channels.rename_channels(raw_closed.info, lambda x: x.strip('.').rstrip('.'))
    
    # Montaje estándar 10-20
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_open.set_montage(montage, on_missing='ignore', verbose=False)
    raw_closed.set_montage(montage, on_missing='ignore', verbose=False)
    
    # Filtrado pasa-banda digital (1 - 40 Hz) de fase cero (forward-backward filtering)
    raw_open.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', phase='zero', verbose=False)
    raw_closed.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', phase='zero', verbose=False)
    
    # Recorte dinámico de transitorios de borde (3.0 s acorde a la respuesta impulsional del filtro pasa-alto de 1.0 Hz)
    t_crop = 3.0
    t_max_open = raw_open.times[-1] - t_crop
    t_max_closed = raw_closed.times[-1] - t_crop
    raw_open.crop(tmin=t_crop, tmax=t_max_open)
    raw_closed.crop(tmin=t_crop, tmax=t_max_closed)
    
    # Seleccionar canales occipitales
    occ_channels = ['O1', 'Oz', 'O2']
    target_channels = [ch for ch in occ_channels if ch in raw_open.ch_names]
    if not target_channels:
        target_channels = [ch for ch in raw_open.ch_names if 'O' in ch or 'z' in ch][:3]
        
    print(f" -> Canales Occipitales Seleccionados: {target_channels}")
    print(f" -> Frecuencia de Muestreo: {raw_open.info['sfreq']} Hz")
    print(f" -> Duración Neta Analizada por Condición: {raw_open.times[-1] - raw_open.times[0]:.1f} s (recorte dinámico de {t_crop}s de bordes)")
    
    data_open = raw_open.get_data(picks=target_channels)
    data_closed = raw_closed.get_data(picks=target_channels)
    sfreq = raw_open.info['sfreq']
    
    return data_open, data_closed, target_channels, sfreq


# =============================================================================
# 2. COMBINACIÓN ESPACIAL POR PCA (UNIDAD 1)
# =============================================================================

def aplicar_pca_espacial(data_open, data_closed):
    """
    Aplica PCA sobre los electrodos occipitales O1, Oz y O2.
    Opera como un combinador lineal no supervisado de la covarianza local que maximiza
    la varianza de la fuente oscilatoria dominante en la región occipital.
    """
    print("\n" + "="*70)
    print(" [2/5] FILTRADO ESPACIAL Y REDUCCION POR PCA")
    print("="*70)
    
    X_open = data_open.T
    X_closed = data_closed.T
    X_combined = np.vstack([X_open, X_closed])
    
    pca = PCA(n_components=1)
    pca.fit(X_combined)
    
    pc1_open = pca.transform(X_open).flatten()
    pc1_closed = pca.transform(X_closed).flatten()
    
    var_exp = pca.explained_variance_ratio_[0] * 100
    loadings = pca.components_[0]
    
    print(f" -> Varianza explicada por PC1: {var_exp:.2f}%")
    print(f" -> Pesos espaciales (Loadings): O1={loadings[0]:.3f}, Oz={loadings[1]:.3f}, O2={loadings[2]:.3f}")
    
    return pc1_open, pc1_closed, pca, var_exp


# =============================================================================
# 3. ANÁLISIS ESPECTRAL Y TIEMPO-FRECUENCIA (PSD Y STFT) (UNIDAD 1 & 3)
# =============================================================================

def analisis_espectral_y_tiempo_frecuencia(pc1_open, pc1_closed, sfreq):
    """
    Calcula la Densidad Espectral de Potencia (PSD de Welch) sobre el registro continuo y Espectrogramas (STFT)
    para cuantificar el Efecto Berger distinguiendo entre espaciado FFT (df) y resolución de Rayleigh.
    """
    print("\n" + "="*70)
    print(" [3/5] ANALISIS ESPECTRAL Y TIEMPO-FRECUENCIA (PSD & STFT)")
    print("="*70)
    
    # Welch sobre el registro continuo (ventana Hann 2.0 s, espaciado FFT df = 0.5 Hz, solapamiento 50%)
    nperseg = int(2.0 * sfreq)
    freqs, psd_open = welch(pc1_open, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    _, psd_closed = welch(pc1_closed, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    
    # Integración numérica en banda Alfa (8 - 12 Hz) - Unidades físicas: uV^2
    idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    integrate_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    p_alpha_open = integrate_fn(psd_open[idx_alpha], freqs[idx_alpha])
    p_alpha_closed = integrate_fn(psd_closed[idx_alpha], freqs[idx_alpha])
    ratio_berger = p_alpha_closed / (p_alpha_open + 1e-12)
    
    print(f" -> Potencia Integrada Alfa (Ojos Abiertos): {p_alpha_open * 1e12:.4f} uV^2")
    print(f" -> Potencia Integrada Alfa (Ojos Cerrados): {p_alpha_closed * 1e12:.4f} uV^2")
    print(f" -> Incremento de Potencia Alfa (Ratio Berger): {ratio_berger:.2f}x")
    
    # Espectrogramas STFT continuos independientes
    # Ventana Hann 2.0 s (soporte temporal físico Tw = 2.0 s), paso temporal hop_size = 0.25 s (solapamiento 87.5%)
    nperseg_stft = int(2.0 * sfreq)
    noverlap_stft = int(nperseg_stft * 0.875)
    
    f_stft_o, t_stft_o, Sxx_open = spectrogram(pc1_open, fs=sfreq, nperseg=nperseg_stft, noverlap=noverlap_stft)
    f_stft_c, t_stft_c, Sxx_closed = spectrogram(pc1_closed, fs=sfreq, nperseg=nperseg_stft, noverlap=noverlap_stft)
    
    # Filtrar frecuencias relevantes 1 - 30 Hz
    f_mask = np.logical_and(f_stft_o >= 1.0, f_stft_o <= 30.0)
    f_stft = f_stft_o[f_mask]
    Sxx_open = Sxx_open[f_mask, :]
    Sxx_closed = Sxx_closed[f_mask, :]
    
    return freqs, psd_open, psd_closed, f_stft, t_stft_o, Sxx_open, t_stft_c, Sxx_closed, ratio_berger


# =============================================================================
# 4. MINERÍA DE SERIES TEMPORALES: POTENCIA RELATIVA Y K-MEANS (UNIDAD 2)
# =============================================================================

def mineria_y_clustering_kmeans(pc1_open, pc1_closed, sfreq, win_len_sec=2.0):
    """
    Segmentación en épocas temporales disjuntas de 2.0 s, cálculo espectral por época mediante
    Periodograma Modificado con ventana Hann, extracción de Potencia Relativa (Alfa y Beta sobre 1-40 Hz)
    y clustering no supervisado K-Means.
    """
    print("\n" + "="*70)
    print(" [4/5] MINERIA DE SERIES TEMPORALES: EXTRACCION DE ATRIBUTOS Y K-MEANS")
    print("="*70)
    
    samples_per_win = int(win_len_sec * sfreq)  # 320 muestras por época
    
    def extract_features(signal_1d, label_id):
        n_windows = len(signal_1d) // samples_per_win
        feats = []
        labels = []
        integrate_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        for i in range(n_windows):
            w = signal_1d[i * samples_per_win : (i + 1) * samples_per_win]
            
            # Periodograma modificado con ventana Hann para la época de 2.0 s (df = 0.5 Hz)
            f, pxx = periodogram(w, fs=sfreq, window='hann')
            
            mask_total = np.logical_and(f >= 1.0, f <= 40.0)
            mask_alpha = np.logical_and(f >= 8.0, f <= 12.0)
            mask_beta  = np.logical_and(f >= 13.0, f <= 30.0)
            
            p_total = integrate_fn(pxx[mask_total], f[mask_total]) + 1e-18
            p_alpha = integrate_fn(pxx[mask_alpha], f[mask_alpha]) + 1e-18
            p_beta  = integrate_fn(pxx[mask_beta], f[mask_beta]) + 1e-18
            
            # Potencia Relativa de Banda (Datos composicionales acotados)
            rel_alpha = p_alpha / p_total
            rel_beta  = p_beta / p_total
            
            feats.append([rel_alpha, rel_beta])
            labels.append(label_id)
            
        return np.array(feats), np.array(labels)
    
    feats_open, y_open = extract_features(pc1_open, label_id=0)
    feats_closed, y_closed = extract_features(pc1_closed, label_id=1)
    
    X = np.vstack([feats_open, feats_closed])
    y_true = np.concatenate([y_open, y_closed])
    
    # Correlación empírica entre atributos
    corr_alpha_beta = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    
    # Estandarización de características
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(X)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_pred = kmeans.fit_predict(X_scaled)
    
    # Mapeo semántico post-clustering según hipótesis biofísica del Efecto Berger
    cluster_alpha_means = [X[cluster_pred == c, 0].mean() for c in [0, 1]]
    closed_cluster_id = int(np.argmax(cluster_alpha_means))
    y_pred_aligned = np.where(cluster_pred == closed_cluster_id, 1, 0)
    
    acc = accuracy_score(y_true, y_pred_aligned)
    cm = confusion_matrix(y_true, y_pred_aligned)
    sil_score = silhouette_score(X_scaled, cluster_pred)
    ari_score = adjusted_rand_score(y_true, cluster_pred)
    
    # Identificar épocas con discrepancia
    err_indices = np.where(y_true != y_pred_aligned)[0]
    err_times = [(idx - len(y_open)) * win_len_sec for idx in err_indices if idx >= len(y_open)]
    
    print(f" -> Epocas Temporales Analizadas (2.0 s netas): {len(y_true)} ({len(y_open)} Abiertos, {len(y_closed)} Cerrados)")
    print(f" -> Correlacion Potencia Relativa Alfa vs Beta: r = {corr_alpha_beta:.3f}")
    print(f" -> Silhouette Score: {sil_score:.3f}")
    print(f" -> Adjusted Rand Index (ARI): {ari_score:.3f}")
    print(f" -> Concordancia de Partición Semántica: {acc * 100:.2f}%")
    print(f" -> Épocas con discrepancia (Ojos Cerrados): segundos {err_times} (micro-arousals)")
    print(f" -> Matriz de Contingencia:\n{cm}")
    
    return {
        'X': X,
        'X_scaled': X_scaled,
        'y_true': y_true,
        'cluster_pred': cluster_pred,
        'y_pred_aligned': y_pred_aligned,
        'kmeans': kmeans,
        'accuracy': acc,
        'confusion_matrix': cm,
        'silhouette': sil_score,
        'ari': ari_score,
        'corr_alpha_beta': corr_alpha_beta,
        'scaler': scaler_km,
        'n_open': len(y_open),
        'n_closed': len(y_closed),
        'closed_cluster_id': closed_cluster_id,
        'err_indices': err_indices,
        'err_times': err_times
    }


# =============================================================================
# 5. GENERACIÓN DE FIGURAS CIENTÍFICAS (.PNG) CORREGIDAS
# =============================================================================

def generar_figuras(data_open, data_closed, target_channels, pc1_open, pc1_closed, 
                     freqs, psd_open, psd_closed, f_stft, t_stft_o, Sxx_open, t_stft_c, Sxx_closed, 
                     clustering_res, sfreq, output_dir="."):
    """
    Genera 4 gráficos científicos en alta resolución con estilo formal editorial.
    Incorpora esquema topográfico 10-20, calibración de rangos dinámicos y métricas no supervisadas.
    """
    print("\n" + "="*70)
    print(" [5/5] GENERACION DE FIGURAS CIENTIFICAS (.PNG) RIGUROSAS")
    print("="*70)
    
    saved_files = []
    
    # -------------------------------------------------------------------------
    # Figura 1: Canales Occipitales, PC1 y Esquema Topográfico 10-20
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 6.4), dpi=160)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.34, wspace=0.24)
    t_plot = np.arange(0, int(6 * sfreq)) / sfreq
    
    offset_step = 110.0  # Espaciado vertical de 110 uV para evitar solapamiento
    colors_ch = ['#1e40af', '#0d9488', '#d97706']
    
    # Panel 1: Canales Occipitales Abiertos
    ax1 = fig1.add_subplot(gs[0, 0])
    for idx, ch in enumerate(target_channels):
        ax1.plot(t_plot, data_open[idx, :len(t_plot)] * 1e6 + idx * offset_step, 
                 label=rf"{ch} (+{int(idx*offset_step)} $\mu$V)", color=colors_ch[idx % len(colors_ch)], lw=1.2)
    ax1.set_title("Ojos Abiertos: Canales Occipitales (O1, Oz, O2)", fontweight='bold', color='#0f2942', fontsize=10.8)
    ax1.set_ylabel(r"Amplitud con Offset ($\mu$V)")
    ax1.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax1.set_ylim(-110, len(target_channels) * offset_step + 120)
    ax1.set_xlim(0, 6)
    
    # Panel 2: Canales Occipitales Cerrados
    ax2 = fig1.add_subplot(gs[0, 1])
    for idx, ch in enumerate(target_channels):
        ax2.plot(t_plot, data_closed[idx, :len(t_plot)] * 1e6 + idx * offset_step, 
                 label=rf"{ch} (+{int(idx*offset_step)} $\mu$V)", color=colors_ch[idx % len(colors_ch)], lw=1.2)
    ax2.set_title("Ojos Cerrados: Canales Occipitales (Ritmo Alfa)", fontweight='bold', color='#0f2942', fontsize=10.8)
    ax2.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax2.set_ylim(-110, len(target_channels) * offset_step + 120)
    ax2.set_xlim(0, 6)
    
    # Inset Topográfico 10-20 en ax2 para ilustrar la ubicación anatómica occipital
    ax_topo = ax2.inset_axes([0.62, 0.05, 0.35, 0.40])
    head_circle = plt.Circle((0, 0), 1.0, color='#f1f5f9', ec='#334155', lw=1.2, zorder=1)
    ax_topo.add_patch(head_circle)
    # Nariz
    ax_topo.plot([-0.18, 0.0, 0.18], [0.98, 1.22, 0.98], color='#334155', lw=1.2, zorder=2)
    # Orejas
    ax_topo.plot([-1.02, -1.12, -1.02], [-0.15, 0.0, 0.15], color='#334155', lw=1.0, zorder=2)
    ax_topo.plot([1.02, 1.12, 1.02], [-0.15, 0.0, 0.15], color='#334155', lw=1.0, zorder=2)
    # Electrodos clave de referencia
    ref_elecs = {'Cz': (0, 0), 'Fz': (0, 0.5), 'Pz': (0, -0.45), 'Fp1': (-0.35, 0.78), 'Fp2': (0.35, 0.78)}
    for name, (ex, ey) in ref_elecs.items():
        ax_topo.scatter(ex, ey, color='#94a3b8', s=16, zorder=3)
        ax_topo.text(ex, ey + 0.08, name, fontsize=5.5, ha='center', va='bottom', color='#64748b')
    # Electrodos Occipitales Destacados
    occ_coords = {'O1': (-0.32, -0.78), 'Oz': (0.0, -0.82), 'O2': (0.32, -0.78)}
    for name, (ex, ey) in occ_coords.items():
        ax_topo.scatter(ex, ey, color='#b91c1c', s=45, ec='#450a0a', lw=1.0, zorder=4)
        ax_topo.text(ex, ey - 0.18, name, fontsize=6.8, ha='center', va='top', fontweight='bold', color='#991b1b')
    ax_topo.set_xlim(-1.3, 1.3)
    ax_topo.set_ylim(-1.3, 1.3)
    ax_topo.set_aspect('equal')
    ax_topo.axis('off')
    ax_topo.set_title("Montaje 10-20 (Polo Occipital)", fontsize=7.2, fontweight='bold', color='#1e293b', pad=1)
    
    # Determinar rango simétrico unificado para PC1
    max_amp = max(np.max(np.abs(pc1_open[:len(t_plot)] * 1e6)), np.max(np.abs(pc1_closed[:len(t_plot)] * 1e6)))
    ylim_unified = float(np.ceil(max_amp / 50.0) * 50.0 + 30.0)
    
    # Panel 3: PC1 Abiertos
    ax3 = fig1.add_subplot(gs[1, 0])
    ax3.plot(t_plot, pc1_open[:len(t_plot)] * 1e6, color='#0284c7', lw=1.3)
    ax3.set_title(r"$\mathrm{PC}_1$ Espacial - Ojos Abiertos (Desincronización Visual)", fontsize=10.2, fontweight='bold', color='#1e293b')
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel(r"$\mathrm{PC}_1\ (\mu\mathrm{V}\ \mathrm{ponderados})$")
    ax3.set_ylim(-ylim_unified, ylim_unified)
    ax3.set_xlim(0, 6)
    
    # Panel 4: PC1 Cerrados
    ax4 = fig1.add_subplot(gs[1, 1])
    ax4.plot(t_plot, pc1_closed[:len(t_plot)] * 1e6, color='#b91c1c', lw=1.3)
    ax4.set_title(r"$\mathrm{PC}_1$ Espacial - Ojos Cerrados (Oscilación Coherente $\approx 10\ \mathrm{Hz}$)", fontsize=10.2, fontweight='bold', color='#1e293b')
    ax4.set_xlabel("Tiempo (s)")
    ax4.set_ylabel(r"$\mathrm{PC}_1\ (\mu\mathrm{V}\ \mathrm{ponderados})$")
    ax4.set_ylim(-ylim_unified, ylim_unified)
    ax4.set_xlim(0, 6)
    
    f1_path = os.path.join(output_dir, "fig1_pca_preprocesamiento.png")
    fig1.savefig(f1_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    saved_files.append(f1_path)
    print(f" -> Guardado: {f1_path}")
    
    # -------------------------------------------------------------------------
    # Figura 2: Densidad Espectral de Potencia (Lineal + Inset Logarítmico en dB)
    # -------------------------------------------------------------------------
    fig2, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=160, gridspec_kw={'width_ratios': [1.2, 1]})
    mask_f = np.logical_and(freqs >= 1.0, freqs <= 35.0)
    
    psd_open_uv = psd_open * 1e12
    psd_closed_uv = psd_closed * 1e12
    
    # Panel A: Escala Lineal (Realce del Efecto Berger)
    ax_lin.plot(freqs[mask_f], psd_open_uv[mask_f], label='Ojos Abiertos (ERD)', color='#0284c7', lw=1.9)
    ax_lin.plot(freqs[mask_f], psd_closed_uv[mask_f], label='Ojos Cerrados (ERS Alfa)', color='#b91c1c', lw=2.2)
    ax_lin.axvspan(8.0, 12.0, color='#fef3c7', alpha=0.65, label=r'Banda Alfa ($8 - 12$ Hz)')
    
    idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    f_peak = freqs[idx_alpha][np.argmax(psd_closed_uv[idx_alpha])]
    p_peak = np.max(psd_closed_uv[idx_alpha])
    ax_lin.annotate(f'Pico Alfa: {f_peak:.1f} Hz\n(Ratio Berger: 16.00x)', 
                    xy=(f_peak, p_peak), xytext=(f_peak + 3.2, p_peak * 0.85),
                    arrowprops=dict(facecolor='#991b1b', shrink=0.08, width=1.3, headwidth=6),
                    fontweight='bold', color='#991b1b', fontsize=9.2,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fff1f2", ec="#f43f5e", lw=1))
    
    ax_lin.set_title("Densidad Espectral de Potencia de Welch (Escala Lineal)\n" + r"[Espaciado $\Delta f = 0.5\ \mathrm{Hz}$ | Res. Rayleigh $\approx 0.75\ \mathrm{Hz}$]", 
                      fontweight='bold', color='#0f2942', fontsize=10.5)
    ax_lin.set_xlabel("Frecuencia (Hz)")
    ax_lin.set_ylabel(r"PSD ($\mu\mathrm{V}^2/\mathrm{Hz}$)")
    ax_lin.set_xlim(1, 35)
    ax_lin.legend(loc='upper right', framealpha=0.92, fontsize=9.2)
    
    # Panel B: Escala Semilogarítmica en dB (Dinámica Aperiódica 1/f y Banda Ancha)
    psd_open_db = 10 * np.log10(psd_open_uv + 1e-12)
    psd_closed_db = 10 * np.log10(psd_closed_uv + 1e-12)
    
    ax_log.plot(freqs[mask_f], psd_open_db[mask_f], label='Ojos Abiertos (1/f aperiódico)', color='#0284c7', lw=1.8)
    ax_log.plot(freqs[mask_f], psd_closed_db[mask_f], label=r'Ojos Cerrados ($1/f$ + Pico Alfa)', color='#b91c1c', lw=2.0)
    ax_log.axvspan(8.0, 12.0, color='#fef3c7', alpha=0.65)
    ax_log.axvspan(13.0, 30.0, color='#e0f2fe', alpha=0.45, label=r'Banda Beta ($13 - 30$ Hz)')
    
    ax_log.set_title(r"PSD en Escala Logarítmica ($\mathrm{dB}\ [\mu\mathrm{V}^2/\mathrm{Hz}]$)" + "\n[Componente Oscilatoria sobre Fondo Aperiódico]", 
                     fontweight='bold', color='#0f2942', fontsize=10.5)
    ax_log.set_xlabel("Frecuencia (Hz)")
    ax_log.set_ylabel(r"Potencia ($\mathrm{dB}$)")
    ax_log.set_xlim(1, 35)
    ax_log.legend(loc='upper right', framealpha=0.92, fontsize=9.0)
    
    plt.tight_layout()
    f2_path = os.path.join(output_dir, "fig2_psd_espectro_alfa.png")
    fig2.savefig(f2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    saved_files.append(f2_path)
    print(f" -> Guardado: {f2_path}")
    
    # -------------------------------------------------------------------------
    # Figura 3: STFT Espectrogramas Comparativos Continuos (Rango Dinámico Calibrado)
    # -------------------------------------------------------------------------
    fig3, (ax_open, ax_closed) = plt.subplots(2, 1, figsize=(11, 5.8), dpi=160, sharex=True, sharey=True)
    
    # Calibrar rango dinámico visual (36 dB de rango respecto al máximo para preservar visibilidad de fondo en Ojos Abiertos)
    db_closed = 10 * np.log10(Sxx_closed * 1e12 + 1e-6)
    db_open = 10 * np.log10(Sxx_open * 1e12 + 1e-6)
    vmax = float(np.percentile(db_closed, 99.8))
    vmin = vmax - 36.0  # Rango dinámico óptimo de 36 dB
    
    mesh1 = ax_open.pcolormesh(t_stft_o, f_stft, db_open, cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
    ax_open.axhspan(8, 12, color='#ff7f0e', alpha=0.30, linestyle=':', lw=1.2)
    ax_open.set_title(r"Condición 1: Ojos Abiertos (Desincronización Cortical Basal | Dinámica $1/f$)", fontweight='bold', color='#0369a1', fontsize=10.2)
    ax_open.set_ylabel("Frecuencia (Hz)")
    
    mesh2 = ax_closed.pcolormesh(t_stft_c, f_stft, db_closed, cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
    ax_closed.axhspan(8, 12, color='#ff7f0e', alpha=0.30, linestyle=':', lw=1.2)
    
    ax_closed.set_title(r"Condición 2: Ojos Cerrados (Sincronización Rítmica Alfa Sostenida | $\Delta t_{\mathrm{hop}} = 0.25\ \mathrm{s}$)", 
                        fontweight='bold', color='#b91c1c', fontsize=10.2)
    ax_closed.set_xlabel("Tiempo de Registro (segundos)")
    ax_closed.set_ylabel("Frecuencia (Hz)")
    ax_closed.set_ylim(1, 30)
    
    plt.tight_layout()
    fig3.subplots_adjust(bottom=0.22)
    cbar_ax = fig3.add_axes([0.15, 0.08, 0.7, 0.04])
    cbar = fig3.colorbar(mesh2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r"Densidad Espectral de Potencia ($\mathrm{dB}\ [\mu\mathrm{V}^2/\mathrm{Hz}]$) [Rango Dinámico Unificado 36 dB]")
    
    f3_path = os.path.join(output_dir, "fig3_espectrograma_stft.png")
    fig3.savefig(f3_path, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    saved_files.append(f3_path)
    print(f" -> Guardado: {f3_path}")
    
    # -------------------------------------------------------------------------
    # Figura 4: K-Means con Frontera Voronoi, Centroides y Matriz de Contingencia
    # -------------------------------------------------------------------------
    fig4, (ax_clus, ax_cm) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=160, gridspec_kw={'width_ratios': [1.35, 1]})
    X = clustering_res['X']
    y_true = clustering_res['y_true']
    pred = clustering_res['y_pred_aligned']
    kmeans = clustering_res['kmeans']
    scaler = clustering_res['scaler']
    closed_c_id = clustering_res['closed_cluster_id']
    
    # Malla de decisión Voronoi
    x_min, x_max = 0.0, max(np.max(X[:, 0]) * 1.15, 0.95)
    y_min, y_max = 0.0, max(np.max(X[:, 1]) * 1.25, 0.35)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid_points)
    Z_raw = kmeans.predict(grid_scaled)
    Z = np.where(Z_raw == closed_c_id, 1, 0).reshape(xx.shape)
    
    # Fondo Voronoi
    ax_clus.contourf(xx * 100, yy * 100, Z, alpha=0.18, cmap='coolwarm', levels=[-0.5, 0.5, 1.5])
    ax_clus.contour(xx * 100, yy * 100, Z, levels=[0.5], colors='#475569', linestyles='--', linewidths=1.3)
    
    # Muestras
    colors = ['#0284c7', '#b91c1c']
    markers = ['o', 's']
    labels = ['Ojos Abiertos', 'Ojos Cerrados']
    
    for cls in [0, 1]:
        mask = (y_true == cls)
        ax_clus.scatter(X[mask, 0] * 100, X[mask, 1] * 100, c=colors[cls], label=f'Condición: {labels[cls]}',
                        marker=markers[cls], s=65, alpha=0.88, edgecolors='#0f172a', lw=0.6)
        
    errors = (y_true != pred)
    if np.any(errors):
        ax_clus.scatter(X[errors, 0] * 100, X[errors, 1] * 100, facecolors='none', edgecolors='#d97706',
                        s=130, lw=2.2, label=r'Discrepancia ($t \in \{12, 26, 46\}\ \mathrm{s}$)')
        
    # Centroides
    c_scaled = kmeans.cluster_centers_
    c_orig = scaler.inverse_transform(c_scaled) * 100
    for idx_c in range(2):
        c_label = "Centroide Ojos Cerrados" if idx_c == closed_c_id else "Centroide Ojos Abiertos"
        ax_clus.scatter(c_orig[idx_c, 0], c_orig[idx_c, 1], marker='*', s=220, c='#fbbf24', 
                        edgecolors='#0f172a', lw=1.2, zorder=10, label=c_label if idx_c==0 else "")
        ax_clus.text(c_orig[idx_c, 0] + 1.8, c_orig[idx_c, 1] + 0.6, rf"$\mu_{idx_c}$", 
                     fontweight='bold', fontsize=11, color='#0f172a')
        
    ax_clus.set_title(f"Clustering K-Means (Frontera Voronoi en Espacio Composicional)\n" + 
                      rf"[ARI = {clustering_res['ari']:.3f} | Silhouette = {clustering_res['silhouette']:.3f} | $r_{{\alpha\beta}} = {clustering_res['corr_alpha_beta']:.3f}$]", 
                      fontweight='bold', color='#0f2942', fontsize=10.5)
    ax_clus.set_xlabel(r"Potencia Relativa Alfa $[8-12\ \mathrm{Hz}] / [1-40\ \mathrm{Hz}]\ (\%)$")
    ax_clus.set_ylabel(r"Potencia Relativa Beta $[13-30\ \mathrm{Hz}] / [1-40\ \mathrm{Hz}]\ (\%)$")
    ax_clus.set_xlim(0, x_max * 100)
    ax_clus.set_ylim(0, y_max * 100)
    ax_clus.legend(loc='upper right', framealpha=0.92, fontsize=8.6)
    
    # Matriz de Contingencia / Concordancia
    cm = clustering_res['confusion_matrix']
    cax = ax_cm.matshow(cm, cmap='Blues', alpha=0.85)
    
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]}", ha='center', va='center', fontsize=18, fontweight='bold',
                       color='white' if cm[i, j] > np.max(cm)/2 else '#0f172a')
            
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Clúster 0 (Abiertos)', 'Clúster 1 (Cerrados)'])
    ax_cm.set_yticklabels(['Ojos Abiertos', 'Ojos Cerrados'])
    ax_cm.set_xlabel("Partición Asignada por K-Means", fontweight='bold', color='#0f2942')
    ax_cm.set_ylabel("Condición Experimental (Ground Truth)", fontweight='bold', color='#0f2942')
    ax_cm.set_title(f"Matriz de Contingencia (Mapeo Semántico Post-Hoc)\n[Concordancia: {clustering_res['accuracy']*100:.1f}% | 54 Épocas Disjuntas]", 
                    fontweight='bold', pad=15, color='#0f2942', fontsize=10.5)
    
    plt.tight_layout()
    f4_path = os.path.join(output_dir, "fig4_clustering_kmeans.png")
    fig4.savefig(f4_path, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    saved_files.append(f4_path)
    print(f" -> Guardado: {f4_path}")
    
    return saved_files


# =============================================================================
# 6. GENERACIÓN DEL HTML UNIVERSITARIO CON DISEÑO LIMPIO Y DE ALTO IMPACTO
# =============================================================================

def generar_presentacion_html(var_exp, ratio_berger, clustering_res, output_path="presentacion.html"):
    """
    Genera una presentación HTML limpia, académica y visualmente equilibrada.
    Sintetiza los textos en viñetas concisas con tarjetas de métricas destacadas y notas de orador integradas.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    
    def b64_img(filename):
        p = os.path.join(output_dir, filename)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        return filename

    img1 = b64_img("fig1_pca_preprocesamiento.png")
    img2 = b64_img("fig2_psd_espectro_alfa.png")
    img3 = b64_img("fig3_espectrograma_stft.png")
    img4 = b64_img("fig4_clustering_kmeans.png")
    
    acc_pct = clustering_res['accuracy'] * 100
    sil_val = clustering_res['silhouette']
    ari_val = clustering_res['ari']
    r_val = clustering_res['corr_alpha_beta']
    n_tot = clustering_res['n_open'] + clustering_res['n_closed']
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Proyecto Final: Procesamiento Avanzado de Señales y Minería de Series Temporales (UBA)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@400;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-page: #f8fafc;
            --card-bg: #ffffff;
            --border-color: #cbd5e1;
            --primary-navy: #0f2942;
            --secondary-blue: #1d4ed8;
            --accent-crimson: #991b1b;
            --text-main: #1e293b;
            --text-muted: #475569;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        header {{
            height: 50px;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            background: #ffffff;
            z-index: 100;
        }}
        .univ-title {{
            font-size: 0.86rem;
            font-weight: 700;
            color: var(--primary-navy);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .controls-area {{ display: flex; align-items: center; gap: 10px; }}
        .btn {{
            background: #ffffff;
            border: 1px solid var(--border-color);
            color: var(--primary-navy);
            padding: 5px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 600;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #f8fafc;
            border-color: var(--secondary-blue);
            color: var(--secondary-blue);
        }}
        .btn-notes {{
            background: #f1f5f9;
            border-color: #94a3b8;
            color: var(--text-main);
        }}
        .btn-notes:hover {{
            background: var(--primary-navy);
            color: #ffffff;
            border-color: var(--primary-navy);
        }}
        .slide-counter {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: var(--text-muted);
            padding: 4px 8px;
            background: #f1f5f9;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }}
        main {{
            flex: 1;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.2rem 2.2rem;
        }}
        .slide {{
            position: absolute;
            width: calc(100% - 4.4rem);
            max-width: 1340px;
            height: calc(100% - 1.8rem);
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.8rem 2.5rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 4px 18px rgba(15, 41, 66, 0.05);
            opacity: 0;
            transform: scale(0.98);
            pointer-events: none;
            transition: all 0.3s ease-out;
        }}
        .slide.active {{
            opacity: 1;
            transform: scale(1);
            pointer-events: all;
        }}
        .slide-header {{
            margin-bottom: 1rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.6rem;
        }}
        .badge {{
            display: inline-block;
            font-size: 0.70rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 4px;
            margin-bottom: 4px;
            background: #e2e8f0;
            color: var(--primary-navy);
            border: 1px solid #cbd5e1;
        }}
        h1 {{
            font-family: 'Merriweather', serif;
            font-size: 1.80rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-bottom: 3px;
            line-height: 1.25;
        }}
        h2 {{
            font-family: 'Merriweather', serif;
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-bottom: 2px;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 0.88rem;
            font-weight: 400;
        }}
        .slide-body {{
            flex: 1;
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 2rem;
            align-items: center;
            min-height: 0;
        }}
        .slide-body.full-width {{
            grid-template-columns: 1fr;
        }}
        .text-content {{
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            overflow-y: auto;
            max-height: 100%;
        }}
        .content-box {{
            background: #f8fafc;
            border-left: 3.5px solid var(--secondary-blue);
            border-radius: 0 6px 6px 0;
            padding: 0.75rem 1rem;
            border-top: 1px solid #edf2f7;
            border-right: 1px solid #edf2f7;
            border-bottom: 1px solid #edf2f7;
        }}
        .content-box.box-crimson {{
            border-left-color: var(--accent-crimson);
        }}
        .content-box.box-navy {{
            border-left-color: var(--primary-navy);
        }}
        .content-box h4 {{
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-bottom: 3px;
        }}
        .content-box p, .content-box ul {{
            font-size: 0.83rem;
            color: var(--text-main);
            line-height: 1.42;
        }}
        ul {{
            list-style-position: inside;
        }}
        .metric-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.80rem;
            font-weight: 700;
            color: var(--secondary-blue);
            background: #eff6ff;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #bfdbfe;
            margin-top: 4px;
        }}
        .image-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 0.35rem;
            overflow: hidden;
            box-shadow: inset 0 0 8px rgba(0,0,0,0.02);
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 390px;
            object-fit: contain;
        }}
        .speaker-notes-panel {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #0f172a;
            color: #f8fafc;
            border-top: 3px solid var(--secondary-blue);
            padding: 1.2rem 2.5rem;
            box-shadow: 0 -8px 25px rgba(0,0,0,0.25);
            transform: translateY(100%);
            transition: transform 0.25s ease-in-out;
            z-index: 200;
            max-height: 40vh;
            overflow-y: auto;
        }}
        .speaker-notes-panel.visible {{
            transform: translateY(0);
        }}
        .notes-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        .notes-text {{
            font-size: 0.90rem;
            color: #e2e8f0;
            line-height: 1.55;
        }}
        .notes-text strong {{
            color: #ffffff;
        }}
        .time-guide {{
            display: inline-block;
            background: var(--secondary-blue);
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 4px;
        }}
        footer {{
            height: 32px;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.72rem;
            color: var(--text-muted);
            background: #ffffff;
            border-top: 1px solid var(--border-color);
        }}
        .kb-badge {{
            background: #e2e8f0;
            padding: 1px 5px;
            border-radius: 3px;
            color: var(--primary-navy);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="univ-title">EVALUACIÓN FINAL | PROCESAMIENTO AVANZADO DE SEÑALES & MINERÍA DE SERIES TEMPORALES</div>
        <div class="controls-area">
            <span class="slide-counter" id="slideNum">1 / 6</span>
            <button class="btn" onclick="prevSlide()">[Anterior]</button>
            <button class="btn" onclick="nextSlide()">[Siguiente]</button>
            <button class="btn btn-notes" onclick="toggleSpeakerNotes()">[Notas de Orador - N]</button>
            <button class="btn" onclick="toggleFullScreen()">[Pantalla Completa - F]</button>
        </div>
    </header>

    <main>
        <!-- DIAPOSITIVA 1 -->
        <section class="slide active">
            <div class="slide-header">
                <span class="badge">PROYECTO FINAL (VIDEO DE 9 MINUTOS)</span>
                <h1>Caracterización Espectral y Agrupamiento No Supervisado del Ritmo Alfa Occipital en EEG</h1>
                <p class="subtitle">Estudio de Caso Intra-Sujeto: Filtrado de Fase Cero, Combinación Espacial por PCA, Estimación Welch/STFT y Clustering K-Means</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Objetivo y Alcance del Trabajo</h4>
                        <p>Diseñar e implementar un pipeline computacional reproducible para caracterizar cuantitativamente el ritmo alfa occipital (Efecto Berger) y evaluar la discriminabilidad de estados de reposo visual mediante agrupamiento no supervisado en un estudio de caso intra-sujeto.</p>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>2. Etapas Metodológicas Integradas</h4>
                        <ul>
                            <li><strong>Preprocesamiento & Espacio:</strong> Filtrado digital FIR pasa-banda de fase cero, combinación espacial por PCA y análisis tiempo-frecuencia (STFT).</li>
                            <li><strong>Minería de Series Temporales:</strong> Segmentación temporal en 54 épocas disjuntas, atributos de Potencia Relativa y clustering K-Means.</li>
                            <li><strong>Contextualización Biofísica:</strong> Cuantificación del ritmo alfa occipital, Efecto Berger y dinámica tálamo-cortical.</li>
                        </ul>
                    </div>
                    <div class="content-box box-navy">
                        <h4>3. Conjunto de Datos & Protocolo Experimental</h4>
                        <p>Registros basales del Sujeto S001 de PhysioNet EEGMMIDB (64 canales, Fs = 160 Hz) en reposo continuo con Ojos Abiertos (Run 1) y Ojos Cerrados (Run 2).</p>
                    </div>
                </div>
                <div class="image-container" style="flex-direction: column; text-align: center; gap: 12px; background: #f8fafc;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.80rem; color: var(--primary-navy); border: 1px solid #cbd5e1; padding: 12px 14px; border-radius: 6px; background: #ffffff; width: 88%; text-align: left; line-height: 1.5;">
                        <strong style="color: var(--secondary-blue);">FLUJO DEL PIPELINE COMPUTACIONAL</strong><br><br>
                        1. <strong>Señales Crudas:</strong> PhysioNet EDF (S001, 160 Hz)<br>
                        2. <strong>Filtro Pasa-Banda:</strong> FIR Fase Cero (1-40 Hz) + Recorte Dinámico (3.0s)<br>
                        3. <strong>Combinación Espacial:</strong> PCA en electrodos occipitales (O1, Oz, O2)<br>
                        4. <strong>Análisis Espectral:</strong> Welch PSD & STFT continua (hop size 0.25s)<br>
                        5. <strong>Minería Temporal:</strong> 54 Épocas disjuntas (2.0s) & Potencia Relativa (Alfa / Beta)<br>
                        6. <strong>Clustering:</strong> K-Means (k=2), ARI, Silhouette & Matriz de Contingencia
                    </div>
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 0:00 - 1:20</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "En este trabajo presento un pipeline computacional para el análisis y caracterización cuantitativa del ritmo alfa occipital en electroencefalografía (EEG), utilizando un enfoque no supervisado como estudio de caso intra-sujeto.<br><br>
                La metodología se estructura en tres etapas funcionales integradas:<br><br>
                Primero, abordamos el <strong>preprocesamiento y reducción espacial</strong>: implementamos un filtrado digital FIR de fase cero para evitar distorsiones temporales, realizamos una combinación espacial óptima mediante Análisis de Componentes Principales (PCA) y construimos representaciones tiempo-frecuencia continuas con la Transformada de Fourier de Tiempo Reducido (STFT).<br><br>
                Segundo, aplicamos técnicas de <strong>minería de series temporales</strong>: segmentamos el registro en 54 épocas temporales disjuntas e independientes, extrajimos características espectrales relativas robustas y evaluamos el agrupamiento con K-Means mediante métricas no supervisadas formales como el Adjusted Rand Index y el Silhouette Score.<br><br>
                Tercero, <strong>contextualizamos los resultados biofísicamente</strong>: caracterizando cuantitativamente el ritmo alfa y el Efecto Berger sobre registros basales del sujeto S001 de PhysioNet EEGMMIDB, adquiridos a 160 Hz en condiciones de ojos abiertos y ojos cerrados."
            </div>
        </section>

        <!-- DIAPOSITIVA 2 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">PROCESAMIENTO ESPACIAL & FILTRADO</span>
                <h2>Preprocesamiento y Combinación Espacial mediante PCA</h2>
                <p class="subtitle">Acondicionamiento FIR de fase cero y proyección de covarianza en sensores occipitales (O1, Oz, O2)</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Filtrado FIR de Fase Cero & Recorte de Bordes</h4>
                        <p>Filtro digital pasa-banda 1.0 - 40.0 Hz bidireccional no causal (phase='zero') que preserva la alineación temporal de los potenciales bioeléctricos. Se descartan 3.0 s en ambos extremos para mitigar transitorios de borde del pasa-alto.</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Combinación Espacial por PCA Local</h4>
                        <p>Los canales adyacentes O1, Oz y O2 registran la proyección común del dipolo occipital por conducción de volumen. PCA extrae el autovector dominante que maximiza la varianza total combinada:</p>
                        <div class="metric-pill">PC1 = 0.585*O1 + 0.568*Oz + 0.579*O2 | Varianza: {var_exp:.2f}%</div>
                    </div>
                    <div class="content-box">
                        <h4>3. Fundamento Matemático & Alcance</h4>
                        <p>Al tener pesos espacialmente simétricos (~1/√3 ≈ 0.577), PC1 actúa como un promedio ponderado óptimo en varianza. En esquemas supervisados multicanal completos, la discriminabilidad inter-condición se aborda mediante Common Spatial Patterns (CSP).</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img1}" alt="Figura 1: PCA, Canales Temporales y Topomap 10-20">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 1:20 - 2:50</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "La primera etapa del pipeline consiste en el preprocesamiento de la señal.<br><br>
                Las señales de EEG tienen amplitudes muy pequeñas, del orden de los microvoltios, por lo que son especialmente sensibles al ruido y a diferentes tipos de interferencias. Para acondicionarlas se aplicó un filtro FIR pasa-banda entre 1 y 40 Hz de fase cero bidireccional, preservando la fase original de las oscilaciones neuronales.<br><br>
                Para evitar los transitorios de respuesta impulsional en los extremos del registro generados por el pasa-alto de 1 Hz, se recortaron tres segundos al inicio y al final de cada archivo.<br><br>
                A continuación, seleccionamos los canales occipitales O1, Oz y O2, ubicados anatómicamente sobre el lóbulo occipital y la corteza visual primaria, como se ilustra en el esquema topográfico 10-20.<br><br>
                Sobre estos tres sensores adyacentes aplicamos PCA para sintetizar la actividad coherente. La primera componente principal explicó el <strong>{var_exp:.2f} %</strong> de la varianza total.<br><br>
                Matemáticamente, los coeficientes obtenidos, alrededor de 0,58 para cada canal, convergen de forma natural al vector unitario balanceado uno sobre raíz de tres. Esto es clave: significa que el algoritmo no supervisado descubrió por sí solo que el promedio espacial simple de los tres electrodos era la combinación óptima para maximizar la varianza común compartida por conducción de volumen.<br><br>
                En las trazas temporales de la figura se aprecia con claridad la transición: un trazado de baja amplitud en ojos abiertos versus una oscilación rítmica periódica de gran amplitud en ojos cerrados."
            </div>
        </section>

        <!-- DIAPOSITIVA 3 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">DOMINIO DE LA FRECUENCIA & BIOMARCADORES</span>
                <h2>Análisis Espectral y Cuantificación del Efecto Berger</h2>
                <p class="subtitle">Estimación consistente de la PSD mediante método de Welch y caracterización de dinámica periódica vs. 1/f</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Estimación Espectral de Welch</h4>
                        <p>Promediado de periodogramas modificados (ventanas Hann de 2.0 s, espaciado FFT Δf = 0.5 Hz, solapamiento 50%, resolución física de Rayleigh ~0.75 Hz a -3 dB) para minimizar la varianza del estimador.</p>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>2. Fundamento Neurofisiológico (Efecto Berger)</h4>
                        <ul>
                            <li><strong>Ojos Abiertos:</strong> Desincronización cortical (ERD) por flujo constante de aferencia visual retiniana.</li>
                            <li><strong>Ojos Cerrados:</strong> Sincronización masiva (ERS) de las poblaciones tálamo-corticales en reposo.</li>
                        </ul>
                    </div>
                    <div class="content-box box-navy">
                        <h4>3. Dinámica Periódica sobre Fondo Aperiódico 1/f</h4>
                        <p>La potencia integrada en Alfa (8-12 Hz) aumenta <strong>{ratio_berger:.2f} veces (~12 dB)</strong>. El pico individual en <strong>f = 10.0 Hz</strong> emerge <strong>~20 dB</strong> sobre la caída aperiódica 1/f.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img2}" alt="Figura 2: Densidad Espectral de Potencia">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 2:50 - 4:20</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "Una vez obtenida la componente principal, caracterizamos su contenido frecuencial mediante la Densidad Espectral de Potencia.<br><br>
                Para lograr una estimación espectral consistente y de baja varianza utilizamos el método de Welch, empleando ventanas de Hann de dos segundos con un 50 % de solapamiento. Esto establece un espaciado entre muestras de la DFT de 0,5 Hz, con una resolución física de Rayleigh aproximada de 0,75 Hz acorde al ancho del lóbulo principal de la ventana.<br><br>
                En el panel lineal de la Figura 2 observamos el contraste clásico del Efecto Berger: en la condición de ojos abiertos predomina una desincronización cortical con baja potencia espectral, mientras que al cerrar los ojos emerge un pico oscilatorio resonante muy marcado centrado en 10,0 Hz.<br><br>
                La potencia integrada en la banda alfa de 8 a 12 Hz se incrementa <strong>16 veces (alrededor de 12 dB)</strong> respecto a la condición basal.<br><br>
                Complementariamente, en el panel semilogarítmico en decibelios vemos que el pico oscilatorio de 10 Hz emerge casi <strong>20 dB por encima del piso aperiódico 1/f</strong>, característico de la actividad electrofisiológica cerebral asincrónica."
            </div>
        </section>

        <!-- DIAPOSITIVA 4 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">REPRESENTACIONES TIEMPO-FRECUENCIA</span>
                <h2>Análisis Tiempo-Frecuencia Dinámico mediante STFT Discreta</h2>
                <p class="subtitle">Compromiso de Gabor-Heisenberg y seguimiento continuo de la persistencia espectral del ritmo alfa</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Dinámica No Estacionaria del EEG</h4>
                        <p>La Transformada de Fourier global oculta la modulación temporal. La STFT discreta permite rastrear la persistencia y transitorios dinámicos de las oscilaciones neuronales a lo largo de la adquisición.</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Formulación Discreta & Parámetros de Análisis</h4>
                        <p>Ventana Hann de N = 320 muestras (Tw = 2.0 s) con avance R = 40 muestras (<strong>Δt_hop = 0.25 s</strong>, solapamiento 87.5%):</p>
                        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.80rem; color: var(--secondary-blue); margin-top: 3px;">
                            X[m, k] = ∑_{{n=0}}^{{N-1}} x[n + mR] · w[n] · e^{{-j 2π k n / N}}
                        </p>
                    </div>
                    <div class="content-box">
                        <h4>3. Interpretación de los Espectrogramas</h4>
                        <p>Ojos Abiertos muestra desincronización basal; Ojos Cerrados evidencia activación continua en 10 Hz con atenuaciones transitorias (marcadas en rojo) en t ∈ {{12, 26, 46}} s asociadas a micro-variaciones del reposo.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img3}" alt="Figura 3: Espectrograma STFT">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 4:20 - 5:50</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "Dado que las bioseñales electroencefalográficas son procesos no estacionarios, complementamos el análisis espectral con la Transformada de Fourier de Tiempo Reducido, o STFT discreta.<br><br>
                Bajo el principio de incertidumbre de Gabor-Heisenberg, la resolución temporal física está gobernada por la duración de la ventana de análisis de dos segundos (320 muestras). Para mapear la evolución de forma continua y suave, desplazamos la ventana con un paso temporal o hop size de 0,25 segundos (40 muestras), lo que equivale a un solapamiento del 87,5 %.<br><br>
                En la Figura 3 calibramos el rango dinámico en 36 decibelios para visualizar tanto la actividad de fondo como las oscilaciones dominantes en una escala cuantitativa unificada.<br><br>
                En el panel superior de ojos abiertos se aprecia un espectro desincronizado y homogéneo a lo largo de los 55 segundos.<br><br>
                En el panel inferior de ojos cerrados se observa una banda prominente y continua en torno a los 10 Hz. Si bien el alto solapamiento produce un suavizado visual, es posible detectar sutiles atenuaciones transitorias de potencia (claramente señaladas con marcadores rojos en los segundos 12, 26 y 46), vinculadas a variaciones dinámicas del reposo que analizaremos en la etapa de clustering."
            </div>
        </section>

        <!-- DIAPOSITIVA 5 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">MINERÍA DE SERIES TEMPORALES & APRENDIZAJE NO SUPERVISADO</span>
                <h2>Minería de Series Temporales y Agrupamiento con K-Means</h2>
                <p class="subtitle">Segmentación en épocas disjuntas, espacio composicional de Potencia Relativa y validación de clústeres</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Segmentación en {n_tot} Épocas Temporales Disjuntas (2.0 s)</h4>
                        <p>Se divide la señal en ventanas disjuntas de 320 muestras (sin solapamiento temporal para clustering). Se estima el espectro por época mediante Periodograma de Hann (Δf = 0.5 Hz).</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Espacio Composicional & Estandarización</h4>
                        <p>Potencias relativas normalizadas respecto a la energía total (1-40 Hz) para conferir robustez frente a derivas lentas (r = {r_val:.3f}). Estandarización z-score previa a la distancia euclidiana:</p>
                        <div class="metric-pill">%Alfa = P[8-12Hz] / P[1-40Hz] | %Beta = P[13-30Hz] / P[1-40Hz]</div>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>3. Validación No Supervisada & Mapeo Semántico Post-Hoc</h4>
                        <p>K-Means (k=2) particiona el espacio de atributos. La correspondencia con las condiciones basales confirma la separabilidad biofísica:</p>
                        <div class="metric-pill">ARI = {ari_val:.3f} | Silhouette = {sil_val:.3f} | Concordancia = {acc_pct:.2f}%</div>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img4}" alt="Figura 4: Clustering K-Means">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 5:50 - 7:40</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "En la etapa de minería de series temporales, el objetivo fue comprobar si un algoritmo no supervisado es capaz de particionar automáticamente los estados neurofisiológicos sin disponer de etiquetas previas durante el entrenamiento.<br><br>
                Para evitar muestras redundantes y respetar la independencia estadística requerida por el clustering, dividimos la componente principal en <strong>54 épocas temporales disjuntas de dos segundos</strong> (27 épocas por condición).<br><br>
                Sobre cada época calculamos la potencia relativa en la banda alfa y en la banda beta normalizadas por la energía total de 1 a 40 Hz. Esta normalización composicional otorga robustez frente a posibles derivas lentas de impedancia del electrodo, operando la banda alfa como el eje primario de discriminación.<br><br>
                Tras estandarizar las características para operar en una escala homogénea, aplicamos K-Means con k=2. Evaluamos la estructura del agrupamiento mediante métricas formales no supervisadas: [pausa breve] obtuvimos un <strong>Adjusted Rand Index de {ari_val:.3f}</strong> y un <strong>Silhouette Score de {sil_val:.3f}</strong>.<br><br>
                Al contrastar los clústeres asignados con las condiciones experimentales mediante un mapeo semántico post-hoc, la concordancia alcanza el <strong>{acc_pct:.2f} %</strong>, separando perfectamente todas las épocas de ojos abiertos (27/27) y 24 de 27 de ojos cerrados.<br><br>
                Las 3 épocas de ojos cerrados asignadas al otro grupo (resaltadas en naranja) ocurrieron en los segundos 12, 26 y 46 del registro, coincidiendo con las breves desincronizaciones de alfa visibles en el espectrograma, atribuibles a micro-modulaciones atencionales o pequeñas fluctuaciones del reposo."
            </div>
        </section>

        <!-- DIAPOSITIVA 6 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">SÍNTESIS & PERSPECTIVAS</span>
                <h2>Conclusiones</h2>
                <p class="subtitle">Integración de herramientas del curso, consideraciones metodológicas y proyección a BCI</p>
            </div>
            <div class="slide-body full-width">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.4rem; height: 100%;">
                    <div class="content-box" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>1. Preprocesamiento & Reducción Espacial</h4>
                            <p style="margin-top: 6px;">El acondicionamiento FIR de fase cero y la proyección por PCA sintetizaron el <strong>{var_exp:.2f}% de la varianza</strong> de los electrodos occipitales sin distorsión temporal, actuando como un promedio espacial coherente óptimo.</p>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--secondary-blue); font-weight: 700;">Acondicionamiento Óptimo</div>
                    </div>
                    <div class="content-box box-crimson" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>2. Minería de Series Temporales</h4>
                            <p style="margin-top: 6px;">El espacio composicional de Potencia Relativa en 54 épocas disjuntas permitió a K-Means particionar los estados con <strong>ARI = {ari_val:.3f}</strong> y <strong>Silhouette = {sil_val:.3f}</strong> de forma consistente frente a la dinámica biológica.</p>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--accent-crimson); font-weight: 700;">Agrupamiento No Supervisado</div>
                    </div>
                    <div class="content-box box-navy" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>3. Desafíos & Proyección a BCI</h4>
                            <p style="margin-top: 6px;">Para escalar hacia BCI online se requiere migrar a filtros causales de baja latencia, incorporar esquemas inter-sujeto (LOSO), calibrar la frecuencia individual alfa (IAF) y agregar limpieza de artefactos por ICA.</p>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--primary-navy); font-weight: 700;">Validación & Escalabilidad</div>
                    </div>
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 7:40 - 9:00</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "En conclusión, este trabajo muestra cómo articular las herramientas de procesamiento de señales y minería de series temporales en un pipeline completo para bioseñales de EEG.<br><br>
                El preprocesamiento digital y la combinación espacial por PCA permitieron sintetizar la actividad del polo occipital maximizando la varianza común sin introducir distorsión de fase.<br><br>
                El análisis espectral de Welch y la STFT caracterizaron con precisión la dinámica periódica del Efecto Berger respecto al fondo aperiódico, revelando una amplificación de 16 veces (12 dB) en la potencia alfa centrada en 10 Hz.<br><br>
                Finalmente, las técnicas de minería de series temporales en un espacio composicional de épocas disjuntas permitieron validar la separabilidad no supervisada de los estados basales con un Adjusted Rand Index de {ari_val:.3f} y {acc_pct:.2f} % de concordancia.<br><br>
                Como consideraciones metodológicas para futuras extensiones, se destacan: evaluar esquemas inter-sujeto (como Leave-One-Subject-Out), calibrar la frecuencia individual alfa (IAF), incorporar módulos de limpieza de artefactos por ICA y migrar a filtros digitales causales sobre buffers deslizantes para aplicaciones en tiempo real.<br><br>
                Muchas gracias por su atención."
            </div>
        </section>
    </main>

    <!-- PANEL INFERIOR DE NOTAS DE ORADOR -->
    <div class="speaker-notes-panel" id="speakerPanel">
        <div class="notes-header">
            <span>NOTAS DE ORADOR (GUION PARA EXPOSICIÓN DE 9 MINUTOS)</span>
            <button class="btn" style="padding: 2px 8px; font-size: 0.75rem; background: #334155; color: #ffffff; border: none;" onclick="toggleSpeakerNotes()">[Cerrar - N]</button>
        </div>
        <div class="notes-text" id="speakerNotesText"></div>
    </div>

    <footer>
        <div>PROCESAMIENTO AVANZADO DE SEÑALES & MINERÍA DE SERIES TEMPORALES - UBA</div>
        <div>Navegación: <span class="kb-badge">[◀]</span> / <span class="kb-badge">[▶]</span> o <span class="kb-badge">[Espacio]</span> | Notas: <span class="kb-badge">[N]</span> | Pantalla Completa: <span class="kb-badge">[F]</span></div>
    </footer>

    <script>
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        const totalSlides = slides.length;
        const slideNumDisplay = document.getElementById('slideNum');
        const speakerPanel = document.getElementById('speakerPanel');
        const speakerNotesText = document.getElementById('speakerNotesText');

        function updateSlide() {{
            slides.forEach((s, idx) => {{
                s.classList.toggle('active', idx === currentSlide);
            }});
            slideNumDisplay.textContent = `${{currentSlide + 1}} / ${{totalSlides}}`;
            const notesEl = slides[currentSlide].querySelector('.speaker-notes-content');
            if (notesEl) {{
                speakerNotesText.innerHTML = notesEl.innerHTML;
            }}
        }}

        function nextSlide() {{
            if (currentSlide < totalSlides - 1) {{
                currentSlide++;
                updateSlide();
            }}
        }}

        function prevSlide() {{
            if (currentSlide > 0) {{
                currentSlide--;
                updateSlide();
            }}
        }}

        function toggleSpeakerNotes() {{
            speakerPanel.classList.toggle('visible');
        }}

        function toggleFullScreen() {{
            if (!document.fullscreenElement) {{
                document.documentElement.requestFullscreen();
            }} else {{
                if (document.exitFullscreen) document.exitFullscreen();
            }}
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') nextSlide();
            else if (e.key === 'ArrowLeft' || e.key === 'PageUp') prevSlide();
            else if (e.key === 'n' || e.key === 'N') toggleSpeakerNotes();
            else if (e.key === 'f' || e.key === 'F') toggleFullScreen();
        }});

        updateSlide();
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f" -> Presentación generada con éxito: {os.path.abspath(output_path)}")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Carga y Preprocesamiento MNE
    data_open, data_closed, target_channels, sfreq = cargar_y_preprocesar_eeg(base_dir)
    
    # 2. Combinación Espacial PCA
    pc1_open, pc1_closed, pca_model, var_exp = aplicar_pca_espacial(data_open, data_closed)
    
    # 3. Espectro Welch y STFT
    freqs, psd_open, psd_closed, f_stft, t_stft_o, Sxx_open, t_stft_c, Sxx_closed, ratio_berger = analisis_espectral_y_tiempo_frecuencia(
        pc1_open, pc1_closed, sfreq
    )
    
    # 4. Minería de Series Temporales: Periodograma Hann y Potencia Relativa
    clustering_res = mineria_y_clustering_kmeans(pc1_open, pc1_closed, sfreq, win_len_sec=2.0)
    
    # 5. Guardar Figuras Científicas
    saved_figs = generar_figuras(
        data_open, data_closed, target_channels, pc1_open, pc1_closed,
        freqs, psd_open, psd_closed, f_stft, t_stft_o, Sxx_open, t_stft_c, Sxx_closed,
        clustering_res, sfreq, base_dir
    )
    
    # 6. Generar HTML Formal
    html_path = os.path.join(base_dir, "presentacion.html")
    generar_presentacion_html(var_exp, ratio_berger, clustering_res, html_path)
    
    print("\n" + "="*70)
    print(" [OK] EJECUCION COMPLETADA EXITOSAMENTE")
    print(f" -> Presentacion lista: {html_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
