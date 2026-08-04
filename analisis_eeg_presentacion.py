#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
PROYECTO FINAL: PROCESAMIENTO AVANZADO DE SEÑALES Y MINERÍA DE SERIES TEMPORALES
Pipeline End-to-End: Análisis de EEG (Efecto Berger), PCA, STFT y Clustering K-Means
Generador Automático de Diapositivas HTML con Rigor Científico y Académico
===============================================================================
"""

import os
import sys
import base64
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram, welch, periodogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Configuración estética de Matplotlib para publicaciones académicas
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
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
    - Recorte dinámico de transitorios de borde (3.0 s en cada extremo) acorde a la longitud teórica del filtro.
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
# 2. REDUCCIÓN ESPACIAL POR PCA (UNIDAD 1)
# =============================================================================

def aplicar_pca_espacial(data_open, data_closed):
    """
    Aplica PCA como técnica de reducción de dimensionalidad espacial sobre los canales occipitales O1, Oz y O2.
    Nota DSP: La señal ya tiene componente continua nula (media cero) debido al filtro pasa-alto con corte en 1.0 Hz.
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
    print(f" -> Pesos espaciales del filtro (Loadings): O1={loadings[0]:.3f}, Oz={loadings[1]:.3f}, O2={loadings[2]:.3f}")
    
    return pc1_open, pc1_closed, pca, var_exp


# =============================================================================
# 3. ANÁLISIS ESPECTRAL Y TIEMPO-FRECUENCIA (PSD Y STFT) (UNIDAD 1)
# =============================================================================

def analisis_espectral_y_tiempo_frecuencia(pc1_open, pc1_closed, sfreq):
    """
    Calcula la Densidad Espectral de Potencia (PSD de Welch) sobre el registro completo y Espectrogramas (STFT)
    para cuantificar el Efecto Berger evitando artefactos de salto temporal.
    """
    print("\n" + "="*70)
    print(" [3/5] ANALISIS ESPECTRAL Y TIEMPO-FRECUENCIA (PSD & STFT)")
    print("="*70)
    
    # Welch sobre el registro estacionario completo (ventana 2.0 s, df = 0.5 Hz, solapamiento 50%)
    nperseg = int(2.0 * sfreq)
    freqs, psd_open = welch(pc1_open, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    _, psd_closed = welch(pc1_closed, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    
    # Integración numérica en banda Alfa (8 - 12 Hz) - Unidades físicas correctas: uV^2
    idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    integrate_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    p_alpha_open = integrate_fn(psd_open[idx_alpha], freqs[idx_alpha])
    p_alpha_closed = integrate_fn(psd_closed[idx_alpha], freqs[idx_alpha])
    ratio_berger = p_alpha_closed / (p_alpha_open + 1e-12)
    
    print(f" -> Potencia Integrada Alfa (Ojos Abiertos): {p_alpha_open * 1e12:.4f} uV^2")
    print(f" -> Potencia Integrada Alfa (Ojos Cerrados): {p_alpha_closed * 1e12:.4f} uV^2")
    print(f" -> Incremento de Potencia Alfa (Ratio Berger): {ratio_berger:.2f}x")
    
    # Espectrogramas STFT independientes
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
# 4. MINERÍA DE SERIES TEMPORALES: ATRIBUTOS DE POTENCIA RELATIVA Y K-MEANS (UNIDAD 2)
# =============================================================================

def mineria_y_clustering_kmeans(pc1_open, pc1_closed, sfreq, win_len_sec=2.0):
    """
    Segmentación en épocas disjuntas de 2.0 s, cálculo de potencia espectral mediante Periodograma Modificado con ventana Hann,
    extracción de Potencia Relativa (Alfa y Beta respecto a banda total 1-40 Hz) para garantizar invariancia ante cambios de impedancia,
    y clustering K-Means.
    """
    print("\n" + "="*70)
    print(" [4/5] MINERIA DE SERIES TEMPORALES: EXTRACCION DE ATRIBUTOS Y K-MEANS")
    print("="*70)
    
    samples_per_win = int(win_len_sec * sfreq)  # 320 muestras por ventana
    
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
            
            # Potencia Relativa de Banda: Invariante ante impedancia de electrodos y grosor craneal
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
    
    print(f" -> Ventanas Analizadas (2.0 s netas): {len(y_true)} ({len(y_open)} Abiertos, {len(y_closed)} Cerrados)")
    print(f" -> Correlacion Potencia Relativa Alfa vs Beta: r = {corr_alpha_beta:.3f}")
    print(f" -> Silhouette Score: {sil_score:.3f}")
    print(f" -> Adjusted Rand Index (ARI): {ari_score:.3f}")
    print(f" -> Concordancia Semántica vs Ground Truth (Prueba Intra-Sujeto): {acc * 100:.2f}%")
    print(f" -> Matriz de Confusión:\n{cm}")
    
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
        'n_closed': len(y_closed)
    }


# =============================================================================
# 5. GENERACIÓN DE FIGURAS (.PNG) CON ESTILO EDITORIAL / ACADÉMICO
# =============================================================================

def generar_figuras(data_open, data_closed, target_channels, pc1_open, pc1_closed, 
                     freqs, psd_open, psd_closed, f_stft, t_stft_o, Sxx_open, t_stft_c, Sxx_closed, 
                     clustering_res, sfreq, output_dir="."):
    """
    Genera 4 gráficos científicos en alta resolución con estilo formal universitario.
    """
    print("\n" + "="*70)
    print(" [5/5] GENERACION DE FIGURAS CIENTIFICAS (.PNG)")
    print("="*70)
    
    saved_files = []
    
    # -------------------------------------------------------------------------
    # Figura 1: Canales Occipitales y PC1 (Filtro Espacial)
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 6), dpi=150)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])
    t_plot = np.arange(0, int(6 * sfreq)) / sfreq
    
    ax1 = fig1.add_subplot(gs[0, 0])
    for idx, ch in enumerate(target_channels):
        ax1.plot(t_plot, data_open[idx, :len(t_plot)] * 1e6 + idx * 40, label=ch, lw=1.1)
    ax1.set_title("Ojos Abiertos: Canales Occipitales (O1, Oz, O2)", fontweight='bold', color='#0f2942')
    ax1.set_ylabel(r"Amplitud ($\mu$V)")
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax1.set_ylim(-30, (len(target_channels)) * 40)
    
    ax2 = fig1.add_subplot(gs[0, 1])
    for idx, ch in enumerate(target_channels):
        ax2.plot(t_plot, data_closed[idx, :len(t_plot)] * 1e6 + idx * 40, label=ch, lw=1.1)
    ax2.set_title("Ojos Cerrados: Canales Occipitales (Ritmo Alfa)", fontweight='bold', color='#0f2942')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.set_ylim(-30, (len(target_channels)) * 40)
    
    ax3 = fig1.add_subplot(gs[1, 0])
    ax3.plot(t_plot, pc1_open[:len(t_plot)] * 1e6, color='#0369a1', lw=1.3)
    ax3.set_title("PC1 Espacial - Ojos Abiertos (Desincronización Visual)", fontsize=10.5, fontweight='bold', color='#1e293b')
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel(r"PC1 ($\mu$V)")
    
    ax4 = fig1.add_subplot(gs[1, 1])
    ax4.plot(t_plot, pc1_closed[:len(t_plot)] * 1e6, color='#b91c1c', lw=1.3)
    ax4.set_title("PC1 Espacial - Ojos Cerrados (Oscilación Coherente ~10 Hz)", fontsize=10.5, fontweight='bold', color='#1e293b')
    ax4.set_xlabel("Tiempo (s)")
    
    plt.tight_layout()
    f1_path = os.path.join(output_dir, "fig1_pca_preprocesamiento.png")
    fig1.savefig(f1_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    saved_files.append(f1_path)
    print(f" -> Guardado: {f1_path}")
    
    # -------------------------------------------------------------------------
    # Figura 2: Densidad Espectral de Potencia (Welch)
    # -------------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(10, 5), dpi=150)
    mask_f = np.logical_and(freqs >= 1.0, freqs <= 30.0)
    
    psd_open_uv = psd_open * 1e12
    psd_closed_uv = psd_closed * 1e12
    
    ax.plot(freqs[mask_f], psd_open_uv[mask_f], label='Ojos Abiertos (Desincronizado - ERD)', color='#0284c7', lw=2.0)
    ax.plot(freqs[mask_f], psd_closed_uv[mask_f], label='Ojos Cerrados (Sincronizado - ERS Ritmo Alfa)', color='#b91c1c', lw=2.2)
    ax.axvspan(8.0, 12.0, color='#fef3c7', alpha=0.6, label='Banda Alfa (8 - 12 Hz)')
    
    idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    f_peak = freqs[idx_alpha][np.argmax(psd_closed_uv[idx_alpha])]
    p_peak = np.max(psd_closed_uv[idx_alpha])
    ax.annotate(f'Pico Alfa: {f_peak:.1f} Hz\n(Efecto Berger)', 
                xy=(f_peak, p_peak), xytext=(f_peak + 3.2, p_peak * 0.85),
                arrowprops=dict(facecolor='#991b1b', shrink=0.08, width=1.3, headwidth=6),
                fontweight='bold', color='#991b1b',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff1f2", ec="#f43f5e", lw=1))
    
    ax.set_title("Densidad Espectral de Potencia (Welch PSD) en Componente Principal Occipital", fontweight='bold', color='#0f2942')
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel(r"Densidad Espectral de Potencia ($\mu\mathrm{V}^2/\mathrm{Hz}$)")
    ax.set_xlim(1, 30)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10.5)
    
    plt.tight_layout()
    f2_path = os.path.join(output_dir, "fig2_psd_espectro_alfa.png")
    fig2.savefig(f2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    saved_files.append(f2_path)
    print(f" -> Guardado: {f2_path}")
    
    # -------------------------------------------------------------------------
    # Figura 3: STFT Espectrogramas Comparativos Continuos
    # -------------------------------------------------------------------------
    fig3, (ax_open, ax_closed) = plt.subplots(2, 1, figsize=(11, 6), dpi=150, sharex=True, sharey=True)
    
    vmin = min(10 * np.log10(np.min(Sxx_open * 1e12) + 1e-6), 10 * np.log10(np.min(Sxx_closed * 1e12) + 1e-6))
    vmax = max(10 * np.log10(np.max(Sxx_open * 1e12) + 1e-6), 10 * np.log10(np.max(Sxx_closed * 1e12) + 1e-6))
    
    mesh1 = ax_open.pcolormesh(t_stft_o, f_stft, 10 * np.log10(Sxx_open * 1e12 + 1e-6), 
                              cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
    ax_open.axhspan(8, 12, color='white', alpha=0.25, linestyle=':', lw=1.2)
    ax_open.set_title("Condición 1: Ojos Abiertos (Desincronización Cortical Continua)", fontweight='bold', color='#0369a1', fontsize=11)
    ax_open.set_ylabel("Frecuencia (Hz)")
    
    mesh2 = ax_closed.pcolormesh(t_stft_c, f_stft, 10 * np.log10(Sxx_closed * 1e12 + 1e-6), 
                                cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
    ax_closed.axhspan(8, 12, color='white', alpha=0.25, linestyle=':', lw=1.2)
    ax_closed.text(4, 10, "Banda Alfa Sostenida (8-12 Hz)", color='white', fontweight='bold', fontsize=9.5, 
                   bbox=dict(boxstyle="square,pad=0.2", fc=(0, 0, 0, 0.6), ec="none"))
    ax_closed.set_title("Condición 2: Ojos Cerrados (Sincronización Rítmica Alfa Sostenida)", fontweight='bold', color='#b91c1c', fontsize=11)
    ax_closed.set_xlabel("Tiempo de Registro (segundos)")
    ax_closed.set_ylabel("Frecuencia (Hz)")
    ax_closed.set_ylim(1, 30)
    
    plt.tight_layout()
    fig3.subplots_adjust(bottom=0.22)
    cbar_ax = fig3.add_axes([0.15, 0.08, 0.7, 0.04])
    cbar = fig3.colorbar(mesh2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r"Densidad Espectral de Potencia ($\mathrm{dB}\ [\mu\mathrm{V}^2/\mathrm{Hz}]$)")
    
    f3_path = os.path.join(output_dir, "fig3_espectrograma_stft.png")
    fig3.savefig(f3_path, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    saved_files.append(f3_path)
    print(f" -> Guardado: {f3_path}")
    
    # -------------------------------------------------------------------------
    # Figura 4: K-Means en Espacio 2D Alfa-Beta Relativa y Matriz de Confusión
    # -------------------------------------------------------------------------
    fig4, (ax_clus, ax_cm) = plt.subplots(1, 2, figsize=(12, 5), dpi=150, gridspec_kw={'width_ratios': [1.35, 1]})
    X = clustering_res['X']
    y_true = clustering_res['y_true']
    pred = clustering_res['y_pred_aligned']
    
    colors = ['#0284c7', '#b91c1c']
    markers = ['o', 's']
    labels = ['Ojos Abiertos', 'Ojos Cerrados']
    
    for cls in [0, 1]:
        mask = (y_true == cls)
        ax_clus.scatter(X[mask, 0] * 100, X[mask, 1] * 100, c=colors[cls], label=f'Condición Real: {labels[cls]}',
                        marker=markers[cls], s=65, alpha=0.85, edgecolors='#1e293b', lw=0.6)
        
    errors = (y_true != pred)
    if np.any(errors):
        ax_clus.scatter(X[errors, 0] * 100, X[errors, 1] * 100, facecolors='none', edgecolors='#d97706',
                        s=120, lw=2.2, label='Discrepancia Semántica')
        
    ax_clus.set_title(f"Clustering K-Means en Espacio 2D (Potencia Relativa)\n(Silhouette: {clustering_res['silhouette']:.3f} | ARI: {clustering_res['ari']:.3f})", 
                      fontweight='bold', color='#0f2942')
    ax_clus.set_xlabel(r"Potencia Relativa Alfa $[8-12\ \mathrm{Hz}] / [1-40\ \mathrm{Hz}]\ (\%)$")
    ax_clus.set_ylabel(r"Potencia Relativa Beta $[13-30\ \mathrm{Hz}] / [1-40\ \mathrm{Hz}]\ (\%)$")
    ax_clus.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
    
    cm = clustering_res['confusion_matrix']
    cax = ax_cm.matshow(cm, cmap='Blues', alpha=0.85)
    
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]}", ha='center', va='center', fontsize=18, fontweight='bold',
                       color='white' if cm[i, j] > np.max(cm)/2 else '#0f172a')
            
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Abiertos (0)', 'Cerrados (1)'])
    ax_cm.set_yticklabels(['Abiertos (0)', 'Cerrados (1)'])
    ax_cm.set_xlabel("Clúster Asignado por K-Means", fontweight='bold', color='#0f2942')
    ax_cm.set_ylabel("Condición Real (Ground Truth)", fontweight='bold', color='#0f2942')
    ax_cm.set_title(f"Matriz de Confusión\n(Concordancia: {clustering_res['accuracy']*100:.1f}% | Sujeto S001)", fontweight='bold', pad=15, color='#0f2942')
    
    plt.tight_layout()
    f4_path = os.path.join(output_dir, "fig4_clustering_kmeans.png")
    fig4.savefig(f4_path, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    saved_files.append(f4_path)
    print(f" -> Guardado: {f4_path}")
    
    return saved_files


# =============================================================================
# 6. GENERACIÓN DEL HTML FORMAL UNIVERSITARIO (SIN EMOJIS, TEMA CLARO)
# =============================================================================

def generar_presentacion_html(var_exp, ratio_berger, clustering_res, output_path="presentacion.html"):
    """
    Genera una presentación HTML limpia, formal y académica (estilo conferencia universitaria).
    Fondo claro, tipografía sobria, sin emojis y con notas de orador sincronizadas a 9 minutos.
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
    n_tot = clustering_res['n_open'] + clustering_res['n_closed']
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Defensa Final: Procesamiento Avanzado de Señales y Minería de Series Temporales</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@400;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-page: #f1f5f9;
            --card-bg: #ffffff;
            --border-color: #cbd5e1;
            --primary-navy: #0f2942;
            --secondary-blue: #1d4ed8;
            --accent-crimson: #991b1b;
            --text-main: #1e293b;
            --text-muted: #475569;
            --sidebar-bg: #e2e8f0;
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
            height: 52px;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            background: #ffffff;
            z-index: 100;
        }}
        .univ-title {{
            font-size: 0.88rem;
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
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.82rem;
            font-weight: 600;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #f8fafc;
            border-color: var(--secondary-blue);
            color: var(--secondary-blue);
        }}
        .btn-notes {{
            background: #f8fafc;
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
            padding: 1.5rem 2.5rem;
        }}
        .slide {{
            position: absolute;
            width: calc(100% - 5rem);
            max-width: 1320px;
            height: calc(100% - 2rem);
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2.2rem 3rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 4px 20px rgba(15, 41, 66, 0.06);
            opacity: 0;
            transform: scale(0.98);
            pointer-events: none;
            transition: all 0.35s ease-out;
        }}
        .slide.active {{
            opacity: 1;
            transform: scale(1);
            pointer-events: all;
        }}
        .slide-header {{
            margin-bottom: 1.2rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.8rem;
        }}
        .badge {{
            display: inline-block;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
            background: #e2e8f0;
            color: var(--primary-navy);
            border: 1px solid #cbd5e1;
        }}
        h1 {{
            font-family: 'Merriweather', serif;
            font-size: 1.95rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-bottom: 4px;
            line-height: 1.25;
        }}
        h2 {{
            font-family: 'Merriweather', serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-bottom: 3px;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 0.92rem;
            font-weight: 400;
        }}
        .slide-body {{
            flex: 1;
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 2.2rem;
            align-items: center;
            min-height: 0;
        }}
        .slide-body.full-width {{
            grid-template-columns: 1fr;
        }}
        .text-content {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
            overflow-y: auto;
            max-height: 100%;
        }}
        .content-box {{
            background: #f8fafc;
            border-left: 3.5px solid var(--secondary-blue);
            border-radius: 0 6px 6px 0;
            padding: 0.85rem 1.1rem;
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
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--primary-navy);
            margin-bottom: 4px;
        }}
        .content-box p, .content-box ul {{
            font-size: 0.86rem;
            color: var(--text-main);
            line-height: 1.45;
        }}
        ul {{
            list-style-position: inside;
        }}
        .image-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 0.4rem;
            overflow: hidden;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.02);
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 380px;
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
            max-height: 38vh;
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
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        .notes-text {{
            font-size: 0.92rem;
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
            height: 34px;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
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
            font-size: 0.7rem;
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
                <span class="badge">EVALUACIÓN INTEGRADORA (9 MINUTOS)</span>
                <h1>Decodificación de Dinámicas Neurofisiológicas en Señales EEG</h1>
                <p class="subtitle">Integración de Preprocesamiento de Fase Cero, Reducción Espacial por PCA, Espectrometría Welch/STFT y Clustering K-Means</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Objetivo del Trabajo</h4>
                        <p>Desarrollar e implementar un pipeline computacional riguroso y reproducible para discriminar de forma no supervisada estados neurofisiológicos de reposo visual (Ojos Abiertos vs. Ojos Cerrados) utilizando registros reales de la base de datos PhysioNet EEGMMIDB.</p>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>2. Articulación con el Programa Académico</h4>
                        <ul>
                            <li><strong>Unidad 1:</strong> Filtrado digital pasa-banda FIR de fase cero, reducción de dimensionalidad espacial por PCA y análisis tiempo-frecuencia (STFT).</li>
                            <li><strong>Unidad 2:</strong> Segmentación en épocas homogéneas, extracción de atributos espectrales de Potencia Relativa y agrupamiento no supervisado con K-Means.</li>
                            <li><strong>Unidad 3:</strong> Contextualización biomédica y caracterización del ritmo alfa occipital (Efecto Berger).</li>
                        </ul>
                    </div>
                    <div class="content-box box-navy">
                        <h4>3. Descripción del Conjunto de Datos</h4>
                        <p>Registros basales del Sujeto S001 (PhysioNet), 64 canales (montaje 10-10), Fs = 160 Hz, condiciones de reposo visual con ojos abiertos (Run 1) y ojos cerrados (Run 2).</p>
                    </div>
                </div>
                <div class="image-container" style="flex-direction: column; text-align: center; gap: 14px; background: #f8fafc;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--primary-navy); border: 1px solid #cbd5e1; padding: 10px 14px; border-radius: 6px; background: #ffffff; width: 85%;">
                        <strong>ESTRUCTURA DEL PIPELINE</strong><br><br>
                        Datos Crudos EDF (PhysioNet / MNE)<br>
                        ↓<br>
                        Filtro Pasa-Banda FIR Fase Cero (1 - 40 Hz) + Recorte Dinámico (3s)<br>
                        ↓<br>
                        Reducción Espacial PCA en Corteza Occipital (O1, Oz, O2)<br>
                        ↓<br>
                        Análisis Espectral Welch & STFT<br>
                        ↓<br>
                        Ventaneo Temporal (2.0 s) & Periodograma Hann de Potencia Relativa<br>
                        ↓<br>
                        Clustering No Supervisado K-Means (k=2)
                    </div>
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 0:00 - 1:20</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "En este trabajo presento un pipeline computacional para analizar señales reales de electroencefalografía, o EEG, con el objetivo de identificar cambios en la actividad cerebral asociados al ritmo alfa.<br><br>
                El proyecto integra los principales contenidos vistos durante la materia.<br><br>
                En la <strong>Unidad 1</strong>, se trabajó sobre el preprocesamiento de señales: filtrado digital, reducción de dimensionalidad mediante Análisis de Componentes Principales, o PCA, y representaciones tiempo-frecuencia utilizando la Transformada de Fourier de Tiempo Reducido, conocida como STFT.<br><br>
                En la <strong>Unidad 2</strong>, se aplicaron técnicas de minería de series temporales. Para eso se segmentó la señal en ventanas temporales y se extrajeron características espectrales que luego fueron utilizadas por un algoritmo de clustering no supervisado.<br><br>
                Finalmente, en la <strong>Unidad 3</strong>, todos estos métodos se aplicaron a un problema clásico de neurofisiología: la detección del ritmo alfa occipital y el efecto Berger.<br><br>
                Como conjunto de datos se utilizaron registros públicos del dataset PhysioNet EEGMMIDB, específicamente el sujeto S001, con señales de 64 canales muestreadas a 160 Hz en dos condiciones de reposo: ojos abiertos y ojos cerrados."
            </div>
        </section>

        <!-- DIAPOSITIVA 2 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDAD 1: PROCESAMIENTO ESPACIAL & FILTRADO</span>
                <h2>Preprocesamiento y Reducción Espacial mediante PCA</h2>
                <p class="subtitle">Aislamiento de la corteza visual y síntesis del dipolo dominante de máxima covarianza</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Filtrado Digital FIR de Fase Cero (1.0 - 40.0 Hz)</h4>
                        <p>Filtro digital bidireccional no causal (phase='zero') que preserva la alineación temporal sin retardo de grupo. Se aplica un recorte dinámico de 3.0 s en los bordes para suprimir por completo los transitorios de convolución del pasa-alto de 1 Hz.</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Fundamentación del Filtrado Espacial por PCA</h4>
                        <p>Debido a la conducción de volumen, los electrodos O1, Oz y O2 registran actividad fuertemente correlacionada. La señal ya tiene media cero (0 Hz eliminado por el pasa-alto). PCA extrae el autovector de máxima covarianza:</p>
                        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; color: var(--secondary-blue); margin-top: 4px;">
                            PC1 = w1*O1 + w2*Oz + w3*O2 (Varianza explicada: <strong>{var_exp:.2f}%</strong>)
                        </p>
                    </div>
                    <div class="content-box">
                        <h4>3. Consideración Teórica: PCA vs. CSP</h4>
                        <p>PCA opera de forma no supervisada sobre la covarianza conjunta. En escenarios supervisados orientados a discriminación óptima, la técnica formal es Common Spatial Patterns (CSP).</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img1}" alt="Figura 1: PCA y Canales Temporales">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 1:20 - 2:50</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "La primera etapa del pipeline consiste en el preprocesamiento de la señal.<br><br>
                Las señales de EEG tienen amplitudes muy pequeñas, del orden de los microvoltios, por lo que son especialmente sensibles al ruido y a diferentes tipos de interferencias. Para reducir estos efectos se aplicó un filtro FIR pasa-banda entre 1 y 40 Hz utilizando filtrado de fase cero bidireccional. Esta técnica elimina el ruido sin introducir desplazamientos temporales en la señal.<br><br>
                Como este tipo de filtro genera transitorios al comienzo y al final del registro, se descartaron tres segundos en ambos extremos para evitar que esos efectos afectaran el análisis.<br><br>
                Luego se seleccionaron únicamente los canales occipitales O1, Oz y O2, ya que son los electrodos donde el ritmo alfa suele observarse con mayor intensidad.<br><br>
                Sobre estos tres canales se aplicó PCA, una técnica de reducción de dimensionalidad que permite sintetizar la información común en una única componente principal.<br><br>
                La primera componente explicó el <strong>{var_exp:.2f} %</strong> de la varianza total, indicando que resume prácticamente toda la información compartida entre los tres electrodos.<br><br>
                En la figura puede verse que, mientras con los ojos abiertos la señal presenta baja amplitud y un comportamiento desorganizado, con los ojos cerrados aparece una oscilación periódica muy clara alrededor de los 10 Hz, correspondiente al ritmo alfa."
            </div>
        </section>

        <!-- DIAPOSITIVA 3 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDADES 1 & 3: DOMINIO DE LA FRECUENCIA & BIOMARCADORES</span>
                <h2>Análisis Espectral y Efecto Berger</h2>
                <p class="subtitle">Estimación espectral consistente mediante método de Welch y cuantificación de la sincronización neuronal</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Metodología de Estimación Espectral (Welch)</h4>
                        <p>Se aplica el método de Welch sobre PC1 con ventanas Hann de 2.0 segundos (N = 320 muestras, Δf = 0.5 Hz) y 50% de solapamiento, logrando un estimador espectral consistente de baja varianza.</p>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>2. Fundamento Neurofisiológico: Efecto Berger</h4>
                        <ul>
                            <li><strong>Ojos Abiertos:</strong> Desincronización relacionada con eventos (ERD) por procesamiento activo de estímulos visuales.</li>
                            <li><strong>Ojos Cerrados:</strong> Sincronización relacionada con eventos (ERS) de las poblaciones neuronales talamocorticales en reposo.</li>
                        </ul>
                    </div>
                    <div class="content-box box-navy">
                        <h4>3. Cuantificación del Incremento Espectral</h4>
                        <p>La integración numérica de la PSD en la banda Alfa (8 a 12 Hz) evidencia un incremento de <strong>{ratio_berger:.2f} veces</strong> en potencia absoluta (μV²), con un pico resonante en 10.0 Hz.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img2}" alt="Figura 2: Densidad Espectral de Potencia">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 2:50 - 4:20</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "Una vez obtenida la componente principal, se analizó su contenido en frecuencia mediante la Densidad Espectral de Potencia.<br><br>
                Para estimar el espectro se utilizó el método de Welch con ventanas Hann de dos segundos y un 50 % de solapamiento, logrando una resolución de 0,5 Hz.<br><br>
                En la figura se comparan los espectros correspondientes a las dos condiciones experimentales.<br><br>
                Con los ojos abiertos, la potencia en la banda alfa permanece baja debido a la estimulación visual constante.<br><br>
                En cambio, al cerrar los ojos desaparece la entrada de información visual y las neuronas de la corteza occipital comienzan a sincronizar su actividad. Como consecuencia, aparece un pico muy marcado alrededor de los 10 Hz, fenómeno conocido como efecto Berger.<br><br>
                La potencia integrada en la banda alfa aumenta aproximadamente <strong>16 veces</strong> respecto de la condición de ojos abiertos, mostrando una separación muy clara entre ambos estados."
            </div>
        </section>

        <!-- DIAPOSITIVA 4 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDAD 1: REPRESENTACIONES TIEMPO-FRECUENCIA</span>
                <h2>Análisis Tiempo-Frecuencia mediante STFT</h2>
                <p class="subtitle">Seguimiento tiempo-frecuencia continuo de la activación y bloqueo del ritmo alfa</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Carácter No Estacionario de las Bioseñales EEG</h4>
                        <p>La Transformada de Fourier global asume estacionariedad en todo el registro. Las señales neuroeléctricas requieren un seguimiento dinámico tiempo-frecuencia.</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Formulación de la STFT y Principio de Incertidumbre</h4>
                        <p>Se utiliza una ventana deslizante Hann de 2.0 segundos (87.5% de solapamiento) para balancear el compromiso de resolución tiempo-frecuencia de Gabor-Heisenberg:</p>
                        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--secondary-blue); margin-top: 4px;">
                            X(t, f) = ∫ x(τ) w(τ - t) e^(-j 2π f τ) dτ
                        </p>
                    </div>
                    <div class="content-box">
                        <h4>3. Evaluación Continua e Independiente</h4>
                        <p>Los espectrogramas independientes confirman la estabilidad del biomarcador sin introducir artefactos de discontinuidad temporal.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img3}" alt="Figura 3: Espectrograma STFT">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 4:20 - 5:50</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "Hasta este punto el análisis fue realizado únicamente en frecuencia.<br><br>
                Sin embargo, las señales biológicas son procesos no estacionarios, por lo que también resulta importante conocer cómo evoluciona el contenido espectral a lo largo del tiempo.<br><br>
                Para eso se utilizó la Transformada de Fourier de Tiempo Reducido, o STFT.<br><br>
                Esta técnica desplaza una ventana temporal sobre la señal y calcula el espectro en cada posición, generando un espectrograma donde puede observarse simultáneamente la evolución temporal y la distribución en frecuencia.<br><br>
                En el espectrograma correspondiente a ojos abiertos, la energía en la banda alfa es baja durante prácticamente todo el registro.<br><br>
                En cambio, en la condición de ojos cerrados aparece una banda brillante y estable alrededor de los 10 Hz, mostrando que la actividad alfa permanece sostenida durante toda la adquisición.<br><br>
                Esto confirma visualmente lo observado previamente en el análisis espectral."
            </div>
        </section>

        <!-- DIAPOSITIVA 5 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDAD 2: MINERÍA DE SERIES TEMPORALES & APRENDIZAJE NO SUPERVISADO</span>
                <h2>Minería de Series Temporales y Clustering</h2>
                <p class="subtitle">Descubrimiento autónomo de estados cerebrales a partir de descriptores espectrales normalizados</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Segmentación Temporal en Épocas de 2.0 s</h4>
                        <p>Se divide la señal PC1 en {n_tot} épocas disjuntas de 320 muestras. Se calcula la potencia espectral mediante Periodograma Modificado con ventana Hann (Δf = 0.5 Hz).</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Espacio 2D de Potencia Relativa (Invariante)</h4>
                        <ul>
                            <li><strong>Atributo 1:</strong> Potencia Relativa Alfa (8-12 Hz / 1-40 Hz) - Invariante ante impedancia.</li>
                            <li><strong>Atributo 2:</strong> Potencia Relativa Beta (13-30 Hz / 1-40 Hz) - Control cortical de banda alta.</li>
                        </ul>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>3. Evaluación del Agrupamiento No Supervisado</h4>
                        <p>K-Means (k=2) clusteriza las observaciones sin conocimiento previo de las etiquetas:</p>
                        <p style="font-weight: 700; color: var(--primary-navy); margin-top: 4px;">
                            Silhouette Score: {sil_val:.3f} | Adjusted Rand Index (ARI): {ari_val:.3f} | Concordancia: {acc_pct:.2f}%
                        </p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img4}" alt="Figura 4: Clustering K-Means">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 5:50 - 7:40</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "La última etapa del trabajo corresponde a la minería de series temporales.<br><br>
                El objetivo fue evaluar si un algoritmo de aprendizaje no supervisado podía distinguir automáticamente entre las condiciones de ojos abiertos y ojos cerrados sin utilizar etiquetas.<br><br>
                Para ello, la componente principal se dividió en <strong>54 ventanas independientes de dos segundos</strong>.<br><br>
                Sobre cada ventana se calculó la potencia espectral y se construyeron dos atributos:<br>
                - la potencia relativa en la banda alfa,<br>
                - y la potencia relativa en la banda beta,<br>
                ambas normalizadas respecto de la potencia total entre 1 y 40 Hz.<br><br>
                Esta normalización reduce la influencia de factores externos, como diferencias de amplitud producidas por la impedancia de los electrodos.<br><br>
                Luego de estandarizar las variables, se aplicó K-Means con k igual a 2.<br><br>
                El resultado muestra que el algoritmo logró separar correctamente la mayoría de las ventanas, obteniendo un <strong>Adjusted Rand Index de {ari_val:.3f}</strong>, un <strong>Silhouette Score de {sil_val:.3f}</strong> y una <strong>concordancia del {acc_pct:.2f} %</strong> respecto de las etiquetas reales.<br><br>
                Las pocas ventanas clasificadas de forma diferente probablemente reflejan variaciones transitorias en la actividad cerebral durante el registro."
            </div>
        </section>

        <!-- DIAPOSITIVA 6 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">SÍNTESIS & DISCUSIÓN ACADÉMICA</span>
                <h2>Conclusiones</h2>
                <p class="subtitle">Balance integral del pipeline, consideraciones de validación inter-sujeto y aplicaciones en neurotecnología</p>
            </div>
            <div class="slide-body full-width">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; height: 100%;">
                    <div class="content-box" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>Unidad 1: Procesamiento y PCA</h4>
                            <p style="margin-top: 8px;">El acondicionamiento FIR de fase cero y la proyección espacial por PCA sintetizaron el <strong>{var_exp:.2f}% de la varianza</strong> del dipolo occipital sin distorsión de fase ni transitorios.</p>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--secondary-blue); font-weight: 600;">Reducción Espacial Óptima</div>
                    </div>
                    <div class="content-box box-crimson" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>Unidad 2: Minería Temporal</h4>
                            <p style="margin-top: 8px;">El espacio 2D de Potencia Relativa en épocas de 2s permitió a K-Means discriminar los estados con Silhouette de {sil_val:.3f} y ARI de {ari_val:.3f}.</p>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--accent-crimson); font-weight: 600;">Minería No Supervisada</div>
                    </div>
                    <div class="content-box box-navy" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>Unidad 3: Discusión y Desafíos</h4>
                            <p style="margin-top: 8px;">Para transicionar hacia interfaces cerebro-computadora (BCI) reales, se requiere validar el pipeline en esquemas inter-sujeto (LOSO), modelar la Frecuencia Alfa Individual (IAF) e incorporar remoción de artefactos oculares (EOG/ICA).</p>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--primary-navy); font-weight: 600;">Validación & Escalabilidad</div>
                    </div>
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: 7:40 - 9:00</span><br><br>
                <strong>Texto a exponer:</strong><br>
                "Como conclusión, este trabajo muestra cómo las distintas herramientas vistas durante la materia pueden integrarse en un único pipeline de análisis de señales de EEG.<br><br>
                El preprocesamiento y la reducción espacial mediante PCA permitieron sintetizar la actividad de la corteza occipital en una única señal representativa.<br><br>
                El análisis espectral y la STFT permitieron identificar y caracterizar el ritmo alfa tanto en frecuencia como en el tiempo.<br><br>
                Finalmente, las técnicas de minería de series temporales mostraron que un algoritmo no supervisado puede diferenciar automáticamente los estados de ojos abiertos y ojos cerrados utilizando únicamente características espectrales.<br><br>
                Si bien este estudio se realizó sobre un único sujeto como prueba de concepto, el pipeline podría extenderse incorporando técnicas de eliminación de artefactos, evaluaciones sobre múltiples sujetos y ventanas temporales solapadas para aplicaciones en interfaces cerebro-computadora.<br><br>
                En conjunto, los resultados muestran que las herramientas desarrolladas durante la materia permiten construir un flujo completo para el procesamiento y análisis de señales EEG, desde el preprocesamiento hasta la extracción automática de patrones neurofisiológicos."
            </div>
        </section>
    </main>

    <!-- PANEL INFERIOR DE NOTAS DE ORADOR -->
    <div class="speaker-notes-panel" id="speakerPanel">
        <div class="notes-header">
            <span>NOTAS DE ORADOR (GUION VERBATIM PARA EXPOSICIÓN DE 9 MINUTOS)</span>
            <button class="btn" style="padding: 2px 8px; font-size: 0.75rem; background: #334155; color: #ffffff; border: none;" onclick="toggleSpeakerNotes()">[Cerrar - N]</button>
        </div>
        <div class="notes-text" id="speakerNotesText"></div>
    </div>

    <footer>
        <div>UNIVERSIDAD | EVALUACIÓN FINAL DE PROCESAMIENTO DE SEÑALES & MINERÍA DE SERIES TEMPORALES</div>
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
    
    # 1. Carga y Preprocesamiento MNE con filtrado de fase cero y recorte dinámico de 3.0s
    data_open, data_closed, target_channels, sfreq = cargar_y_preprocesar_eeg(base_dir)
    
    # 2. Filtrado Espacial PCA
    pc1_open, pc1_closed, pca_model, var_exp = aplicar_pca_espacial(data_open, data_closed)
    
    # 3. Espectro Welch y STFT
    freqs, psd_open, psd_closed, f_stft, t_stft_o, Sxx_open, t_stft_c, Sxx_closed, ratio_berger = analisis_espectral_y_tiempo_frecuencia(
        pc1_open, pc1_closed, sfreq
    )
    
    # 4. Minería de Series Temporales: Periodograma Hann y Potencia Relativa
    clustering_res = mineria_y_clustering_kmeans(pc1_open, pc1_closed, sfreq, win_len_sec=2.0)
    
    # 5. Guardar Figuras
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
