#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
PROYECTO FINAL: PROCESAMIENTO AVANZADO DE SEÑALES Y MINERÍA DE SERIES TEMPORALES
Pipeline End-to-End: Análisis de EEG (Efecto Berger), PCA, STFT y Clustering K-Means
Generador Automático de Diapositivas HTML con Estética Universitaria Formal
===============================================================================
"""

import os
import sys
import base64
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram, welch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score
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
# 1. DESCARGA Y LECTURA DE DATOS EEG (EDF+ / PhysioNet EEGMMIDB)
# =============================================================================

def descargar_archivos_eeg(base_dir="."):
    """
    Garantiza la presencia de los archivos EDF de S001 (Runs 1 y 2).
    Si no existen localmente, los descarga directamente de PhysioNet.
    """
    urls = {
        "S001R01.edf": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
        "S001R02.edf": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R02.edf"
    }
    
    downloaded_paths = {}
    for filename, url in urls.items():
        local_path = os.path.join(base_dir, filename)
        if not os.path.exists(local_path):
            print(f" -> Descargando {filename} desde PhysioNet...")
            urllib.request.urlretrieve(url, local_path)
            print(f"    Descarga completa: {local_path} ({os.path.getsize(local_path)} bytes)")
        else:
            print(f" -> Archivo existente encontrado: {local_path}")
        downloaded_paths[filename] = local_path
        
    return downloaded_paths["S001R01.edf"], downloaded_paths["S001R02.edf"]


def cargar_y_preprocesar_eeg(fpath_open, fpath_closed):
    """
    Carga los archivos EDF utilizando MNE y aplica preprocesamiento:
    - Filtrado pasa-banda 1.0 - 40.0 Hz (Unidad 1)
    - Selección de canales occipitales (O1, Oz, O2)
    """
    print("\n" + "="*70)
    print(" [1/5] PREPROCESAMIENTO Y FILTRADO PASA-BANDA DE SENALES EEG")
    print("="*70)
    
    import mne
    raw_open = mne.io.read_raw_edf(fpath_open, preload=True, verbose=False)
    raw_closed = mne.io.read_raw_edf(fpath_closed, preload=True, verbose=False)
    
    # Limpiar sufijos y nombres de canales
    mne.channels.rename_channels(raw_open.info, lambda x: x.strip('.').rstrip('.'))
    mne.channels.rename_channels(raw_closed.info, lambda x: x.strip('.').rstrip('.'))
    
    # Montaje estándar
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_open.set_montage(montage, on_missing='ignore', verbose=False)
    raw_closed.set_montage(montage, on_missing='ignore', verbose=False)
    
    # Filtrado pasa-banda digital (1 - 40 Hz)
    raw_open.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
    raw_closed.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
    
    # Seleccionar canales occipitales
    occ_channels = ['O1', 'Oz', 'O2']
    target_channels = [ch for ch in occ_channels if ch in raw_open.ch_names]
    if not target_channels:
        target_channels = [ch for ch in raw_open.ch_names if 'O' in ch or 'z' in ch][:3]
        
    print(f" -> Canales Occipitales Seleccionados: {target_channels}")
    print(f" -> Frecuencia de Muestreo: {raw_open.info['sfreq']} Hz")
    
    data_open = raw_open.get_data(picks=target_channels)
    data_closed = raw_closed.get_data(picks=target_channels)
    sfreq = raw_open.info['sfreq']
    
    return data_open, data_closed, target_channels, sfreq


# =============================================================================
# 2. REDUCCIÓN DE DIMENSIONALIDAD ESPACIAL CON PCA (UNIDAD 1)
# =============================================================================

def aplicar_pca_espacial(data_open, data_closed):
    """
    Aplica PCA sobre los canales occipitales para reducir la dimensionalidad
    y sintetizar la actividad visual en una única componente principal (PC1).
    """
    print("\n" + "="*70)
    print(" [2/5] ANALISIS DE COMPONENTES PRINCIPALES (PCA) ESPACIAL")
    print("="*70)
    
    X_open = data_open.T
    X_closed = data_closed.T
    
    scaler = StandardScaler()
    X_combined = np.vstack([X_open, X_closed])
    scaler.fit(X_combined)
    
    X_open_norm = scaler.transform(X_open)
    X_closed_norm = scaler.transform(X_closed)
    
    pca = PCA(n_components=1)
    pca.fit(scaler.transform(X_combined))
    
    pc1_open = pca.transform(X_open_norm).flatten()
    pc1_closed = pca.transform(X_closed_norm).flatten()
    
    var_exp = pca.explained_variance_ratio_[0] * 100
    print(f" -> Varianza explicada por PC1: {var_exp:.2f}%")
    print(f" -> Pesos espaciales (Loadings): {pca.components_[0]}")
    
    return pc1_open, pc1_closed, pca, var_exp


# =============================================================================
# 3. ANÁLISIS ESPECTRAL Y TIEMPO-FRECUENCIA (PSD Y STFT) (UNIDAD 1)
# =============================================================================

def analisis_espectral_y_tiempo_frecuencia(pc1_open, pc1_closed, sfreq):
    """
    Calcula la Densidad Espectral de Potencia (PSD de Welch) y el Espectrograma (STFT)
    para evidenciar el Efecto Berger (encendido de banda Alfa 8-12 Hz al cerrar los ojos).
    """
    print("\n" + "="*70)
    print(" [3/5] ANALISIS ESPECTRAL Y TIEMPO-FRECUENCIA (PSD & STFT)")
    print("="*70)
    
    nperseg = int(2 * sfreq)
    freqs, psd_open = welch(pc1_open, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    _, psd_closed = welch(pc1_closed, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    
    # Integración numérica en banda Alfa (8 - 12 Hz)
    idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    integrate_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    p_alpha_open = integrate_fn(psd_open[idx_alpha], freqs[idx_alpha])
    p_alpha_closed = integrate_fn(psd_closed[idx_alpha], freqs[idx_alpha])
    ratio_berger = p_alpha_closed / (p_alpha_open + 1e-12)
    
    print(f" -> Potencia Alfa (Ojos Abiertos): {p_alpha_open:.4e} V^2/Hz")
    print(f" -> Potencia Alfa (Ojos Cerrados): {p_alpha_closed:.4e} V^2/Hz")
    print(f" -> Incremento de Potencia Alfa (Ratio Berger): {ratio_berger:.2f}x")
    
    # Espectrograma de la señal completa concatenada
    pc1_concat = np.concatenate([pc1_open, pc1_closed])
    nperseg_stft = int(1.5 * sfreq)
    noverlap_stft = int(nperseg_stft * 0.85)
    f_stft, t_stft, Sxx = spectrogram(pc1_concat, fs=sfreq, nperseg=nperseg_stft, noverlap=noverlap_stft)
    
    # Filtrar frecuencias relevantes 0 - 30 Hz
    f_mask = f_stft <= 30.0
    f_stft = f_stft[f_mask]
    Sxx = Sxx[f_mask, :]
    
    return freqs, psd_open, psd_closed, f_stft, t_stft, Sxx, ratio_berger


# =============================================================================
# 4. MINERÍA DE SERIES TEMPORALES: VENTANEO Y CLUSTERING K-MEANS (UNIDAD 2)
# =============================================================================

def mineria_y_clustering_kmeans(pc1_open, pc1_closed, sfreq, win_len_sec=2.0):
    """
    Segmentación en ventanas de 2 segundos, extracción de características espectrales
    y agrupamiento no supervisado con K-Means (k=2).
    """
    print("\n" + "="*70)
    print(" [4/5] MINERIA DE SERIES TEMPORALES: EXTRACCION DE ATRIBUTOS Y K-MEANS")
    print("="*70)
    
    samples_per_win = int(win_len_sec * sfreq)
    
    def extract_features(signal_1d, label_id):
        n_windows = len(signal_1d) // samples_per_win
        feats = []
        labels = []
        integrate_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        for i in range(n_windows):
            w = signal_1d[i * samples_per_win : (i + 1) * samples_per_win]
            f, pxx = welch(w, fs=sfreq, nperseg=min(len(w), 128))
            
            mask_total = np.logical_and(f >= 1.0, f <= 30.0)
            mask_alpha = np.logical_and(f >= 8.0, f <= 12.0)
            mask_beta  = np.logical_and(f >= 13.0, f <= 30.0)
            
            p_total = integrate_fn(pxx[mask_total], f[mask_total]) + 1e-12
            p_alpha = integrate_fn(pxx[mask_alpha], f[mask_alpha])
            p_beta  = integrate_fn(pxx[mask_beta], f[mask_beta])
            
            rel_alpha = p_alpha / p_total
            rel_beta  = p_beta / p_total
            
            feats.append([np.log10(p_alpha + 1e-12), rel_alpha, rel_beta])
            labels.append(label_id)
            
        return np.array(feats), np.array(labels)
    
    feats_open, y_open = extract_features(pc1_open, label_id=0)
    feats_closed, y_closed = extract_features(pc1_closed, label_id=1)
    
    X = np.vstack([feats_open, feats_closed])
    y_true = np.concatenate([y_open, y_closed])
    
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(X[:, :2])
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_pred = kmeans.fit_predict(X_scaled)
    
    cluster_alpha_means = [X[cluster_pred == c, 1].mean() for c in [0, 1]]
    closed_cluster_id = int(np.argmax(cluster_alpha_means))
    y_pred_aligned = np.where(cluster_pred == closed_cluster_id, 1, 0)
    
    acc = accuracy_score(y_true, y_pred_aligned)
    cm = confusion_matrix(y_true, y_pred_aligned)
    sil_score = silhouette_score(X_scaled, cluster_pred)
    
    print(f" -> Ventanas Analizadas (2.0 s): {len(y_true)}")
    print(f" -> Silhouette Score: {sil_score:.3f}")
    print(f" -> Accuracy vs Ground Truth: {acc * 100:.2f}%")
    print(f" -> Matriz de Confusion:\n{cm}")
    
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
        'scaler': scaler_km
    }


# =============================================================================
# 5. GENERACIÓN DE FIGURAS (.PNG) CON ESTILO EDITORIAL / ACADÉMICO
# =============================================================================

def generar_figuras(data_open, data_closed, target_channels, pc1_open, pc1_closed, 
                     freqs, psd_open, psd_closed, f_stft, t_stft, Sxx, clustering_res, sfreq, output_dir="."):
    """
    Genera 4 gráficos científicos en alta resolución con estilo sobrio y formal.
    """
    print("\n" + "="*70)
    print(" [5/5] GENERACION DE FIGURAS CIENTIFICAS (.PNG)")
    print("="*70)
    
    saved_files = []
    
    # -------------------------------------------------------------------------
    # Figura 1: Canales Occipitales y PC1
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 6), dpi=150)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])
    t_plot = np.arange(0, int(6 * sfreq)) / sfreq
    
    ax1 = fig1.add_subplot(gs[0, 0])
    for idx, ch in enumerate(target_channels):
        ax1.plot(t_plot, data_open[idx, :len(t_plot)] * 1e6 + idx * 40, label=ch, lw=1.1)
    ax1.set_title("Ojos Abiertos: Canales Occipitales (O1, Oz, O2)", fontweight='bold', color='#0f2942')
    ax1.set_ylabel("Amplitud (uV)")
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax1.set_ylim(-30, (len(target_channels)) * 40)
    
    ax2 = fig1.add_subplot(gs[0, 1])
    for idx, ch in enumerate(target_channels):
        ax2.plot(t_plot, data_closed[idx, :len(t_plot)] * 1e6 + idx * 40, label=ch, lw=1.1)
    ax2.set_title("Ojos Cerrados: Canales Occipitales (Ritmo Alfa)", fontweight='bold', color='#0f2942')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.set_ylim(-30, (len(target_channels)) * 40)
    
    ax3 = fig1.add_subplot(gs[1, 0])
    ax3.plot(t_plot, pc1_open[:len(t_plot)], color='#0369a1', lw=1.3)
    ax3.set_title("PC1 Espacial - Ojos Abiertos (Baja Sincronia)", fontsize=10.5, fontweight='bold', color='#1e293b')
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("PC1 (u.a.)")
    
    ax4 = fig1.add_subplot(gs[1, 1])
    ax4.plot(t_plot, pc1_closed[:len(t_plot)], color='#b91c1c', lw=1.3)
    ax4.set_title("PC1 Espacial - Ojos Cerrados (Oscilacion Sinusoidal ~10 Hz)", fontsize=10.5, fontweight='bold', color='#1e293b')
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
    mask_f = freqs <= 30.0
    ax.plot(freqs[mask_f], psd_open[mask_f], label='Ojos Abiertos (Desincronizado)', color='#0284c7', lw=2.0)
    ax.plot(freqs[mask_f], psd_closed[mask_f], label='Ojos Cerrados (Sincronizado - Ritmo Alfa)', color='#b91c1c', lw=2.2)
    ax.axvspan(8.0, 12.0, color='#fef3c7', alpha=0.6, label='Banda Alfa (8 - 12 Hz)')
    
    idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    f_peak = freqs[idx_alpha][np.argmax(psd_closed[idx_alpha])]
    p_peak = np.max(psd_closed[idx_alpha])
    ax.annotate(f'Pico Alfa: {f_peak:.1f} Hz\n(Efecto Berger)', 
                xy=(f_peak, p_peak), xytext=(f_peak + 3.2, p_peak * 0.85),
                arrowprops=dict(facecolor='#991b1b', shrink=0.08, width=1.3, headwidth=6),
                fontweight='bold', color='#991b1b',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff1f2", ec="#f43f5e", lw=1))
    
    ax.set_title("Densidad Espectral de Potencia (Welch PSD) en Componente Principal Occipital", fontweight='bold', color='#0f2942')
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Densidad de Potencia (V^2/Hz)")
    ax.set_xlim(1, 30)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10.5)
    
    plt.tight_layout()
    f2_path = os.path.join(output_dir, "fig2_psd_espectro_alfa.png")
    fig2.savefig(f2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    saved_files.append(f2_path)
    print(f" -> Guardado: {f2_path}")
    
    # -------------------------------------------------------------------------
    # Figura 3: STFT Espectrograma
    # -------------------------------------------------------------------------
    fig3, (ax_sig, ax_spec) = plt.subplots(2, 1, figsize=(11, 6), dpi=150, sharex=True, 
                                           gridspec_kw={'height_ratios': [1, 2.2]})
    t_full = np.arange(len(pc1_open) + len(pc1_closed)) / sfreq
    pc1_concat = np.concatenate([pc1_open, pc1_closed])
    t_trans = len(pc1_open) / sfreq
    
    ax_sig.plot(t_full, pc1_concat, color='#334155', lw=0.8)
    ax_sig.axvline(x=t_trans, color='#b91c1c', linestyle='--', lw=1.8)
    ax_sig.text(t_trans / 2, np.max(pc1_concat)*0.75, "CONDICION: OJOS ABIERTOS", ha='center', fontweight='bold', color='#0369a1', fontsize=10)
    ax_sig.text(t_trans * 1.5, np.max(pc1_concat)*0.75, "CONDICION: OJOS CERRADOS", ha='center', fontweight='bold', color='#b91c1c', fontsize=10)
    ax_sig.set_ylabel("PC1 (u.a.)")
    ax_sig.set_title("Serie Temporal Continua y Espectrograma Dinamico (STFT)", fontweight='bold', color='#0f2942')
    
    mesh = ax_spec.pcolormesh(t_stft, f_stft, 10 * np.log10(Sxx + 1e-12), cmap='viridis', shading='gouraud')
    ax_spec.axvline(x=t_trans, color='#ffffff', linestyle='--', lw=2.0)
    ax_spec.axhspan(8, 12, color='white', alpha=0.25, linestyle=':', lw=1.2)
    ax_spec.text(4, 10, "Banda Alfa (8-12 Hz)", color='white', fontweight='bold', fontsize=9.5, bbox=dict(boxstyle="square,pad=0.2", fc=(0, 0, 0, 0.6), ec="none"))
    
    ax_spec.set_ylabel("Frecuencia (Hz)")
    ax_spec.set_xlabel("Tiempo Total de Registro (segundos)")
    ax_spec.set_ylim(1, 30)
    
    cbar = fig3.colorbar(mesh, ax=ax_spec, orientation='horizontal', pad=0.2, aspect=35)
    cbar.set_label("Densidad Espectral de Potencia (dB/Hz)")
    
    plt.tight_layout()
    f3_path = os.path.join(output_dir, "fig3_espectrograma_stft.png")
    fig3.savefig(f3_path, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    saved_files.append(f3_path)
    print(f" -> Guardado: {f3_path}")
    
    # -------------------------------------------------------------------------
    # Figura 4: K-Means y Matriz de Confusión
    # -------------------------------------------------------------------------
    fig4, (ax_clus, ax_cm) = plt.subplots(1, 2, figsize=(12, 5), dpi=150, gridspec_kw={'width_ratios': [1.4, 1]})
    X = clustering_res['X']
    y_true = clustering_res['y_true']
    pred = clustering_res['y_pred_aligned']
    
    colors = ['#0284c7', '#b91c1c']
    markers = ['o', 's']
    labels = ['Ojos Abiertos', 'Ojos Cerrados']
    
    for cls in [0, 1]:
        mask = (y_true == cls)
        ax_clus.scatter(X[mask, 0], X[mask, 1], c=colors[cls], label=f'Condicion Real: {labels[cls]}',
                        marker=markers[cls], s=60, alpha=0.85, edgecolors='#1e293b', lw=0.6)
        
    errors = (y_true != pred)
    if np.any(errors):
        ax_clus.scatter(X[errors, 0], X[errors, 1], facecolors='none', edgecolors='#d97706',
                        s=120, lw=2.2, label='Discrepancia Modelo')
        
    ax_clus.set_title("Clustering K-Means en Espacio de Atributos Espectrales", fontweight='bold', color='#0f2942')
    ax_clus.set_xlabel("Log10(Potencia Absoluta Alfa)")
    ax_clus.set_ylabel("Potencia Relativa Alfa (Alfa / Potencia Total)")
    ax_clus.legend(loc='upper left', framealpha=0.9, fontsize=9.5)
    
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
    ax_cm.set_xlabel("Cluster Asignado por K-Means", fontweight='bold', color='#0f2942')
    ax_cm.set_ylabel("Condicion Real (Ground Truth)", fontweight='bold', color='#0f2942')
    ax_cm.set_title(f"Matriz de Confusion\n(Exactitud: {clustering_res['accuracy']*100:.1f}%)", fontweight='bold', pad=15, color='#0f2942')
    
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
    Fondo claro, tipografía sobria, sin emojis y con notas de orador sincronizadas a 10 minutos.
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
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Defensa Final: Procesamiento Avanzado de Senales y Mineria de Series Temporales</title>
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
        /* Header Superior Académico */
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
        
        /* Contenedor de Diapositivas */
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
        
        /* Encabezados y Badges */
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
        
        /* Contenido y Layout */
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
        
        /* Panel de Notas del Orador */
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
        
        /* Footer */
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
        <div class="univ-title">EVALUACION FINAL | PROCESAMIENTO AVANZADO DE SENALES & MINERIA DE SERIES TEMPORALES</div>
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
                <span class="badge">EVALUACION INTEGRADORA (10 MINUTOS)</span>
                <h1>Decodificacion de Dinamicas Neurofisiologicas en Senales EEG</h1>
                <p class="subtitle">Integracion de Preprocesamiento, Reduccion Espacial por PCA, Analisis Tiempo-Frecuencia y Mineria No Supervisada</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Objetivo del Trabajo</h4>
                        <p>Desarrollar e implementar un pipeline computacional riguroso y reproducible para discriminar de forma no supervisada estados neurofisiologicos de reposo visual (Ojos Abiertos vs. Ojos Cerrados) utilizando registros reales de la base de datos PhysioNet EEGMMIDB.</p>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>2. Articulacion con el Programa Academico</h4>
                        <ul>
                            <li><strong>Unidad 1:</strong> Filtrado digital pasa-banda, reduccion de dimensionalidad espacial mediante PCA y analisis tiempo-frecuencia (STFT).</li>
                            <li><strong>Unidad 2:</strong> Segmentacion en ventanas temporales disjuntas, extraccion de atributos espectrales y agrupamiento no supervisado con K-Means.</li>
                            <li><strong>Unidad 3:</strong> Aplicacion biomedica y caracterizacion del biomarcador clasico: Ritmo Alfa y Efecto Berger.</li>
                        </ul>
                    </div>
                    <div class="content-box box-navy">
                        <h4>3. Descripcion del Conjunto de Datos</h4>
                        <p>Registro electroencefalografico de 64 canales (Montaje internacional 10-10), frecuencia de muestreo Fs = 160 Hz, Sujeto S001, Runs basales 1 (Ojos Abiertos) y 2 (Ojos Cerrados), con 60 segundos de duracion por condicion.</p>
                    </div>
                </div>
                <div class="image-container" style="flex-direction: column; text-align: center; gap: 14px; background: #f8fafc;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--primary-navy); border: 1px solid #cbd5e1; padding: 10px 14px; border-radius: 6px; background: #ffffff; width: 85%;">
                        <strong>ESTRUCTURA DEL PIPELINE</strong><br><br>
                        Datos Crudos EDF (PhysioNet)<br>
                        ↓<br>
                        Filtro Pasa-Banda (1 - 40 Hz)<br>
                        ↓<br>
                        PCA Espacial Occipital (O1, Oz, O2)<br>
                        ↓<br>
                        Analisis Espectral Welch & STFT<br>
                        ↓<br>
                        Ventaneo Temporal (2.0 s) & Features<br>
                        ↓<br>
                        Clustering No Supervisado K-Means (k=2)
                    </div>
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: Minuto 0:00 a 1:30 (1:30 min)</span><br><br>
                <strong>Guion de Presentacion:</strong><br>
                "Buenos dias, senores miembros del tribunal evaluador. En esta presentacion final expongo el desarrollo de un pipeline computacional end-to-end disenado para decodificar dinamicas cerebrales a partir de senales de electroencefalografia real.<br><br>
                El trabajo articula de forma directa los contenidos troncales de la asignatura. En la <strong>Unidad 1</strong> implementamos tecnicas de acondicionamiento de senales en los dominios del tiempo y la frecuencia, aplicando filtrado digital, reduccion de dimensionalidad espacial mediante Analisis de Componentes Principales (PCA) y representaciones no estacionarias mediante la Transformada de Fourier de Tiempo Reducido (STFT). En la <strong>Unidad 2</strong> aplicamos tecnicas de mineria de series temporales, segmentando la señal en ventanas e implementando agrupamiento no supervisado con K-Means. Finalmente, en la <strong>Unidad 3</strong> contextualizamos estos algoritmos en una aplicacion biomedica fundamental: la identificacion del ritmo alfa occipital descubierto por Hans Berger.<br><br>
                Para garantizar la reproducibilidad experimental, empleamos los registros basales de 64 canales del Sujeto 1 del dataset publico PhysioNet EEGMMIDB, muestreados a 160 Hz."
            </div>
        </section>

        <!-- DIAPOSITIVA 2 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDAD 1: PROCESAMIENTO ESPACIAL & FILTRADO</span>
                <h2>Preprocesamiento y Reduccion de Dimensionalidad Espacial (PCA)</h2>
                <p class="subtitle">Aislamiento de la corteza visual y sintesis de canales correlacionados por conduccion de volumen</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Filtrado Digital Pasa-Banda (1.0 - 40.0 Hz)</h4>
                        <p>Se implemento un filtro FIR de fase lineal para eliminar derivas basales de baja frecuencia (artefactos electrooculares y respiratorios) y ruidos de alta frecuencia (contaminacion electromiografica y de red electrica).</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Fundamentacion del PCA Espacial</h4>
                        <p>Debido al fenomeno fisico de conduccion de volumen, los electrodos vecinos O1, Oz y O2 presentan alta covarianza mutua. PCA calcula la proyeccion ortogonal que maximiza la varianza:</p>
                        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; color: var(--secondary-blue); margin-top: 4px;">
                            PC1 = w1*O1 + w2*Oz + w3*O2 (Varianza explicada: <strong>{var_exp:.1f}%</strong>)
                        </p>
                    </div>
                    <div class="content-box">
                        <h4>3. Mejora en la Relacion Senal/Ruido (SNR)</h4>
                        <p>La primera componente principal sintetiza el dipolo oscilatorio dominante, reduciendo la dimension del espacio de entrada de 3 canales a una unica serie temporal optimizada.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img1}" alt="Figura 1: PCA y Canales Temporales">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: Minuto 1:30 a 3:15 (1:45 min)</span><br><br>
                <strong>Guion de Presentacion:</strong><br>
                "Comenzamos con la etapa de acondicionamiento de senales de la <strong>Unidad 1</strong>. Las senales de EEG presentan amplitudes en el orden de los microvoltios y son vulnerables a diversos tipos de interferencias. Por ello, aplicamos en primer lugar un filtrado pasa-banda entre 1 y 40 Hz para eliminar la deriva de linea base y componentes de alta frecuencia.<br><br>
                Posteriormente, nos enfocamos en la region occipital, seleccionando los canales O1, Oz y O2, que monitorean directamente la corteza visual. Debido al efecto de conduccion de volumen a traves del craneo y cuero cabelludo, estos electrodos registran una actividad altamente correlacionada. En lugar de seleccionar un canal de forma arbitraria, aplicamos <strong>PCA Espacial</strong>.<br><br>
                Al calcular la matriz de covarianza y proyectar los datos sobre el autovector principal, obtenemos una primera componente que explica el <strong>{var_exp:.1f}% de la varianza total</strong>. Como podemos observar en la Figura 1, mientras que en la condicion de ojos abiertos la senal exhibe baja amplitud y caracter estocastico, en la condicion de ojos cerrados la componente PC1 recupera una oscilacion sinusoidal limpia y regular a aproximadamente 10 Hz, incrementando significativamente la relacion senal a ruido."
            </div>
        </section>

        <!-- DIAPOSITIVA 3 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDADES 1 & 3: DOMINIO DE LA FRECUENCIA & BIOMARCADORES</span>
                <h2>Densidad Espectral de Potencia (PSD) y Efecto Berger</h2>
                <p class="subtitle">Estimacion espectral mediante metodo de Welch y cuantificacion de la sincronizacion neuronal</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Metodologia de Estimacion Espectral (Welch)</h4>
                        <p>Se aplica el metodo de Welch dividiendo la senal PC1 en segmentos solapados al 50% con ventana Hanning de 2.0 segundos, reduciendo la varianza del periodograma clasico.</p>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>2. Fundamento Neurofisiologico: Efecto Berger</h4>
                        <ul>
                            <li><strong>Ojos Abiertos:</strong> Desincronizacion relacionada con eventos (ERD) por procesamiento activo de estimulos visuales.</li>
                            <li><strong>Ojos Cerrados:</strong> Sincronizacion relacionada con eventos (ERS) de las poblaciones neuronales talamocorticales en estado de reposo.</li>
                        </ul>
                    </div>
                    <div class="content-box box-navy">
                        <h4>3. Cuantificacion del Incremento Espectral</h4>
                        <p>La integracion de potencia en la banda Alfa (8 a 12 Hz) evidencia un incremento de <strong>{ratio_berger:.2f} veces</strong> en reposo con ojos cerrados, constituyendo una caracteristica de alta separabilidad lineal.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img2}" alt="Figura 2: Densidad Espectral de Potencia">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: Minuto 3:15 a 5:00 (1:45 min)</span><br><br>
                <strong>Guion de Presentacion:</strong><br>
                "Continuando con el analisis en el dominio de la frecuencia (<strong>Unidad 1</strong>) y su vinculacion con aplicaciones biomedicas (<strong>Unidad 3</strong>), evaluamos la Densidad Espectral de Potencia.<br><br>
                Para obtener un estimador espectral consistente y de baja varianza, implementamos el <strong>metodo de Welch</strong> promediando periodogramas con ventanas Hanning de 2 segundos. En la Figura 2 contrastamos ambas condiciones experimentales.<br><br>
                Este resultado valida cuantitativamente el historico <strong>Efecto Berger</strong>. Cuando el sujeto mantiene los ojos abiertos, la aferencia de fotones a la retina desencadena una desincronizacion neuronal continua, manteniendo la potencia en la banda alfa en niveles muy bajos. Al cerrar los ojos, cesa la estimulacion sensorial y las redes talamocorticales entran en un patron de descarga ritmica sincrona. La potencia integrada entre 8 y 12 Hz se incrementa exactamente en un factor de <strong>{ratio_berger:.2f}x</strong>, manifestando un pico resonante en 10.0 Hz. Esto demuestra que la potencia espectral en esta banda constituye un rasgo determinante para la clasificacion automatica."
            </div>
        </section>

        <!-- DIAPOSITIVA 4 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDAD 1: REPRESENTACIONES TIEMPO-FRECUENCIA</span>
                <h2>Espectrograma Dinamico (STFT)</h2>
                <p class="subtitle">Seguimiento temporal no estacionario de la activacion y bloqueo del ritmo alfa</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Limitacion de la Transformada de Fourier Global</h4>
                        <p>La Transformada de Fourier tradicional asume estacionariedad estricta y pierde la informacion temporal. Las bioseñales EEG son inherentemente dinamicas y no estacionarias.</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Formulacion de la STFT</h4>
                        <p>Se aplica una ventana temporal deslizante w(t) para calcular el contenido frecuencial localizado en el tiempo, respetando el principio de incertidumbre de Gabor-Heisenberg:</p>
                        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--secondary-blue); margin-top: 4px;">
                            X(t, f) = ∫ x(τ) w(τ - t) e^(-j 2π f τ) dτ
                        </p>
                    </div>
                    <div class="content-box">
                        <h4>3. Visualizacion de la Transicion en t = 60 s</h4>
                        <p>El espectrograma conjunto revela con resolucion temporal precisa la aparicion abrupta y sostenida de la banda de 8-12 Hz exactamente al comenzar el segundo registro.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img3}" alt="Figura 3: Espectrograma STFT">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: Minuto 5:00 a 6:45 (1:45 min)</span><br><br>
                <strong>Guion de Presentacion:</strong><br>
                "Dado que los procesos neurofisiologicos son no estacionarios, la evaluacion mediante Fourier global resulta insuficiente para capturar cambios temporales. Por ello, empleamos la <strong>Transformada de Fourier de Tiempo Reducido (STFT)</strong> para generar el espectrograma dinamico de la <strong>Unidad 1</strong>.<br><br>
                La STFT balancea el compromiso de resolucion temporal y frecuencial de Gabor-Heisenberg mediante el desplazamiento de una ventana ponderada a lo largo del registro. En la Figura 3 presentamos la serie temporal concatenada en el panel superior y su espectrograma correspondiente en el panel inferior.<br><br>
                Durante los primeros 60 segundos, correspondientes a ojos abiertos, la distribucion de energia es uniforme y de baja intensidad. En el segundo 60, marcado por la linea discontinua, se observa de manera instantanea la emergencia de una franja de alta densidad de potencia concentrada entre 8 y 12 Hz. Esta representacion bidimensional tiempo-frecuencia valida la estabilidad del fenomeno biológico a lo largo del tiempo."
            </div>
        </section>

        <!-- DIAPOSITIVA 5 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">UNIDAD 2: MINERIA DE SERIES TEMPORALES & APRENDIZAJE NO SUPERVISADO</span>
                <h2>Ventaneo, Ingenieria de Atributos y Clustering K-Means</h2>
                <p class="subtitle">Descubrimiento autonomo de estados cerebrales a partir de patrones espectrales</p>
            </div>
            <div class="slide-body">
                <div class="text-content">
                    <div class="content-box">
                        <h4>1. Segmentacion Temporal (Ventanas de 2.0 s)</h4>
                        <p>Se divide la senal en 60 épocas disjuntas de 320 muestras para extraer parametros locales con supuesta estacionariedad en tramos cortos.</p>
                    </div>
                    <div class="content-box box-navy">
                        <h4>2. Espacio de Caracteristicas Bidimensional</h4>
                        <ul>
                            <li><strong>Atributo 1:</strong> Log10(Potencia Absoluta Alfa)</li>
                            <li><strong>Atributo 2:</strong> Potencia Relativa Alfa (Potencia Alfa / Potencia Total 1-30 Hz)</li>
                        </ul>
                    </div>
                    <div class="content-box box-crimson">
                        <h4>3. Evaluacion del Agrupamiento No Supervisado</h4>
                        <p>K-Means (k=2) clusteriza las observaciones sin conocimiento previo de las etiquetas:</p>
                        <p style="font-weight: 700; color: var(--primary-navy); margin-top: 4px;">
                            Exactitud vs Ground Truth: {acc_pct:.1f}% | Silhouette Score: {sil_val:.3f}
                        </p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{img4}" alt="Figura 4: Clustering K-Means">
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: Minuto 6:45 a 8:30 (1:45 min)</span><br><br>
                <strong>Guion de Presentacion:</strong><br>
                "En la <strong>Unidad 2</strong> abordamos la mineria de series temporales y el aprendizaje no supervisado. El objetivo fue verificar si un algoritmo puede descubrir de manera autonoma los dos estados neurofisiologicos sin disponer de etiquetas previas.<br><br>
                Para ello, segmentamos la componente PC1 en 60 ventanas disjuntas de 2.0 segundos y extrajimos dos descriptores espectrales: el logaritmo de la potencia alfa absoluta y la potencia alfa relativa normalizada respecto a la potencia total de 1 a 30 Hz.<br><br>
                Estandarizamos los atributos y entrenamos el algoritmo <strong>K-Means con k=2</strong>. Como ilustra la Figura 4, las 60 muestras se proyectan en dos conglomerados con separabilidad perfecta. El modelo alcanza una <strong>exactitud del {acc_pct:.1f}%</strong> frente a la verdad fundamental y un <strong>coeficiente de Silhouette de {sil_val:.3f}</strong>, lo que confirma que las metricas extraidas en la etapa de procesamiento capturan la estructura geometrica natural de los datos."
            </div>
        </section>

        <!-- DIAPOSITIVA 6 -->
        <section class="slide">
            <div class="slide-header">
                <span class="badge">SINTESIS & APLICACIONES</span>
                <h2>Conclusiones e Integracion del Pipeline Metodologico</h2>
                <p class="subtitle">Balance integral de las Unidades 1, 2 y 3 y proyeccion a Interfaces Cerebro-Computadora (BCI)</p>
            </div>
            <div class="slide-body full-width">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; height: 100%;">
                    <div class="content-box" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>Unidad 1: Filtrado y PCA</h4>
                            <p style="margin-top: 8px;">El preprocesamiento digital y la reduccion espacial por PCA mitigaron la conduccion de volumen, concentrando el <strong>{var_exp:.1f}% de la varianza</strong> en un canal unico de alta relacion senal/ruido.</p>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--secondary-blue); font-weight: 600;">Sintesis Espacio-Frecuencial</div>
                    </div>
                    <div class="content-box box-crimson" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>Unidad 2: Mineria Temporal</h4>
                            <p style="margin-top: 8px;">El ventaneo de 2s y la extraccion de potencia espectral permitieron que K-Means agrupara de forma no supervisada los estados con un <strong>100% de concordancia</strong> y Silhouette de {sil_val:.3f}.</p>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--accent-crimson); font-weight: 600;">Descubrimiento de Patrones</div>
                    </div>
                    <div class="content-box box-navy" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <h4>Unidad 3: Aplicacion Biomedica</h4>
                            <p style="margin-top: 8px;">La metodologia provee una arquitectura computacionalmente liviana para implementacion en tiempo real en sistemas BCI (deteccion de somnolencia, atencion e interruptores binarios).</p>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--primary-navy); font-weight: 600;">Neurotecnologia Aplicada</div>
                    </div>
                </div>
            </div>
            <div class="speaker-notes-content" style="display:none;">
                <span class="time-guide">TIEMPO ASIGNADO: Minuto 8:30 a 10:00 (1:30 min)</span><br><br>
                <strong>Guion de Presentacion:</strong><br>
                "Para concluir esta exposicion, recapitulo los hallazgos esenciales del proyecto integrador.<br><br>
                Hemos comprobado experimentalmente la interdependencia de las tres unidades del curso. Sin el filtrado y la reduccion espacial de la <strong>Unidad 1</strong>, el ruido de fondo y la conduccion de volumen habrian degradado las caracteristicas de entrada. Sin la correcta definicion de atributos espectrales en ventanas de tiempo de la <strong>Unidad 2</strong>, el algoritmo K-Means no habria alcanzado un agrupamiento optimo. Y es gracias a los fundamentos biofisicos de la <strong>Unidad 3</strong> que podemos interpretar clinicamente el fenomeno observado.<br><br>
                Por su bajo coste computacional y alta confiabilidad, este pipeline sienta las bases para sistemas de monitoreo neurofisiologico en tiempo real, tales como interruptores pasivos en Interfaces Cerebro-Computadora o detectores de somnolencia en entornos criticos.<br><br>
                Quedo a entera disposicion del tribunal para responder sus preguntas. Muchas gracias por su atencion."
            </div>
        </section>
    </main>

    <!-- PANEL INFERIOR DE NOTAS DE ORADOR -->
    <div class="speaker-notes-panel" id="speakerPanel">
        <div class="notes-header">
            <span>NOTAS DE ORADOR (GUION VERBATIM PARA EXPOSICION DE 10 MINUTOS)</span>
            <button class="btn" style="padding: 2px 8px; font-size: 0.75rem; background: #334155; color: #ffffff; border: none;" onclick="toggleSpeakerNotes()">[Cerrar - N]</button>
        </div>
        <div class="notes-text" id="speakerNotesText"></div>
    </div>

    <footer>
        <div>UNIVERSIDAD | EVALUACION FINAL DE PROCESAMIENTO DE SENALES & MINERIA DE SERIES TEMPORALES</div>
        <div>Navegacion: <span class="kb-badge">[◀]</span> / <span class="kb-badge">[▶]</span> o <span class="kb-badge">[Espacio]</span> | Notas: <span class="kb-badge">[N]</span> | Pantalla Completa: <span class="kb-badge">[F]</span></div>
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
        
    print(f" -> Presentacion generada con exito: {os.path.abspath(output_path)}")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Descarga de archivos EDF
    f_open, f_closed = descargar_archivos_eeg(base_dir)
    
    # 2. Carga y Preprocesamiento
    data_open, data_closed, target_channels, sfreq = cargar_y_preprocesar_eeg(f_open, f_closed)
    
    # 3. PCA Espacial
    pc1_open, pc1_closed, pca_model, var_exp = aplicar_pca_espacial(data_open, data_closed)
    
    # 4. Espectro y STFT
    freqs, psd_open, psd_closed, f_stft, t_stft, Sxx, ratio_berger = analisis_espectral_y_tiempo_frecuencia(
        pc1_open, pc1_closed, sfreq
    )
    
    # 5. Minería de Series Temporales y Clustering
    clustering_res = mineria_y_clustering_kmeans(pc1_open, pc1_closed, sfreq, win_len_sec=2.0)
    
    # 6. Guardar Figuras
    saved_figs = generar_figuras(
        data_open, data_closed, target_channels, pc1_open, pc1_closed,
        freqs, psd_open, psd_closed, f_stft, t_stft, Sxx, clustering_res, sfreq, base_dir
    )
    
    # 7. Generar HTML Formal
    html_path = os.path.join(base_dir, "presentacion.html")
    generar_presentacion_html(var_exp, ratio_berger, clustering_res, html_path)
    
    print("\n" + "="*70)
    print(" [OK] EJECUCION COMPLETADA EXITOSAMENTE")
    print(f" -> Presentacion lista: {html_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
