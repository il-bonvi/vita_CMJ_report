import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tkinter import Tk, Label, Entry, Button, filedialog
from matplotlib.backends.backend_pdf import PdfPages

# ============================
# FUNZIONI DI CALCOLO
# ============================

def load_pedana(file):
    df = pd.read_csv(file, sep=",", header=None, comment="#", engine="python", skip_blank_lines=True, on_bad_lines="skip")
    df = df.iloc[:, :3]
    df.columns = ["time", "pedana_sinistra", "pedana_destra"]
    return df


def preprocess(df, offset_sx=0, offset_dx=0, soglia_contatto=3):
    df = df.copy()
    df["pedana_sinistra_cor"] = (df["pedana_sinistra"] - offset_sx).clip(lower=0)
    df["pedana_destra_cor"]   = (df["pedana_destra"] - offset_dx).clip(lower=0)

    # normalizza a 0 quando sotto soglia_contatto
    df["pedana_sinistra_cor"] = df["pedana_sinistra_cor"].where(df["pedana_sinistra_cor"]>soglia_contatto, 0)
    df["pedana_destra_cor"]   = df["pedana_destra_cor"].where(df["pedana_destra_cor"]>soglia_contatto, 0)

    df["forza_tot"] = df["pedana_sinistra_cor"] + df["pedana_destra_cor"]
    df['time_s'] = df['time'] / 1000  # ms -> s per calcoli interni
    return df


def detect_flight_phase(df, soglia=5, durata_min=0.5):
    df = df.copy()
    df['in_volo'] = False
    in_volo = df['forza_tot'] < soglia
    start_idx = None
    for i, val in enumerate(in_volo):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            if df['time_s'].iloc[i-1] - df['time_s'].iloc[start_idx] >= durata_min:
                df.loc[start_idx:i-1, 'in_volo'] = True
            start_idx = None
    if start_idx is not None and df['time_s'].iloc[-1] - df['time_s'].iloc[start_idx] >= durata_min:
        df.loc[start_idx:len(df)-1, 'in_volo'] = True
    return df


def compute_flight_times(df):
    intervals = []
    in_volo = df['in_volo'].values
    start = None
    for i, val in enumerate(in_volo):
        if val and start is None:
            start = i
        elif not val and start is not None:
            intervals.append((start, i-1))
            start = None
    if start is not None:
        intervals.append((start, len(df)-1))
    return [(df['time_s'].iloc[end] - df['time_s'].iloc[start]) for start,end in intervals]


def analyze_cmj_force(df, baseline=55, delta=5, soglia_volo=5, durata_min=0.5, massa=66, finestra_media=3):
    df = df.copy()
    df['forza_filt'] = df['forza_tot'].rolling(finestra_media, center=True, min_periods=1).mean()

    df = detect_flight_phase(df, soglia_volo, durata_min)

    # onset fase eccentrica
    onset_idx = df[df['forza_filt'] < baseline - delta].index
    idx_onset = onset_idx[0] if len(onset_idx)>0 else 0

    # fase spinta
    mask_spinta = (df.index > idx_onset) & (df['forza_filt'] > baseline + delta)
    idx_spinta = df.index[mask_spinta][0] if mask_spinta.any() else idx_onset

    # take-off
    mask_takeoff = (df.index > idx_spinta) & (df['forza_filt'] < soglia_volo)
    idx_takeoff = df.index[mask_takeoff][0] if mask_takeoff.any() else df.index[-1]

    tempo_ecc = df.loc[idx_spinta, 'time_s'] - df.loc[idx_onset, 'time_s']
    tempo_spinta = df.loc[idx_takeoff, 'time_s'] - df.loc[idx_spinta, 'time_s']

    flight_times = compute_flight_times(df)
    tempo_volo = flight_times[0] if flight_times else 0

    Fmax = df.loc[idx_spinta:idx_takeoff, 'forza_filt'].max()
    impulso = np.trapz(df.loc[idx_spinta:idx_takeoff, 'forza_filt'] - baseline,
                        df.loc[idx_spinta:idx_takeoff, 'time_s'])
    Pmax = Fmax * (impulso / massa) / tempo_spinta if tempo_spinta>0 else 0

    h = (9.81 * tempo_volo**2) / 8

    df['asimmetria_%'] = 100 * (df['pedana_destra_cor'] - df['pedana_sinistra_cor']) / (
        0.5*(df['pedana_destra_cor'] + df['pedana_sinistra_cor']) + 1e-6)
    df['asimmetria_%'] = df['asimmetria_%'].where(df['forza_tot']>3,0)
    asim_media = df[df['forza_tot']>3]['asimmetria_%'].mean()

    return {
        'tempo_eccentrico': tempo_ecc,
        'tempo_spinta': tempo_spinta,
        'tempo_volo': tempo_volo,
        'Fmax': Fmax,
        'impulso': impulso,
        'Pmax': Pmax,
        'altezza': h,
        'asimmetria_media': asim_media,
        'df': df
    }

# ============================
# FUNZIONI GUI
# ============================

def run_analysis():
    file_path = filedialog.askopenfilename(filetypes=[("Text files","*.txt")])
    if not file_path: return

    try:
        offset_sx_val = float(offset_sx_entry.get())
        offset_dx_val = float(offset_dx_entry.get())
        soglia_volo_val = float(soglia_entry.get())
        durata_min_val = float(durata_entry.get())
    except ValueError:
        print("Inserisci valori numerici validi!")
        return

    df = load_pedana(file_path)
    df = preprocess(df, offset_sx_val, offset_dx_val)
    cmj = analyze_cmj_force(df, soglia_volo=soglia_volo_val, durata_min=durata_min_val)

    print(f"Tempo eccentrico: {cmj['tempo_eccentrico']:.3f} s")
    print(f"Tempo spinta: {cmj['tempo_spinta']:.3f} s")
    print(f"Tempo volo: {cmj['tempo_volo']:.3f} s")
    print(f"Fmax: {cmj['Fmax']:.1f} N")
    print(f"Impulso: {cmj['impulso']:.2f} N·s")
    print(f"Pmax: {cmj['Pmax']:.1f} W")
    print(f"Altezza salto: {cmj['altezza']:.3f} m")
    print(f"Asimmetria media: {cmj['asimmetria_media']:.2f} %")

    pdf_file = file_path.replace('.txt','_report.pdf')
    with PdfPages(pdf_file) as pdf:
        df_plot = cmj['df']

        # Forza Totale
        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['forza_tot'], label='Forza Totale')
        plt.axhline(soglia_volo_val, color='red', linestyle='--', label='Soglia volo')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.title('Forza Totale e volo')
        plt.legend()
        pdf.savefig(); plt.close()

        # Pedane SX/DX
        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['pedana_sinistra_cor'], label='SX')
        plt.plot(df_plot['time'], df_plot['pedana_destra_cor'], label='DX')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.title('Forza Pedane')
        plt.legend()
        pdf.savefig(); plt.close()

        # Asimmetria
        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['asimmetria_%'], label='Asimmetria (%)')
        plt.axhline(0, linestyle='--', color='black')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Asimmetria (%)')
        plt.title('Asimmetria Destra-Sinistra')
        plt.legend()
        pdf.savefig(); plt.close()

        # Tabella parametri
        fig, ax = plt.subplots(figsize=(10,6))
        ax.axis('off')
        ax.set_title('Parametri CMJ', fontsize=18, fontweight='bold')
        cmj_data = [
            ['Parametro', 'Valore'],
            ['Tempo eccentrico (s)', f"{cmj['tempo_eccentrico']:.3f}"],
            ['Tempo spinta (s)', f"{cmj['tempo_spinta']:.3f}"],
            ['Tempo volo (s)', f"{cmj['tempo_volo']:.3f}"],
            ['Fmax (N)', f"{cmj['Fmax']:.1f}"],
            ['Impulso (N·s)', f"{cmj['impulso']:.2f}"],
            ['Pmax (W)', f"{cmj['Pmax']:.1f}"],
            ['Altezza salto (m)', f"{cmj['altezza']:.3f}"],
            ['Asimmetria media (%)', f"{cmj['asimmetria_media']:.2f}"]
        ]
        table = ax.table(cellText=cmj_data, loc='center', cellLoc='center', colWidths=[0.5,0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.5, 2)
        pdf.savefig(); plt.close()

    print(f"Report PDF generato: {pdf_file}")

# ============================
# CREAZIONE GUI
# ============================
root = Tk()
root.title("CMJ Analysis")

Label(root, text="Offset pedana SX").grid(row=0, column=0)
offset_sx_entry = Entry(root); offset_sx_entry.insert(0,"50"); offset_sx_entry.grid(row=0, column=1)

Label(root, text="Offset pedana DX").grid(row=1, column=0)
offset_dx_entry = Entry(root); offset_dx_entry.insert(0,"40"); offset_dx_entry.grid(row=1, column=1)

Label(root, text="Soglia volo (N)").grid(row=2, column=0)
soglia_entry = Entry(root); soglia_entry.insert(0,"5"); soglia_entry.grid(row=2, column=1)

Label(root, text="Durata minima volo (s)").grid(row=3, column=0)
durata_entry = Entry(root); durata_entry.insert(0,"0.5"); durata_entry.grid(row=3, column=1)

Button(root, text="Seleziona file e calcola", command=run_analysis).grid(row=4, column=0, columnspan=2, pady=10)
root.mainloop()