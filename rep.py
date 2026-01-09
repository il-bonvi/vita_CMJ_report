import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, filedialog, Text, Frame
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# ============================
# VARIABILI GLOBALI
# ============================
cmj_global = None
file_global = None
soglia_volo_global = 5

# ============================
# FUNZIONI DI CALCOLO
# ============================

def load_pedana(file):
    df = pd.read_csv(file, sep=",", header=None, comment="#", engine="python",
                     skip_blank_lines=True, on_bad_lines="skip")
    df = df.iloc[:, :3]
    df.columns = ["time", "pedana_sinistra", "pedana_destra"]
    return df

def preprocess(df, offset_sx=0, offset_dx=0, soglia_contatto=3):
    df = df.copy()
    df["pedana_sinistra_cor"] = (df["pedana_sinistra"] - offset_sx).clip(lower=0)
    df["pedana_destra_cor"]   = (df["pedana_destra"] - offset_dx).clip(lower=0)
    df["pedana_sinistra_cor"] = df["pedana_sinistra_cor"].where(df["pedana_sinistra_cor"]>soglia_contatto, 0)
    df["pedana_destra_cor"]   = df["pedana_destra_cor"].where(df["pedana_destra_cor"]>soglia_contatto, 0)
    df["forza_tot"] = df["pedana_sinistra_cor"] + df["pedana_destra_cor"]
    df['time_s'] = df['time'] / 1000  # ms -> s
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

    onset_idx = df[df['forza_filt'] < baseline - delta].index
    idx_onset = onset_idx[0] if len(onset_idx) > 0 else 0

    mask_spinta = (df.index > idx_onset) & (df['forza_filt'] > baseline + delta)
    idx_spinta = df.index[mask_spinta][0] if mask_spinta.any() else idx_onset

    mask_takeoff = (df.index > idx_spinta) & (df['forza_filt'] < soglia_volo)
    idx_takeoff = df.index[mask_takeoff][0] if mask_takeoff.any() else df.index[-1]

    tempo_ecc = df.loc[idx_spinta, 'time_s'] - df.loc[idx_onset, 'time_s']
    tempo_spinta = df.loc[idx_takeoff, 'time_s'] - df.loc[idx_spinta, 'time_s']

    flight_times = compute_flight_times(df)
    tempo_volo = flight_times[0] if flight_times else 0

    Fmax = df.loc[idx_spinta:idx_takeoff, 'forza_filt'].max()
    impulso = np.trapz(df.loc[idx_spinta:idx_takeoff, 'forza_filt'] - baseline,
                        df.loc[idx_spinta:idx_takeoff, 'time_s'])
    Pmax = Fmax * (impulso / massa) / tempo_spinta if tempo_spinta > 0 else 0
    h = (9.81 * tempo_volo**2) / 8

    # Calcolo asimmetria media concentrica
    df_conc = df.loc[idx_spinta:idx_takeoff].copy()
    df_conc['asimmetria'] = np.abs(df_conc['pedana_sinistra_cor'] - df_conc['pedana_destra_cor']) / \
                            (df_conc['pedana_sinistra_cor'] + df_conc['pedana_destra_cor'] + 1e-6) * 100
    asimmetria_media = df_conc['asimmetria'].mean()

    return {
        'tempo_eccentrico': tempo_ecc,
        'tempo_spinta': tempo_spinta,
        'tempo_volo': tempo_volo,
        'Fmax': Fmax,
        'impulso': impulso,
        'Pmax': Pmax,
        'altezza': h,
        'df': df,
        'df_conc': df_conc,
        'asimmetria_media': asimmetria_media
    }

# ============================
# FUNZIONI GUI
# ============================

def update_plots(cmj, soglia_volo_val):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    df_plot = cmj['df']

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    axes[0].plot(df_plot['time'], df_plot['forza_tot'], label='Forza Totale')
    axes[0].axhline(soglia_volo_val, color='red', linestyle='--', label='Soglia volo')
    axes[0].set_xlabel('Tempo (ms)')
    axes[0].set_ylabel('Forza (N)')
    axes[0].set_title('Forza Totale e volo')
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    axes[1].plot(df_plot['time'], df_plot['pedana_sinistra_cor'], label='SX')
    axes[1].plot(df_plot['time'], df_plot['pedana_destra_cor'], label='DX')
    axes[1].set_xlabel('Tempo (ms)')
    axes[1].set_ylabel('Forza (N)')
    axes[1].set_title('Forza Pedane')
    axes[1].set_ylim(bottom=0)
    axes[1].legend()

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def run_analysis():
    global cmj_global, file_global, soglia_volo_global
    file_path = filedialog.askopenfilename(filetypes=[("Text/CSV files","*.txt;*.csv")])
    if not file_path: return
    try:
        offset_sx_val = float(offset_sx_entry.get())
        offset_dx_val = float(offset_dx_entry.get())
        soglia_volo_val = float(soglia_entry.get())
        durata_min_val = float(durata_entry.get())
        massa_val = float(massa_entry.get())
    except ValueError:
        print("Inserisci valori numerici validi!")
        return

    df = load_pedana(file_path)
    df = preprocess(df, offset_sx_val, offset_dx_val)
    cmj = analyze_cmj_force(df, soglia_volo=soglia_volo_val, durata_min=durata_min_val, massa=massa_val)
    cmj['massa'] = massa_val

    # Salva nei globali
    cmj_global = cmj
    file_global = file_path
    soglia_volo_global = soglia_volo_val

    # Preview GUI
    preview_text.delete("1.0", "end")
    preview_text.insert("end", f"File: {os.path.basename(file_path)}\n\n")
    preview_text.insert("end", f"Tempo eccentrico (s): {cmj['tempo_eccentrico']:.3f}\n")
    preview_text.insert("end", f"Tempo spinta (s): {cmj['tempo_spinta']:.3f}\n")
    preview_text.insert("end", f"Tempo volo (s): {cmj['tempo_volo']:.3f}\n")
    preview_text.insert("end", f"Fmax (N): {cmj['Fmax']:.0f}\n")
    preview_text.insert("end", f"Impulso (N·s): {cmj['impulso']:.1f}\n")
    preview_text.insert("end", f"Pmax (W): {cmj['Pmax']:.0f}\n")
    preview_text.insert("end", f"Altezza salto (m): {cmj['altezza']:.3f}\n")
    preview_text.insert("end", f"Massa soggetto (kg): {cmj['massa']:.1f}\n")
    preview_text.insert("end", f"Asimmetria media concentrica (%): {cmj['asimmetria_media']:.2f}\n")

    update_plots(cmj, soglia_volo_val)

def export_results():
    global cmj_global, file_global, soglia_volo_global
    if cmj_global is None:
        print("Nessun dato da esportare! Prima esegui un'analisi.")
        return

    base_name = os.path.splitext(os.path.basename(file_global))[0]

    pdf_file = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files","*.pdf")],
        initialfile=f"report_{base_name}_.pdf"
    )
    csv_file = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files","*.csv")],
        initialfile=f"report_{base_name}_.csv"
    )
    if not pdf_file or not csv_file: return

    df_plot = cmj_global['df']
    df_conc = cmj_global['df_conc']

    # ================= PDF
    with PdfPages(pdf_file) as pdf:
        # 1) Forza totale e soglia volo
        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['forza_tot'], label='Forza Totale')
        plt.axhline(soglia_volo_global, color='red', linestyle='--', label='Soglia volo')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.title('Forza Totale e volo')
        plt.legend()
        plt.ylim(bottom=0)
        pdf.savefig(); plt.close()

        # 2) Forza pedane SX/DX
        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['pedana_sinistra_cor'], label='SX')
        plt.plot(df_plot['time'], df_plot['pedana_destra_cor'], label='DX')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.title('Forza Pedane')
        plt.legend()
        plt.ylim(bottom=0)
        pdf.savefig(); plt.close()

        # 3) Asimmetria media concentrica
        plt.figure(figsize=(8,5))
        plt.plot(df_conc['time_s'], df_conc['asimmetria'], color='purple', label='Asimmetria (%)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Asimmetria (%)')
        plt.title('Asimmetria Media Concentrica')
        plt.ylim(0, max(df_conc['asimmetria'].max()*1.1, 10))
        plt.legend()
        pdf.savefig(); plt.close()

        # 4) Tabella parametri CMJ
        fig, ax = plt.subplots(figsize=(10,6))
        ax.axis('off')
        ax.set_title('Parametri CMJ', fontsize=18, fontweight='bold')
        cmj_data = [
            ['Tempo eccentrico (s)', f"{cmj_global['tempo_eccentrico']:.3f}"],
            ['Tempo spinta (s)', f"{cmj_global['tempo_spinta']:.3f}"],
            ['Tempo volo (s)', f"{cmj_global['tempo_volo']:.3f}"],
            ['Fmax (N)', f"{cmj_global['Fmax']:.0f}"],
            ['Impulso (N·s)', f"{cmj_global['impulso']:.1f}"],
            ['Pmax (W)', f"{cmj_global['Pmax']:.0f}"],
            ['Altezza salto (m)', f"{cmj_global['altezza']:.3f}"],
            ['Asimmetria media concentrica (%)', f"{cmj_global['asimmetria_media']:.2f}"],
            ['Massa soggetto (kg)', f"{cmj_global['massa']:.1f}"]
        ]
        table = ax.table(cellText=cmj_data, loc='center', cellLoc='center', colWidths=[0.5,0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.5,2)
        pdf.savefig(); plt.close()

    # ================= CSV
    df_csv = pd.DataFrame({
        'Parametro': ['Tempo eccentrico (s)','Tempo spinta (s)','Tempo volo (s)','Fmax (N)',
                      'Impulso (N·s)','Pmax (W)','Altezza salto (m)','Massa soggetto (kg)','Asimmetria media concentrica (%)'],
        'Valore': [f"{cmj_global['tempo_eccentrico']:.3f}", f"{cmj_global['tempo_spinta']:.3f}", f"{cmj_global['tempo_volo']:.3f}",
                   f"{cmj_global['Fmax']:.0f}", f"{cmj_global['impulso']:.1f}", f"{cmj_global['Pmax']:.0f}",
                   f"{cmj_global['altezza']:.3f}", f"{cmj_global['massa']:.1f}", f"{cmj_global['asimmetria_media']:.2f}"]
    })
    df_csv.to_csv(csv_file, index=False)
    print(f"Report PDF generato: {pdf_file}")
    print(f"CSV generato: {csv_file}")

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

Label(root, text="Peso soggetto (kg)").grid(row=4, column=0)
massa_entry = Entry(root); massa_entry.insert(0,"66"); massa_entry.grid(row=4, column=1)

Button(root, text="Seleziona file e calcola", command=run_analysis).grid(row=5, column=0, pady=5)
Button(root, text="Esporta PDF/CSV", command=export_results).grid(row=5, column=1, pady=5)

preview_text = Text(root, height=12, width=50)
preview_text.grid(row=6, column=0, columnspan=2, pady=5)

plot_frame = Frame(root)
plot_frame.grid(row=7, column=0, columnspan=2, pady=5)

root.mainloop()