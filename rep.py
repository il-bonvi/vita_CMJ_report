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

def analyze_cmj_force(df, soglia_volo=5, durata_min=0.5, massa=66, finestra_media=3):
    df = df.copy()
    df['forza_filt'] = df['forza_tot'].rolling(finestra_media, center=True, min_periods=1).mean()
    df = detect_flight_phase(df, soglia_volo, durata_min)

    # Take-off: primo indice in volo
    takeoff_idx = df.index[df['in_volo']].min()
    takeoff_time = df.loc[takeoff_idx, 'time_s'] if not np.isnan(takeoff_idx) else df['time_s'].iloc[-1]

    # Landing: ultimo indice in volo
    landing_idx = df.index[df['in_volo']].max()
    landing_time = df.loc[landing_idx, 'time_s'] if not np.isnan(landing_idx) else df['time_s'].iloc[-1]

    # Tempo di volo
    flight_time = landing_time - takeoff_time

    # Picco di forza totale **prima del take-off**
    df_pre_takeoff = df[df.index <= takeoff_idx]
    Fmax = df_pre_takeoff['forza_filt'].max()
    idx_peak = df_pre_takeoff['forza_filt'].idxmax()
    peak_time = df_pre_takeoff.loc[idx_peak, 'time_s']

    return {
        'Fmax': Fmax,
        'peak_time': peak_time,
        'takeoff_time': takeoff_time,
        'landing_time': landing_time,
        'flight_time': flight_time,
        'df': df,
        'massa': massa
    }

# ============================
# FUNZIONI GUI
# ============================

def update_plots(cmj, soglia_volo_val):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    df_plot = cmj['df']

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # --- Forza Totale ---
    axes[0].plot(df_plot['time'], df_plot['forza_tot'], label='Forza Totale')
    axes[0].axhline(soglia_volo_val, color='red', linestyle='--', label='Soglia volo')

    # Linee Take-off e Landing
    axes[0].axvline(cmj['takeoff_time']*1000, color='green', linestyle='--', label='Take-off')
    axes[0].axvline(cmj['landing_time']*1000, color='orange', linestyle='--', label='Landing')

    # Punto Fmax
    axes[0].scatter(cmj['peak_time']*1000, cmj['Fmax'], color='red', s=60, zorder=5, label='Fmax')

    axes[0].set_xlabel('Tempo (ms)')
    axes[0].set_ylabel('Forza (N)')
    axes[0].set_title('Forza Totale e volo')
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    # --- Forza Pedane ---
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

    cmj_global = cmj
    file_global = file_path
    soglia_volo_global = soglia_volo_val

    # Preview GUI
    preview_text.delete("1.0", "end")
    preview_text.insert("end", f"File: {os.path.basename(file_path)}\n\n")
    preview_text.insert("end", f"Fmax (N): {cmj['Fmax']:.0f}\n")
    preview_text.insert("end", f"Tempo picco forza (s): {cmj['peak_time']:.3f}\n")
    preview_text.insert("end", f"Take-off (s): {cmj['takeoff_time']:.3f}\n")
    preview_text.insert("end", f"Landing (s): {cmj['landing_time']:.3f}\n")
    preview_text.insert("end", f"Tempo di volo (s): {cmj['flight_time']:.3f}\n")
    preview_text.insert("end", f"Massa soggetto (kg): {cmj['massa']:.1f}\n")

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

    # ================= PDF
    with PdfPages(pdf_file) as pdf:
        # --- Forza Totale ---
        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['forza_tot'], label='Forza Totale')
        plt.axhline(soglia_volo_global, color='red', linestyle='--', label='Soglia volo')

        # Linee take-off e landing
        plt.axvline(cmj_global['takeoff_time']*1000, color='green', linestyle='--', label='Take-off')
        plt.axvline(cmj_global['landing_time']*1000, color='orange', linestyle='--', label='Landing')

        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.title('Forza Totale e volo')
        plt.legend()
        plt.ylim(bottom=0)
        pdf.savefig(); plt.close()

        # --- Pedane ---
        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['pedana_sinistra_cor'], label='SX')
        plt.plot(df_plot['time'], df_plot['pedana_destra_cor'], label='DX')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.title('Forza Pedane')
        plt.legend()
        plt.ylim(bottom=0)
        pdf.savefig(); plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(df_plot['time'], df_plot['pedana_sinistra_cor'], label='SX')
        plt.plot(df_plot['time'], df_plot['pedana_destra_cor'], label='DX')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.title('Forza Pedane')
        plt.legend()
        plt.ylim(bottom=0)
        pdf.savefig(); plt.close()

        fig, ax = plt.subplots(figsize=(10,6))
        ax.axis('off')
        ax.set_title('Parametri CMJ', fontsize=18, fontweight='bold')
        cmj_data = [
            ['Fmax (N)', f"{cmj_global['Fmax']:.0f}"],
            ['Tempo picco forza (s)', f"{cmj_global['peak_time']:.3f}"],
            ['Take-off (s)', f"{cmj_global['takeoff_time']:.3f}"],
            ['Landing (s)', f"{cmj_global['landing_time']:.3f}"],
            ['Tempo di volo (s)', f"{cmj_global['flight_time']:.3f}"],
            ['Massa soggetto (kg)', f"{cmj_global['massa']:.1f}"]
        ]
        table = ax.table(cellText=cmj_data, loc='center', cellLoc='center', colWidths=[0.5,0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.5,2)
        pdf.savefig(); plt.close()

    # ================= CSV
    df_csv = pd.DataFrame({
        'Parametro': ['Fmax (N)','Tempo picco forza (s)','Take-off (s)','Landing (s)','Tempo di volo (s)','Massa soggetto (kg)'],
        'Valore': [f"{cmj_global['Fmax']:.0f}", f"{cmj_global['peak_time']:.3f}",
                  f"{cmj_global['takeoff_time']:.3f}", f"{cmj_global['landing_time']:.3f}",
                  f"{cmj_global['flight_time']:.3f}", f"{cmj_global['massa']:.1f}"]
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
durata_entry = Entry(root); durata_entry.insert(0,"0.2"); durata_entry.grid(row=3, column=1)

Label(root, text="Peso soggetto (kg)").grid(row=4, column=0)
massa_entry = Entry(root); massa_entry.insert(0,"75"); massa_entry.grid(row=4, column=1)

Button(root, text="Seleziona file e calcola", command=run_analysis).grid(row=5, column=0, pady=5)
Button(root, text="Esporta PDF/CSV", command=export_results).grid(row=5, column=1, pady=5)

preview_text = Text(root, height=10, width=50)
preview_text.grid(row=6, column=0, columnspan=2, pady=5)

plot_frame = Frame(root)
plot_frame.grid(row=7, column=0, columnspan=2, pady=5)

root.mainloop()
