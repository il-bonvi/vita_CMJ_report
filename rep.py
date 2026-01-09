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
eccentric_start_idx = None
concentric_start_idx = None
massa_global = None
g = 9.81  # gravità

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

    # Take-off e landing
    volo_idx = df[df['in_volo']].index
    takeoff_idx = volo_idx[0] if len(volo_idx) > 0 else None
    landing_idx = volo_idx[-1] if len(volo_idx) > 0 else None
    takeoff_time = df.loc[takeoff_idx, 'time_s'] if takeoff_idx is not None else None
    landing_time = df.loc[landing_idx, 'time_s'] if landing_idx is not None else None

    # Picco di forza totale (prima del take-off)
    if takeoff_idx is not None:
        Fmax = df['forza_filt'].iloc[:takeoff_idx].max()
        idx_peak = df['forza_filt'].iloc[:takeoff_idx].idxmax()
        peak_time = df.loc[idx_peak, 'time_s']
    else:
        Fmax, peak_time = df['forza_filt'].max(), df['time_s'][df['forza_filt'].idxmax()]

    return {
        'Fmax': Fmax,
        'peak_time': peak_time,
        'df': df,
        'massa': massa,
        'takeoff_idx': takeoff_idx,
        'landing_idx': landing_idx,
        'takeoff_time': takeoff_time,
        'landing_time': landing_time
    }

# ============================
# CALCOLO POTENZA CONCENTRICA
# ============================

def compute_concentric_power(df, concentric_start_idx, takeoff_idx, massa):
    if concentric_start_idx is None or takeoff_idx is None:
        return None, None

    F_conc = df['forza_tot'].iloc[concentric_start_idx:takeoff_idx+1].values
    time_conc = df['time_s'].iloc[concentric_start_idx:takeoff_idx+1].values /1000

    if len(F_conc) <= 1:
        return None, None

    acc = (F_conc - massa*g) / massa
    dt = np.diff(time_conc, prepend=time_conc[0])
    vel = np.cumsum(acc * dt)
    vel = np.maximum(vel, 0)
    pot = F_conc * vel
    pot_media = np.trapz(pot, time_conc) / (time_conc[-1]-time_conc[0])
    pot_max = np.max(pot)

    return pot_media, pot_max

# ============================
# FUNZIONI GUI
# ============================

def update_plots(cmj, soglia_volo_val):
    global eccentric_start_idx, concentric_start_idx
    for widget in plot_frame.winfo_children():
        widget.destroy()
    df_plot = cmj['df']

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].plot(df_plot['time'], df_plot['forza_tot'], label='Forza Totale')
    axes[0].axhline(soglia_volo_val, color='red', linestyle='--', label='Soglia volo')
    if cmj['takeoff_time'] is not None:
        axes[0].axvline(cmj['takeoff_time']*1000, color='green', linestyle='--', label='Take-off')
    if cmj['landing_time'] is not None:
        axes[0].axvline(cmj['landing_time']*1000, color='orange', linestyle='--', label='Landing')
    axes[0].scatter(cmj['peak_time']*1000, cmj['Fmax'], color='red', s=60, zorder=5, label='Fmax')
    if eccentric_start_idx is not None:
        axes[0].axvline(df_plot.iloc[eccentric_start_idx]['time'], color='purple', linestyle='--', label='Inizio eccentrica')
    if concentric_start_idx is not None:
        axes[0].axvline(df_plot.iloc[concentric_start_idx]['time'], color='brown', linestyle='--', label='Inizio concentrica')
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

# ============================
# SELEZIONE PUNTI
# ============================

def pick_point(title, cmj):
    df = cmj['df']
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['time'], df['forza_tot'], label='Forza Totale')
    ax.set_title(title)
    ax.set_xlabel('Tempo (ms)')
    ax.set_ylabel('Forza (N)')
    plt.ylim(bottom=0)
    point = []

    annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(event):
        if event.xdata is None or event.ydata is None: 
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        x = event.xdata
        idx = (np.abs(df['time'] - x)).idxmin()
        y = df['forza_tot'].iloc[idx]
        annot.xy = (df['time'].iloc[idx], y)
        annot.set_text(f"Time: {df['time'].iloc[idx]:.2f} ms\nForce: {y:.1f} N")
        annot.set_visible(True)
        fig.canvas.draw_idle()

    def onclick(event):
        if event.xdata is not None:
            idx = (np.abs(df['time'] - event.xdata)).idxmin()
            point.append(idx)
            plt.close('all')  # <-- chiude tutte le figure

    fig.canvas.mpl_connect("motion_notify_event", update_annot)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return point[0] if point else None

def select_eccentric():
    global eccentric_start_idx, cmj_global
    idx = pick_point("Clicca per selezionare inizio eccentrica", cmj_global)
    if idx is not None:
        eccentric_start_idx = idx
        update_plots(cmj_global, soglia_volo_global)

def select_concentric():
    global concentric_start_idx, cmj_global
    idx = pick_point("Clicca per selezionare inizio concentrica", cmj_global)
    if idx is not None:
        concentric_start_idx = idx
        update_plots(cmj_global, soglia_volo_global)

# ============================
# RUN ANALYSIS
# ============================

def run_analysis():
    global cmj_global, file_global, soglia_volo_global, massa_global
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
    massa_global = massa_val

    preview_text.delete("1.0", "end")
    preview_text.insert("end", f"File: {os.path.basename(file_path)}\n")
    preview_text.insert("end", f"Fmax (N): {cmj['Fmax']:.0f}\n")
    preview_text.insert("end", f"Tempo picco forza (s): {cmj['peak_time']:.3f}\n")
    preview_text.insert("end", f"Take-off (s): {cmj['takeoff_time']:.3f}\n")
    preview_text.insert("end", f"Landing (s): {cmj['landing_time']:.3f}\n")
    preview_text.insert("end", f"Massa soggetto (kg): {massa_val:.1f}\n")

    if concentric_start_idx is not None and cmj_global['takeoff_idx'] is not None:
        pot_media, pot_max = compute_concentric_power(cmj_global['df'],
                                                      concentric_start_idx,
                                                      cmj_global['takeoff_idx'],
                                                      massa_global)
        if pot_media is not None:
            preview_text.insert("end", f"Potenza media concentrica (W): {pot_media:.1f}\n")
            preview_text.insert("end", f"Potenza massima concentrica (W): {pot_max:.1f}\n")

    update_plots(cmj, soglia_volo_val)

# ============================
# EXPORT PDF/CSV
# ============================

def export_results():
    global cmj_global, file_global, soglia_volo_global, eccentric_start_idx, concentric_start_idx, massa_global
    if cmj_global is None:
        print("Nessun dato da esportare! Prima esegui un'analisi.")
        return

    df = cmj_global['df']
    base_name = os.path.splitext(os.path.basename(file_global))[0]

    t_ecc = t_conc = pot_media = pot_max = None
    if eccentric_start_idx is not None and concentric_start_idx is not None:
        t_ecc = df['time_s'].iloc[concentric_start_idx] - df['time_s'].iloc[eccentric_start_idx]
    if concentric_start_idx is not None and cmj_global['takeoff_idx'] is not None:
        t_conc = df['time_s'].iloc[cmj_global['takeoff_idx']] - df['time_s'].iloc[concentric_start_idx]
        pot_media, pot_max = compute_concentric_power(df, concentric_start_idx, cmj_global['takeoff_idx'], massa_global)

    pdf_file = filedialog.asksaveasfilename(defaultextension=".pdf",
                                            filetypes=[("PDF files","*.pdf")],
                                            initialfile=f"report_{base_name}_.pdf")
    csv_file = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files","*.csv")],
                                            initialfile=f"report_{base_name}_.csv")
    if not pdf_file or not csv_file: return

    with PdfPages(pdf_file) as pdf:
        # Plot forza totale
        plt.figure(figsize=(8,5))
        plt.plot(df['time'], df['forza_tot'], label='Forza Totale')
        plt.axhline(soglia_volo_global, color='red', linestyle='--', label='Soglia volo')
        if cmj_global['takeoff_time'] is not None:
            plt.axvline(cmj_global['takeoff_time']*1000, color='green', linestyle='--', label='Take-off')
        if cmj_global['landing_time'] is not None:
            plt.axvline(cmj_global['landing_time']*1000, color='orange', linestyle='--', label='Landing')
        plt.scatter(cmj_global['peak_time']*1000, cmj_global['Fmax'], color='red', s=60, zorder=5, label='Fmax')
        if eccentric_start_idx is not None:
            plt.axvline(df.iloc[eccentric_start_idx]['time'], color='purple', linestyle='--', label='Inizio eccentrica')
        if concentric_start_idx is not None:
            plt.axvline(df.iloc[concentric_start_idx]['time'], color='brown', linestyle='--', label='Inizio concentrica')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.legend()
        plt.ylim(bottom=0)
        pdf.savefig(); plt.close()

        # Plot pedane
        plt.figure(figsize=(8,5))
        plt.plot(df['time'], df['pedana_sinistra_cor'], label='SX')
        plt.plot(df['time'], df['pedana_destra_cor'], label='DX')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.legend()
        plt.ylim(bottom=0)
        pdf.savefig(); plt.close()

        # Tabella
        fig, ax = plt.subplots(figsize=(10,6))
        ax.axis('off')
        ax.set_title('Parametri CMJ', fontsize=18, fontweight='bold')
        cmj_data = [
            ['Fmax (N)', f"{cmj_global['Fmax']:.0f}"],
            ['Tempo picco forza (s)', f"{cmj_global['peak_time']:.3f}"],
            ['t eccentrica (s)', f"{t_ecc:.3f}" if t_ecc is not None else "-"],
            ['t concentrica (s)', f"{t_conc:.3f}" if t_conc is not None else "-"],
            ['Massa soggetto (kg)', f"{massa_global:.1f}"]
        ]
        table = ax.table(cellText=cmj_data, loc='center', cellLoc='center', colWidths=[0.5,0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.5,2)
        pdf.savefig(); plt.close()

    df_csv = pd.DataFrame({'Parametro':[r[0] for r in cmj_data], 'Valore':[r[1] for r in cmj_data]})
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

preview_text = Text(root, height=14, width=70)
preview_text.grid(row=6, column=0, columnspan=2, pady=5)

plot_frame = Frame(root)
plot_frame.grid(row=7, column=0, columnspan=2, pady=5)

Button(root, text="Seleziona inizio eccentrica", command=select_eccentric).grid(row=8, column=0, pady=5)
Button(root, text="Seleziona inizio concentrica", command=select_concentric).grid(row=8, column=1, pady=5)

root.mainloop()
