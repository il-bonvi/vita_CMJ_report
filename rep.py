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


def preprocess(df, offset_sx=0, offset_dx=0):
    df = df.copy()
    df["pedana_sinistra_cor"] = df["pedana_sinistra"] - offset_sx
    df["pedana_destra_cor"] = df["pedana_destra"] - offset_dx
    df["forza_tot"] = df["pedana_sinistra_cor"] + df["pedana_destra_cor"]
    df["forza_tot"] = df["forza_tot"].clip(lower=0)
    return df


def detect_flight_phase(df, soglia=5, durata_min=0.5):
    df = df.copy()
    df['in_volo'] = False
    time_sec = df['time'] / 1000  # assume ms input
    in_volo = (df['forza_tot'] < soglia)
    # verifica durata minima
    start_idx = None
    for i, val in enumerate(in_volo):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            if time_sec[i-1] - time_sec[start_idx] >= durata_min:
                df.loc[start_idx:i-1, 'in_volo'] = True
            start_idx = None
    if start_idx is not None and time_sec[len(df)-1] - time_sec[start_idx] >= durata_min:
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
    times = [(df['time'].iloc[end] - df['time'].iloc[start])/1000 for start,end in intervals]
    return times


def jump_height_from_flight(time_volo, g=9.81):
    return g * time_volo**2 / 8

# ============================
# FUNZIONI GUI
# ============================

def run_analysis():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return

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
    df = detect_flight_phase(df, soglia_volo_val, durata_min_val)
    flight_times = compute_flight_times(df)

    if flight_times:
        h = jump_height_from_flight(flight_times[0])
    else:
        h = 0

    print(f"Tempi di volo (s): {flight_times}")
    print(f"Altezza salto (m): {h:.3f}")

    # Grafico di esempio
    plt.figure(figsize=(8,5))
    plt.plot(df['time'], df['forza_tot'], label='Forza Totale')
    plt.axhline(soglia_volo_val, color='red', linestyle='--', label='Soglia volo')
    plt.xlabel('Tempo (ms)')
    plt.ylabel('Forza (N)')
    plt.legend()
    plt.title('Forza Totale e volo')
    plt.show()

    # PDF
    pdf_file = file_path.replace('.txt','_report.pdf')
    with PdfPages(pdf_file) as pdf:
        plt.figure(figsize=(8,5))
        plt.plot(df['time'], df['forza_tot'], label='Forza Totale')
        plt.axhline(soglia_volo_val, color='red', linestyle='--', label='Soglia volo')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Forza (N)')
        plt.legend()
        plt.title('Forza Totale e volo')
        pdf.savefig()
        plt.close()

    print(f"Report PDF generato: {pdf_file}")


# ============================
# CREAZIONE GUI
# ============================
root = Tk()
root.title("CMJ Analysis")

Label(root, text="Offset pedana SX").grid(row=0, column=0)
offset_sx_entry = Entry(root)
offset_sx_entry.insert(0,"41")
offset_sx_entry.grid(row=0, column=1)

Label(root, text="Offset pedana DX").grid(row=1, column=0)
offset_dx_entry = Entry(root)
offset_dx_entry.insert(0,"50")
offset_dx_entry.grid(row=1, column=1)

Label(root, text="Soglia volo (N)").grid(row=2, column=0)
soglia_entry = Entry(root)
soglia_entry.insert(0,"5")
soglia_entry.grid(row=2, column=1)

Label(root, text="Durata minima volo (s)").grid(row=3, column=0)
durata_entry = Entry(root)
durata_entry.insert(0,"0.5")
durata_entry.grid(row=3, column=1)

Button(root, text="Seleziona file e calcola", command=run_analysis).grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
