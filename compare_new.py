import pandas as pd
import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Frame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

# ============================
# LOGICA DATI
# ============================
pre_data = None
post_data = None

def load_csv(pre_or_post):
    global pre_data, post_data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if not file_path: return
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    if pre_or_post == "pre":
        pre_data = df
        pre_label.config(text=f"Pre: {os.path.basename(file_path)}")
    else:
        post_data = df
        post_label.config(text=f"Post: {os.path.basename(file_path)}")
    update_preview()

def get_merged_df():
    if pre_data is None or post_data is None: return None
    merged = pd.merge(pre_data, post_data, on='Parametro', suffixes=('_Pre', '_Post'))
    for col in ['Valore_Pre', 'Valore_Post']:
        merged[col] = pd.to_numeric(merged[col].astype(str).str.replace('%', '').str.strip(), errors='coerce')
    merged['Diff %'] = ((merged['Valore_Post'] - merged['Valore_Pre']) / merged['Valore_Pre']) * 100
    return merged

def update_preview():
    preview_text.delete("1.0", "end")
    df = get_merged_df()
    if df is not None:
        preview_text.insert("end", df.round(2).to_string(index=False))
        plot_comparison_gui(df)

def plot_comparison_gui(df):
    for widget in canvas_frame.winfo_children(): widget.destroy()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=80)
    x = np.arange(len(df['Parametro']))
    width = 0.35
    ax.bar(x - width/2, df['Valore_Pre'], width, label='Pre', color='#90caf9')
    ax.bar(x + width/2, df['Valore_Post'], width, label='Post', color='#ef5350')
    ax.set_xticks(x); ax.set_xticklabels(df['Parametro'], rotation=15, ha='right')
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    FigureCanvasTkAgg(fig, master=canvas_frame).get_tk_widget().pack()

# ============================
# EXPORT PDF PRO V4
# ============================

def export_pdf():
    df = get_merged_df()
    if df is None: return
    path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile="Analisi_Evolutiva_Magistrale.pdf")
    if not path: return

    with PdfPages(path) as pdf:
        # --- PAGINA 1: TABELLA E CONFRONTO ALTEZZE ---
        fig1, (ax_t, ax_h) = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={'height_ratios': [0.7, 1.3]})
        ax_t.axis('off')
        vals = df[['Parametro', 'Valore_Pre', 'Valore_Post', 'Diff %']].round(2).values
        table = ax_t.table(cellText=vals, colLabels=["Parametro", "Pre", "Post", "Var %"], loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2)
        ax_t.set_title("SINTESI COMPARATIVA", fontsize=14, fontweight='bold', pad=20)

        # Solo Altezze
        h_df = df[df['Parametro'].str.contains('Altezza', case=False)]
        x_h = np.arange(len(h_df))
        ax_h.bar(x_h - 0.17, h_df['Valore_Pre'], 0.35, label='Iniziale (Pre)', color='#2196F3', edgecolor='black')
        ax_h.bar(x_h + 0.17, h_df['Valore_Post'], 0.35, label='Finale (Post)', color='#F44336', edgecolor='black')
        ax_h.set_xticks(x_h); ax_h.set_xticklabels(h_df['Parametro'])
        ax_h.set_ylabel("Centimetri (cm)"); ax_h.legend(); ax_h.grid(axis='y', alpha=0.3)
        ax_h.set_title("EVOLUZIONE ALTEZZE DI SALTO", fontsize=12, fontweight='bold')
        
        fig1.tight_layout(pad=4.0); pdf.savefig(); plt.close()

        # --- PAGINA 2: CONFRONTO INDICI (EUR, RSI, STIFFNESS) ---
        fig2, ax_i = plt.subplots(figsize=(8.5, 6))
        i_df = df[~df['Parametro'].str.contains('Altezza', case=False)]
        x_i = np.arange(len(i_df))
        ax_i.bar(x_i - 0.17, i_df['Valore_Pre'], 0.35, label='Pre', color='#90CAF9', alpha=0.7, edgecolor='black')
        ax_i.bar(x_i + 0.17, i_df['Valore_Post'], 0.35, label='Post', color='#EF9A9A', alpha=0.7, edgecolor='black')
        ax_i.set_xticks(x_i); ax_i.set_xticklabels(i_df['Parametro'], rotation=15)
        ax_i.set_title("INDICI NEUROMUSCOLARI E REATTIVITÀ", fontsize=12, fontweight='bold')
        ax_i.legend(); ax_i.grid(axis='y', alpha=0.3)
        
        fig2.tight_layout(pad=4.0); pdf.savefig(); plt.close()

        # --- PAGINA 3: VARIAZIONE PERCENTUALE (IMPATTO VISIVO) ---
        fig3, ax3 = plt.subplots(figsize=(8.5, 8))
        colors = ['#4CAF50' if v >= 0 else '#D32F2F' for v in df['Diff %']]
        bars = ax3.barh(df['Parametro'], df['Diff %'], color=colors, edgecolor='black', height=0.6)
        
        # Linea centrale sullo 0
        ax3.axvline(0, color='black', linewidth=1.5)
        
        # Imposta limite fisso a +/- 100 per impatto grafico
        limit = max(100, df['Diff %'].abs().max() + 20)
        ax3.set_xlim(-limit, limit)
        
        ax3.set_title("VARIAZIONE PERCENTUALE DELLE PERFORMANCE (%)", fontsize=14, fontweight='bold', pad=20)
        
        # Label dentro le barre
        for bar in bars:
            width = bar.get_width()
            # Posiziona il testo: se la barra è piccola lo mette fuori, se grande lo mette dentro
            x_pos = width/2 if abs(width) > limit/4 else (width + (5 if width > 0 else -15))
            ax3.text(x_pos, bar.get_y() + bar.get_height()/2, f'{width:+.1f}%', 
                     va='center', ha='center', fontweight='bold', 
                     color='white' if abs(width) > limit/4 else 'black')

        ax3.grid(axis='x', linestyle='--', alpha=0.4)
        ax3.invert_yaxis() # Parametri dall'alto verso il basso
        
        fig3.tight_layout(pad=4.0); pdf.savefig(); plt.close()

# ============================
# GUI TKINTER
# ============================
root = tk.Tk()
root.title("Analisi Performance Andrea - Magistrale")
frame_btns = Frame(root); frame_btns.pack(pady=10)
Button(frame_btns, text="1. Carica Report PRE", command=lambda: load_csv("pre"), width=25).grid(row=0, column=0, padx=5)
pre_label = Label(frame_btns, text="Nessun file Pre"); pre_label.grid(row=0, column=1, sticky='w')
Button(frame_btns, text="2. Carica Report POST", command=lambda: load_csv("post"), width=25).grid(row=1, column=0, padx=5)
post_label = Label(frame_btns, text="Nessun file Post"); post_label.grid(row=1, column=1, sticky='w')
Button(root, text="GENERA PDF COMPARATIVO PRO", command=export_pdf, bg="#2196F3", fg="white", font=('Arial', 10, 'bold')).pack(pady=10)
preview_text = Text(root, height=8, width=80, font=('Consolas', 9)); preview_text.pack(padx=10)
canvas_frame = Frame(root); canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
root.mainloop()