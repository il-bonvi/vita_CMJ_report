import pandas as pd
import tkinter as tk
from tkinter import filedialog, Label, Button, Text
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

# ============================
# VARIABILI GLOBALI
# ============================
pre_file = None
post_file = None
pre_data = None
post_data = None

# ============================
# FUNZIONI
# ============================

def load_csv(pre_or_post):
    global pre_file, post_file, pre_data, post_data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if not file_path:
        return
    if pre_or_post == "pre":
        pre_file = file_path
        pre_data = pd.read_csv(file_path)
        pre_label.config(text=f"Pre: {os.path.basename(file_path)}")
    else:
        post_file = file_path
        post_data = pd.read_csv(file_path)
        post_label.config(text=f"Post: {os.path.basename(file_path)}")
    update_preview()

def update_preview():
    preview_text.delete("1.0", "end")
    if pre_data is None or post_data is None:
        return

    param_visibili = [
        'Fmax (N)',
        't concentrica (s)',
        'Forza media concentrica (N)',
        'Impulso concentrico (N·s)',
        'Δv al take-off (m/s)',
        'Impulso / BW (s)',
        'Tempo di volo (s)',
        'Altezza salto (cm)',
        'Bilanciamento medio DX (%)',
        'Massa soggetto (kg)'
    ]

    combined = pd.DataFrame({
        "Parametro": pre_data['Parametro'],
        "Pre": pre_data['Valore'],
        "Post": post_data['Valore']
    })

    combined = combined[combined['Parametro'].isin(param_visibili)]

    combined['Diff (%)'] = (pd.to_numeric(combined['Post'], errors='coerce') -
                            pd.to_numeric(combined['Pre'], errors='coerce')) / \
                            pd.to_numeric(combined['Pre'], errors='coerce') * 100
    combined['Diff (%)'] = combined['Diff (%)'].round(1)

    preview_text.insert("end", combined[['Parametro','Pre','Post']].to_string(index=False))
    update_plot(combined)

def update_plot(df):
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Gruppi aggiornati
    forza_param = ['Fmax (N)', 'Forza media concentrica (N)', 'Impulso concentrico (N·s)']
    velocita_param = ['Δv al take-off (m/s)']
    tempo_param = ['t concentrica (s)', 'Tempo di volo (s)', 'Impulso / BW (s)']
    altezza_param = ['Altezza salto (cm)']
    bil_param = ['Bilanciamento medio DX (%)']
    massa_param = ['Massa soggetto (kg)']

    gruppi = [
        ('Forza / Potenza', forza_param),
        ('Velocità', velocita_param),
        ('Tempo', tempo_param),
        ('Altezza', altezza_param),
        ('Bilanciamento', bil_param),
        ('Massa', massa_param)
    ]

    for titolo, param_list in gruppi:
        subset = df[df['Parametro'].isin(param_list)]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8,4))
        param = subset['Parametro']
        pre_vals = pd.to_numeric(subset['Pre'], errors='coerce')
        post_vals = pd.to_numeric(subset['Post'], errors='coerce')

        x = range(len(param))
        width = 0.35
        ax.bar(x, pre_vals, width=width, label='Pre', color='skyblue')
        ax.bar([i + width for i in x], post_vals, width=width, label='Post', color='orange')

        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(param, rotation=45, ha='right')
        ax.set_ylabel('Valore')
        ax.set_title(f'Confronto Pre vs Post - {titolo}')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=5)

    # Grafico unico delle variazioni percentuali
    fig2, ax2 = plt.subplots(figsize=(10,4))
    param_all = df['Parametro']
    diff_all = df['Diff (%)']
    x_all = range(len(param_all))
    ax2.bar(x_all, diff_all, color=['blue' if v>=0 else 'red' for v in diff_all])
    ax2.set_xticks(x_all)
    ax2.set_xticklabels(param_all, rotation=45, ha='right')
    ax2.set_ylabel('Variazione (%)')
    ax2.set_title('Variazioni percentuali Pre vs Post')
    ax2.grid(alpha=0.3)
    fig2.tight_layout()

    canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=5)

def export_pdf():
    if pre_data is None or post_data is None:
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                             filetypes=[("PDF files","*.pdf")],
                                             initialfile="report_compare.pdf")
    if not file_path:
        return

    df = pd.DataFrame({
        "Parametro": pre_data['Parametro'],
        "Pre": pre_data['Valore'],
        "Post": post_data['Valore']
    })
    df = df[df['Parametro'].isin([
        'Fmax (N)','t concentrica (s)','Forza media concentrica (N)','Impulso concentrico (N·s)',
        'Δv al take-off (m/s)','Impulso / BW (s)','Tempo di volo (s)','Altezza salto (cm)',
        'Bilanciamento medio DX (%)','Massa soggetto (kg)'
    ])]
    df['Diff (%)'] = (pd.to_numeric(df['Post'], errors='coerce') - pd.to_numeric(df['Pre'], errors='coerce')) / \
                     pd.to_numeric(df['Pre'], errors='coerce') * 100
    df['Diff (%)'] = df['Diff (%)'].round(1)

    with PdfPages(file_path) as pdf:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.axis('off')
        table_data = df[['Parametro','Pre','Post']].values.tolist()
        table = ax.table(cellText=table_data, colLabels=['Parametro','Pre','Post'], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2,1.5)
        ax.set_title("Tabella Pre vs Post", fontsize=16, fontweight='bold')
        pdf.savefig(); plt.close()

        gruppi = [
            ('Forza / Potenza',['Fmax (N)','Forza media concentrica (N)','Impulso concentrico (N·s)']),
            ('Velocità',['Δv al take-off (m/s)']),
            ('Tempo',['t concentrica (s)','Tempo di volo (s)','Impulso / BW (s)']),
            ('Altezza',['Altezza salto (cm)']),
            ('Bilanciamento',['Bilanciamento medio DX (%)']),
            ('Massa',['Massa soggetto (kg)'])
        ]
        for titolo, param_list in gruppi:
            subset = df[df['Parametro'].isin(param_list)]
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(8,4))
            x = range(len(subset))
            width = 0.35
            ax.bar(x, subset['Pre'], width=width, label='Pre', color='skyblue')
            ax.bar([i+width for i in x], subset['Post'], width=width, label='Post', color='orange')
            ax.set_xticks([i + width/2 for i in x])
            ax.set_xticklabels(subset['Parametro'], rotation=45, ha='right')
            ax.set_ylabel('Valore')
            ax.set_title(f'Confronto Pre vs Post - {titolo}')
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            pdf.savefig(); plt.close()

        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.bar(range(len(df)), df['Diff (%)'], color=['green' if v>=0 else 'red' for v in df['Diff (%)']])
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['Parametro'], rotation=45, ha='right')
        ax2.set_ylabel('Variazione (%)')
        ax2.set_title('Variazioni percentuali Pre vs Post')
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        pdf.savefig(); plt.close()

    print(f"Report PDF generato: {file_path}")

# ============================
# CREAZIONE GUI
# ============================

root = tk.Tk()
root.title("Confronto Pre/Post CMJ")

Button(root, text="Seleziona CSV Pre", command=lambda: load_csv("pre")).grid(row=0, column=0, pady=5)
pre_label = Label(root, text="Pre: nessun file selezionato")
pre_label.grid(row=0, column=1, sticky='w')

Button(root, text="Seleziona CSV Post", command=lambda: load_csv("post")).grid(row=1, column=0, pady=5)
post_label = Label(root, text="Post: nessun file selezionato")
post_label.grid(row=1, column=1, sticky='w')

Button(root, text="Esporta PDF", command=export_pdf).grid(row=2, column=0, pady=5)

preview_text = Text(root, height=15, width=80)
preview_text.grid(row=3, column=0, columnspan=2, pady=5)

canvas = tk.Canvas(root, width=900, height=400)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.grid(row=4, column=0, columnspan=2, pady=5)
scrollbar.grid(row=4, column=2, sticky='ns')

plot_frame = scrollable_frame

root.mainloop()