import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def select_file():
    file_path = filedialog.askopenfilename(
        title="Seleziona il file Excel",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    entry_file.delete(0, tk.END)
    entry_file.insert(0, file_path)

def select_folder():
    folder_path = filedialog.askdirectory(title="Seleziona la cartella di destinazione")
    entry_folder.delete(0, tk.END)
    entry_folder.insert(0, folder_path)

def export_csv():
    excel_file = entry_file.get()
    output_folder = entry_folder.get()

    if not os.path.isfile(excel_file):
        messagebox.showerror("Errore", "Seleziona un file Excel valido.")
        return
    if not os.path.isdir(output_folder):
        messagebox.showerror("Errore", "Seleziona una cartella di destinazione valida.")
        return

    try:
        # Legge tutti i fogli in un dizionario {nome_foglio: DataFrame}
        all_sheets = pd.read_excel(excel_file, sheet_name=None, usecols="A:C")
        
        for sheet_name, df in all_sheets.items():
            csv_filename = os.path.join(output_folder, f"{sheet_name}.csv")
            df.to_csv(csv_filename, index=False, sep=",")
        
        messagebox.showinfo("Successo", "Esportazione completata!")
    except Exception as e:
        messagebox.showerror("Errore", f"Qualcosa è andato storto:\n{e}")

# Creazione GUI
root = tk.Tk()
root.title("Excel → CSV per foglio")

# File selection
tk.Label(root, text="File Excel:").grid(row=0, column=0, sticky="e")
entry_file = tk.Entry(root, width=50)
entry_file.grid(row=0, column=1)
tk.Button(root, text="Sfoglia", command=select_file).grid(row=0, column=2)

# Folder selection
tk.Label(root, text="Cartella di destinazione:").grid(row=1, column=0, sticky="e")
entry_folder = tk.Entry(root, width=50)
entry_folder.grid(row=1, column=1)
tk.Button(root, text="Sfoglia", command=select_folder).grid(row=1, column=2)

# Export button
tk.Button(root, text="Esporta CSV", command=export_csv, bg="green", fg="white").grid(row=2, column=1, pady=10)

root.mainloop()