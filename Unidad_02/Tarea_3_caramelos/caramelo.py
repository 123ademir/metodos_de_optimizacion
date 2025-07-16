import random
import customtkinter as ctk
from collections import Counter
from tkinter import messagebox

CARAMEL_TYPES = ['limon', 'huevo', 'pera']

# ---------- LÓGICA DEL JUEGO ----------
def can_make_chupetin(inv):
    return inv['limon'] >= 2 and inv['huevo'] >= 2 and inv['pera'] >= 2

def get_faltantes(inv):
    return {
        'limon': max(0, 2 - inv['limon']),
        'huevo': max(0, 2 - inv['huevo']),
        'pera': max(0, 2 - inv['pera'])
    }

def make_chupetin(inv, pasos, chupetines_actuales, caramelos_extra_total):
    usados = {'limon': 2, 'huevo': 2, 'pera': 2}
    for k in usados:
        inv[k] -= usados[k]

    # Elegir 2 caramelos estratégicos
    faltantes = get_faltantes(inv)
    extra = []
    for dulce, cantidad in sorted(faltantes.items(), key=lambda x: -x[1]):
        extra.extend([dulce] * min(cantidad, 2 - len(extra)))
    while len(extra) < 2:
        extra.append(random.choice(CARAMEL_TYPES))
    inv.update(extra)
    caramelos_extra_total.extend(extra)

    pasos.append(f"\n🍭 Chupetín #{chupetines_actuales + 1} obtenido:")
    pasos.append(f"   - Usado: {usados}")
    pasos.append(f"   - Caramelos extra: {extra}")
    pasos.append(f"   - Inventario: {dict(inv)}")

def vender_chupetin(inv, pasos, caramelos_extra_total, chupetines_restantes, venta_num):
    # Elegir 6 caramelos estratégicos para maximizar posibilidades
    faltantes = get_faltantes(inv)
    elegidos = []
    for dulce, cantidad in sorted(faltantes.items(), key=lambda x: -x[1]):
        elegidos.extend([dulce] * min(6 - len(elegidos), cantidad if cantidad > 0 else 2))
    while len(elegidos) < 6:
        elegidos.append(random.choice(CARAMEL_TYPES))
    inv.update(elegidos)
    caramelos_extra_total.extend(elegidos)

    pasos.append(f"\n🔁 Venta #{venta_num}: 1 chupetín vendido por 6 caramelos")
    pasos.append(f"   - Caramelos recibidos: {elegidos}")
    pasos.append(f"   - Inventario: {dict(inv)}")

def get_inventory_from_selection(selecciones):
    all_candies = []
    for pair in selecciones:
        all_candies.extend(pair)
    return Counter(all_candies)

def actualizar_progreso(valor, total):
    progreso = min(valor / total, 1.0)
    progress_bar.set(progreso)
    progress_label.configure(text=f"Chupetines: {valor}")
    app.update_idletasks()

# ---------- SIMULACIÓN PRINCIPAL ----------
def simular_juego():
    output_text.configure(state="normal")
    output_text.delete("0.0", "end")
    num_jugadores = len(jugadores_comboboxes)
    progress_label.configure(text=f"Chupetines: 0")
    progress_bar.set(0)

    pasos = []
    selecciones = []

    for par in jugadores_comboboxes:
        dulces = [par[0].get(), par[1].get()]
        if dulces[0] not in CARAMEL_TYPES or dulces[1] not in CARAMEL_TYPES:
            messagebox.showerror("Error", "Todos los jugadores deben tener 2 caramelos válidos.")
            return
        selecciones.append(dulces)

    pasos.append("📦 Caramelos iniciales de cada jugador:")
    for i, dulces in enumerate(selecciones, 1):
        pasos.append(f"👤 Persona {i}: {dulces}")

    inv = get_inventory_from_selection(selecciones)
    pasos.append(f"\n🎒 Inventario inicial: {dict(inv)}")

    chupetines = 0
    caramelos_extra_total = []
    ventas = 0

    pasos.append("\n🍭 Fase de intercambios iniciales:")
    while can_make_chupetin(inv):
        make_chupetin(inv, pasos, chupetines, caramelos_extra_total)
        chupetines += 1
        actualizar_progreso(chupetines, num_jugadores)

    # Fase de ventas de chupetines si no se alcanza el objetivo
    pasos.append("\n🔄 Fase de ventas para intentar generar más chupetines:")
    while chupetines < num_jugadores:
        if chupetines == 0:
            pasos.append("⛔ No se pueden formar más chupetines ni vender.")
            break
        # Vender 1 chupetín
        vender_chupetin(inv, pasos, caramelos_extra_total, chupetines, ventas + 1)
        chupetines -= 1
        ventas += 1
        while can_make_chupetin(inv):
            make_chupetin(inv, pasos, chupetines, caramelos_extra_total)
            chupetines += 1
            actualizar_progreso(chupetines, num_jugadores)

    pasos.append(f"\n📊 Total chupetines obtenidos: {chupetines}")
    pasos.append(f"🔁 Total de ventas realizadas: {ventas}")
    pasos.append(f"🎁 Total caramelos extra obtenidos: {caramelos_extra_total}")
    pasos.append(f"📦 Inventario final: {dict(inv)}")

    resultado = "✅ ¡Objetivo logrado! 🎉" if chupetines >= num_jugadores else "❌ No se logró el objetivo."
    pasos.append(f"\n📌 Resultado final: {resultado}")

    for linea in pasos:
        output_text.insert("end", linea + "\n")
    output_text.configure(state="disabled")

    if chupetines >= num_jugadores:
        messagebox.showinfo("🎉 Éxito", "¡Todos tienen al menos un chupetín!")
    else:
        messagebox.showwarning("❌ Fallo", "No se logró que todos tengan chupetín.")

# ---------- INTERFAZ CUSTOMTKINTER ----------
def crear_jugadores():
    global jugadores_comboboxes
    for widget in jugadores_frame.winfo_children():
        widget.destroy()

    try:
        num = int(entry_jugadores.get())
        if num <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa un número válido de jugadores.")
        return

    jugadores_comboboxes = []
    for i in range(num):
        fila = ctk.CTkFrame(jugadores_frame)
        fila.pack(pady=4)
        label = ctk.CTkLabel(fila, text=f"Jugador {i+1}", width=100)
        label.pack(side="left", padx=5)

        combo1 = ctk.CTkComboBox(fila, values=CARAMEL_TYPES, width=120)
        combo1.pack(side="left", padx=5)
        combo1.set(CARAMEL_TYPES[0])

        combo2 = ctk.CTkComboBox(fila, values=CARAMEL_TYPES, width=120)
        combo2.pack(side="left", padx=5)
        combo2.set(CARAMEL_TYPES[1])

        jugadores_comboboxes.append((combo1, combo2))

# ---------- CONFIGURACIÓN DE LA VENTANA ----------
ctk.set_appearance_mode("light")  # o "dark"
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("🍬 Juego de Caramelos y Chupetines")
app.geometry("940x800")
app.resizable(False, False)

frame = ctk.CTkFrame(app, corner_radius=10)
frame.pack(pady=20, padx=20, fill="both", expand=True)

title = ctk.CTkLabel(frame, text="🍬 Todos por el Chupetín 🍭", font=("Arial", 22, "bold"))
title.pack(pady=10)

control_frame = ctk.CTkFrame(frame)
control_frame.pack(pady=10)

entry_label = ctk.CTkLabel(control_frame, text="Número de jugadores:")
entry_label.pack(side="left", padx=10)

entry_jugadores = ctk.CTkEntry(control_frame, width=80)
entry_jugadores.pack(side="left", padx=5)
entry_jugadores.insert(0, "5")

crear_btn = ctk.CTkButton(control_frame, text="Crear Jugadores", command=crear_jugadores)
crear_btn.pack(side="left", padx=10)

jugadores_frame = ctk.CTkFrame(frame)
jugadores_frame.pack(pady=10)

progress_label = ctk.CTkLabel(frame, text="Chupetines: 0", font=("Arial", 14))
progress_label.pack(pady=(15, 5))

progress_bar = ctk.CTkProgressBar(frame, width=400)
progress_bar.pack(pady=(0, 20))
progress_bar.set(0)

simular_btn = ctk.CTkButton(frame, text="▶ Simular Juego", command=simular_juego, font=("Arial", 14))
simular_btn.pack(pady=10)

output_text = ctk.CTkTextbox(frame, width=880, height=300, font=("Courier New", 12))
output_text.pack(pady=10)
output_text.configure(state="disabled")

crear_jugadores()
app.mainloop()