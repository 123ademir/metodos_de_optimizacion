from flask import Flask, render_template, request
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # Tu código actual aquí...
    z_max = None
    punto = None
    if request.method == "POST":
        a = float(request.form["a"])
        b = float(request.form["b"])
        c = float(request.form["c"])
        x_max = float(request.form["x_max"])
        y_max = float(request.form["y_max"])

        vertices = []
        
        # Origen
        vertices.append((0, 0))
        
        # Intersección con x = x_max
        if x_max <= c:
            vertices.append((x_max, 0))
            y_int = min(y_max, c - x_max)
            vertices.append((x_max, y_int))
        
        # Intersección con y = y_max
        if y_max <= c:
            vertices.append((0, y_max))
            x_int = min(x_max, c - y_max)
            vertices.append((x_int, y_max))
        
        # Intersección de x + y = c con los ejes
        if c <= x_max:
            vertices.append((c, 0))
        if c <= y_max:
            vertices.append((0, c))
        
        # Eliminar duplicados
        vertices = list(set(vertices))
        
        # Filtrar solo puntos factibles
        vertices_factibles = []
        for x, y in vertices:
            if x >= 0 and y >= 0 and x <= x_max and y <= y_max and x + y <= c:
                vertices_factibles.append((x, y))
        
        # Calcular z para cada vértice factible
        if vertices_factibles:
            z_vals = [(a * x + b * y, (x, y)) for x, y in vertices_factibles]
            z_max, punto = max(z_vals)

    return render_template("index.html", z_max=z_max, punto=punto)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    # Abrir navegador después de 1 segundo
    threading.Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)