from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def resolver_sistema():
    resultado = None
    if request.method == 'POST':
        try:
            # Leer número de incógnitas
            n = int(request.form['num_incognitas'])

            # Leer coeficientes y términos independientes
            coeficientes = []
            for i in range(n):
                fila = request.form[f'fila_{i}'].split()
                coeficientes.append([float(num) for num in fila])

            independientes = request.form['independientes'].split()
            independientes = [float(num) for num in independientes]

            # Resolver el sistema
            A = np.array(coeficientes)
            b = np.array(independientes)
            solucion = np.linalg.solve(A, b)

            letras = ['x', 'y', 'z', 'w', 'v']
            resultado = [f"{letras[i]} = {solucion[i]}" for i in range(n)]

        except Exception as e:
            resultado = [f"Error: {str(e)}"]

    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)