# Wine Classification

Proyecto de clasificación de calidad de vinos utilizando modelos de Machine Learning en Python.

## Objetivo

Desarrollar modelos de clasificación capaces de predecir la calidad de un vino a partir de sus características fisicoquímicas.

El proyecto incluye:

* Limpieza y preparación de datos
* Entrenamiento de modelos
* Evaluación con métricas de clasificación
* Modularización de funciones
* Persistencia de modelos con Joblib
* Pruebas separadas en notebooks de testing

---

# Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Jupyter Notebook

---

# Modelos implementados

## Random Forest

Modelo de ensamble basado en árboles de decisión.

Características:

* Uso de `class_weight='balanced'`
* Evaluación mediante matriz de confusión
* Reporte de clasificación

---

## Regresión Logística

Modelo lineal para clasificación multiclase.

Características:

* Normalización con `StandardScaler`
* Entrenamiento balanceado
* Interpretación mediante coeficientes
* Evaluación con métricas y matriz de confusión

---

# Métricas utilizadas

* Accuracy
* Precision
* Recall
* F1-score
* Matriz de confusión

---

# Cómo ejecutar el proyecto

## 1. Clonar el repositorio

```bash
git clone https://github.com/Diana-robles20/wine-classification.git
```

## 2. Entrar al proyecto

```bash
cd wine-classification
```

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4. Ejecutar notebooks

Abrir Jupyter Notebook o VS Code y ejecutar:

* `training.ipynb`
* `testing.ipynb`

---

# Resultados

Se compararon distintos modelos de clasificación para identificar cuál logra un mejor desempeño en la predicción de calidad de vinos.

La regresión logística permitió interpretar el impacto de las variables mediante sus coeficientes, mientras que Random Forest ofreció una alternativa no lineal para comparar desempeño.

---

# Autor

Diana Robles

Estudiante de Licenciatura en Inteligencia Artificial y Ciencia de Datos.
