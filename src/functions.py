# Funciones de entrenamiento y métricas de 
# random forest 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

def modelo_rf(data,dropping_vars,trees):
  Y = ['quality_label']
  if dropping_vars is None:
    X = data.drop(columns = Y)
  else:
    X = data.drop(columns = dropping_vars + Y)

  # Variable objetivo
  y = data['quality_label']

  # División train/test
  # stratify = y -> para que train y test conserven aproximadamente la misma proporción de clases que tiene el dataset original.
  X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=0.2,
      random_state=42,
      stratify=y
  )

  # Modelo Random Forest
  # class_weight = 'balanced' -> penalización más alta en los errores sobre la clase minoritaria
  # Sin pesos:
  # - accuracy alta, pero nunca predice “malo”.
  # Con pesos:
  # - quizá baja un poco accuracy, pero mejora detección de “malo”.

  rf = RandomForestClassifier(
      n_estimators=trees,
      random_state=42,
      class_weight='balanced'
  )

  # Entrenamiento
  rf.fit(X_train, y_train)

  return rf, X_test, y_test

def metricas(rf, X_test, y_test):
  # Predicciones
  y_pred = rf.predict(X_test)

  # Métricas
  print(classification_report(y_test, y_pred))

  # Matriz de confusión
  cm = confusion_matrix(y_test, y_pred)

  disp = ConfusionMatrixDisplay(
      confusion_matrix=cm,
      display_labels=rf.classes_
  )

  disp.plot(cmap='Blues')
  plt.title('Matriz de confusión')
  plt.show()


# Funciones de entrenamiento y métricas de 
# regresión logística
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def modelo_rl(data, dropping_vars=None):

    Y = ['quality_label']

    # Variables predictoras
    if dropping_vars is None:
        X = data.drop(columns=Y)
    else:
        X = data.drop(columns=dropping_vars + Y)

    # Variable objetivo
    y = data['quality_label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Normalización
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    logreg = LogisticRegression(
        max_iter=5000,
        class_weight='balanced'
    )

    # Entrenamiento
    logreg.fit(X_train_scaled, y_train)

    return logreg, X_test_scaled, y_test, X.columns


def metricas_lr(logreg, X_test_scaled, y_test, columnas):

    # Predicciones
    y_pred = logreg.predict(X_test_scaled)

    # Métricas
    print(classification_report(y_test, y_pred))

    # Coeficientes
    coeficientes = pd.DataFrame({
        'Variable': columnas,
        'Coeficiente': logreg.coef_[0]
    })

    print(coeficientes.sort_values(by='Coeficiente'))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=logreg.classes_
    )

    disp.plot(cmap='Blues')

    plt.title('Matriz de confusión')

    plt.show()

