# Excel Analyzer API

API en FastAPI para procesar archivos Excel, detectar tipos de datos, limpiar registros y generar gráficos dinámicos con Plotly.

## Requisitos

- Python 3.8+
- Entorno virtual (recomendado)

## Instalación

1. Activar el entorno virtual:

```bash
# En Windows
venv\Scripts\activate

# En macOS/Linux
source venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

El servidor se iniciará en `http://localhost:8000`

## Endpoints

- `GET /`: Verificar que la API está funcionando
- `POST /analyze`: Analizar un archivo Excel
  - Recibe un archivo Excel mediante `multipart/form-data`
  - Devuelve un informe de limpieza y gráficos generados

## Respuesta de ejemplo

```json
{
  "report": {
    "rows_before": 1000,
    "rows_after": 950,
    "cleaning_summary": {
      "missing_values": 30,
      "invalid_types": 15,
      "duplicates_removed": 5
    }
  },
  "charts": [
    {
      "title": "Ventas por Categoría",
      "type": "bar",
      "plotly_json": {...}
    },
    {
      "title": "Tendencia de Ventas",
      "type": "line",
      "plotly_json": {...}
    }
  ]
}
```