from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import List, Dict, Any, Optional
import io

app = FastAPI(title="Excel Analyzer API")

# Configurar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://excel-analyzer-xi.vercel.app"],  # Origen del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Excel Analyzer API is running"}


@app.post("/analyze")
async def analyze_excel(file: UploadFile = File(...)):
    # Verificar que el archivo sea un Excel
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="El archivo debe ser un Excel (.xlsx o .xls)")
    
    try:
        # Leer el archivo Excel
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Guardar número de filas original
        rows_before = len(df)
        
        # Detectar y limpiar datos
        df_cleaned, cleaning_summary = clean_data(df)
        
        # Guardar número de filas después de la limpieza
        rows_after = len(df_cleaned)
        
        # Generar gráficos basados en los tipos de datos
        charts = generate_charts(df_cleaned)
        
        # Preparar respuesta
        response = {
            "report": {
                "rows_before": rows_before,
                "rows_after": rows_after,
                "cleaning_summary": cleaning_summary
            },
            "charts": charts
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")


def clean_data(df: pd.DataFrame):
    """Limpia el DataFrame y devuelve un resumen de las operaciones realizadas"""
    # Inicializar contadores para el resumen
    missing_values = 0
    invalid_types = 0
    duplicates = 0
    
    # Copia del DataFrame original
    df_cleaned = df.copy()
    
    # Detectar y convertir tipos de datos
    for column in df_cleaned.columns:
        # Intentar convertir a numérico si es posible
        if df_cleaned[column].dtype == 'object':
            # Verificar si es una fecha
            try:
                df_cleaned[column] = pd.to_datetime(df_cleaned[column])
                continue
            except:
                pass
            
            # Verificar si es numérico
            try:
                numeric_values = pd.to_numeric(df_cleaned[column], errors='coerce')
                # Si más del 70% de los valores son numéricos, convertir la columna
                if numeric_values.notna().sum() / len(numeric_values) > 0.7:
                    invalid_values = df_cleaned[column][numeric_values.isna()].count()
                    invalid_types += invalid_values
                    df_cleaned[column] = numeric_values
            except:
                pass
    
    # Contar valores nulos antes de eliminarlos
    missing_values = df_cleaned.isna().sum().sum()
    
    # Eliminar filas con valores nulos
    df_cleaned = df_cleaned.dropna()
    
    # Eliminar duplicados
    duplicates = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Resumen de la limpieza
    cleaning_summary = {
        "missing_values": int(missing_values),
        "invalid_types": int(invalid_types),
        "duplicates_removed": int(duplicates)
    }
    
    return df_cleaned, cleaning_summary


def detect_chart_type(column1, column2=None):
    """Detecta el mejor tipo de gráfico basado en los tipos de datos"""
    if column2 is None:
        # Para una sola columna
        if pd.api.types.is_numeric_dtype(column1):
            return "histogram"  # Histograma para datos numéricos
        elif pd.api.types.is_datetime64_dtype(column1):
            return "line"  # Línea para series temporales
        else:
            return "pie"  # Pastel para datos categóricos
    else:
        # Para dos columnas
        if pd.api.types.is_numeric_dtype(column1) and pd.api.types.is_numeric_dtype(column2):
            return "scatter"  # Dispersión para dos variables numéricas
        elif (pd.api.types.is_numeric_dtype(column1) and not pd.api.types.is_numeric_dtype(column2)) or \
             (pd.api.types.is_numeric_dtype(column2) and not pd.api.types.is_numeric_dtype(column1)):
            return "bar"  # Barras para categórico + numérico
        elif pd.api.types.is_datetime64_dtype(column1) or pd.api.types.is_datetime64_dtype(column2):
            return "line"  # Línea para series temporales
        else:
            return "heatmap"  # Mapa de calor para dos variables categóricas


def generate_charts(df: pd.DataFrame):
    """Genera gráficos basados en los tipos de datos del DataFrame"""
    charts = []
    columns = df.columns.tolist()
    
    # Generar gráficos univariados para cada columna
    for column in columns:
        chart_type = detect_chart_type(df[column])
        
        if chart_type == "histogram" and pd.api.types.is_numeric_dtype(df[column]):
            fig = px.histogram(df, x=column, title=f"Distribución de {column}")
            charts.append({
                "title": f"Distribución de {column}",
                "type": "histogram",
                "plotly_json": json.loads(fig.to_json())
            })
        
        elif chart_type == "pie" and not pd.api.types.is_numeric_dtype(df[column]):
            # Limitar a las 10 categorías más frecuentes si hay muchas
            value_counts = df[column].value_counts().head(10)
            fig = px.pie(names=value_counts.index, values=value_counts.values, title=f"Distribución de {column}")
            charts.append({
                "title": f"Distribución de {column}",
                "type": "pie",
                "plotly_json": json.loads(fig.to_json())
            })
        
        elif chart_type == "line" and pd.api.types.is_datetime64_dtype(df[column]):
            # Para columnas de fecha, buscar una columna numérica para hacer un gráfico de línea
            numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_columns:
                for num_col in numeric_columns[:2]:  # Limitar a 2 gráficos por fecha
                    fig = px.line(df, x=column, y=num_col, title=f"{num_col} a lo largo del tiempo")
                    charts.append({
                        "title": f"{num_col} a lo largo del tiempo",
                        "type": "line",
                        "plotly_json": json.loads(fig.to_json())
                    })
    
    # Generar gráficos bivariados para combinaciones interesantes
    # Máximo 5 gráficos bivariados para no sobrecargar
    bivariate_charts = 0
    
    # Priorizar relaciones entre numéricas y categóricas
    numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_columns = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col]) 
                          and not pd.api.types.is_datetime64_dtype(df[col])]
    
    # Gráficos de barras (categórica vs numérica)
    for cat_col in categorical_columns:
        if bivariate_charts >= 5:
            break
            
        for num_col in numeric_columns[:2]:  # Limitar a 2 numéricas por categórica
            if bivariate_charts >= 5:
                break
                
            # Limitar a las 10 categorías más frecuentes
            top_categories = df[cat_col].value_counts().head(10).index
            df_filtered = df[df[cat_col].isin(top_categories)]
            
            fig = px.bar(df_filtered, x=cat_col, y=num_col, title=f"{num_col} por {cat_col}")
            charts.append({
                "title": f"{num_col} por {cat_col}",
                "type": "bar",
                "plotly_json": json.loads(fig.to_json())
            })
            bivariate_charts += 1
    
    # Gráficos de dispersión (numérica vs numérica)
    if len(numeric_columns) >= 2 and bivariate_charts < 5:
        for i in range(min(len(numeric_columns)-1, 2)):  # Limitar a 2 gráficos de dispersión
            if bivariate_charts >= 5:
                break
                
            fig = px.scatter(df, x=numeric_columns[i], y=numeric_columns[i+1], 
                           title=f"Relación entre {numeric_columns[i]} y {numeric_columns[i+1]}")
            charts.append({
                "title": f"Relación entre {numeric_columns[i]} y {numeric_columns[i+1]}",
                "type": "scatter",
                "plotly_json": json.loads(fig.to_json())
            })
            bivariate_charts += 1
    
    return charts


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)