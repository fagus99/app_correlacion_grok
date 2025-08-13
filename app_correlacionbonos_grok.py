# Aplicación Streamlit para análisis de datos de casino online
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import io
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

# Función adaptada del script proporcionado para clasificar bonos
def classify_bonuses(df):
    if 'BONOS' not in df.columns:
        raise ValueError("La columna 'BONOS' no se encontró en el archivo.")
    
    df['BONOS'] = df['BONOS'].astype(str)
    
    # Eliminar fila con 'total' si existe
    total_row_index = df[df['BONOS'].str.lower().str.contains('total', na=False)].index
    if not total_row_index.empty:
        df = df.iloc[:total_row_index[0]]
    
    # Limpiar y convertir a numérico
    df['BONOS'] = df['BONOS'].str.replace(',', '').str.replace('.', '')
    df['BONOS'] = pd.to_numeric(df['BONOS'], errors='coerce')
    
    # Eliminar nulos en BONOS
    df = df.dropna(subset=['BONOS'])
    
    # Calcular promedio después de eliminar 4 más bajos y 4 más altos
    df_sorted = df.sort_values(by='BONOS')
    df_truncated = df_sorted.iloc[4:-4] if len(df_sorted) > 8 else df_sorted
    promedio_bonos = df_truncated['BONOS'].mean()
    
    # Función de categorización
    def categorize_bonos(value, promedio):
        bajo_threshold = promedio * 0.70
        medio_threshold = promedio * 1.00
        medio_alto_threshold = promedio * 1.30
        
        if value <= bajo_threshold:
            return 'Bonos bajos'
        elif value <= medio_threshold:
            return 'Bonos medios'
        elif value <= medio_alto_threshold:
            return 'Bonos medios altos'
        else:
            return 'Bonos altos'
    
    df['Categoria_Bonos'] = df['BONOS'].apply(lambda x: categorize_bonos(x, promedio_bonos))
    
    return df, promedio_bonos

# Aplicación Streamlit
st.title('Analizador de Datos de Casino Online')
st.markdown("""
Esta aplicación analiza datos diarios de un casino online para verificar si los bonos con rollover x1 están inflando el GGR.
Sube un archivo Excel con las columnas requeridas y explora los insights.
""")

# 1. Carga de Datos
uploaded_file = st.file_uploader("Sube el archivo Excel (.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Leer el archivo (asumimos la hoja principal es 'Hoja1' como en el script adjunto)
        df = pd.read_excel(uploaded_file, sheet_name='Hoja1')
        
        # Verificar columnas requeridas
        required_cols = ['FECHA', 'GGR TOTAL', 'APOSTADO', 'PAGADO', 'GGR SPORTS', 'GGR CASINO', 'GGR SLOTS', 'ALTAS', 'LOGGS', 'BONOS', 'ACREDITACIONES', 'RETIROS', 'TOTAL USUARIOS']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas: {', '.join(missing_cols)}")
        
        # Mostrar vista previa
        st.subheader('Vista Previa de los Datos (Primeras 5 Filas)')
        st.dataframe(df.head(5))
        
        # 2. Clasificación de Bonos
        df, promedio_bonos = classify_bonuses(df)
        
        # 3. Limpieza y Preparación
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
        df = df.dropna(subset=['FECHA'])
        
        numeric_cols = ['GGR TOTAL', 'APOSTADO', 'PAGADO', 'GGR SPORTS', 'GGR CASINO', 'GGR SLOTS', 'ALTAS', 'LOGGS', 'BONOS', 'ACREDITACIONES', 'RETIROS', 'TOTAL USUARIOS']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Columnas derivadas
        df['GGR Ajustado'] = df['GGR TOTAL'] - df['BONOS']
        df['Porcentaje Bonos en GGR'] = (df['BONOS'] / df['GGR TOTAL'].replace(0, float('nan'))) * 100
        df['Mes'] = df['FECHA'].dt.to_period('M')
        
        # 4. Cálculo de Correlaciones
        st.subheader('Análisis de Correlaciones')
        
        # Matriz de correlaciones general
        corr_vars = ['BONOS', 'GGR TOTAL', 'APOSTADO', 'PAGADO', 'ACREDITACIONES', 'RETIROS']
        corr_matrix = df[corr_vars].corr()
        
        fig_heatmap, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_heatmap)
        
        # Gráficos de dispersión con regresión
        for var in ['GGR TOTAL', 'APOSTADO', 'PAGADO', 'ACREDITACIONES', 'RETIROS']:
            st.subheader(f'Bonos vs {var}')
            fig_scatter, ax = plt.subplots()
            sns.regplot(x='BONOS', y=var, data=df, ax=ax)
            st.pyplot(fig_scatter)
        
        # Correlaciones por categoría de bonos
        st.subheader('Correlaciones por Categoría de Bonos')
        categories = ['Bonos bajos', 'Bonos medios', 'Bonos medios altos', 'Bonos altos']
        for cat in categories:
            sub_df = df[df['Categoria_Bonos'] == cat]
            if len(sub_df) > 1:
                st.write(f'**{cat}:**')
                corrs = {}
                for var in ['GGR TOTAL', 'APOSTADO', 'PAGADO', 'ACREDITACIONES', 'RETIROS']:
                    r, _ = pearsonr(sub_df['BONOS'], sub_df[var])
                    corrs[var] = round(r, 2)
                st.write(corrs)
            else:
                st.write(f'**{cat}:** Insuficientes datos para calcular correlaciones.')
        
        # 5. Otras Métricas Relevantes
        st.subheader('Tendencias Mensuales')
        monthly_avg = df.groupby('Mes').agg({
            'GGR TOTAL': 'mean',
            'APOSTADO': 'mean',
            'BONOS': 'mean',
            'RETIROS': 'mean'
        }).reset_index()
        monthly_avg['Mes'] = monthly_avg['Mes'].astype(str)  # Para plotting
        
        for col in ['GGR TOTAL', 'APOSTADO', 'BONOS', 'RETIROS']:
            fig_line, ax = plt.subplots()
            sns.lineplot(x='Mes', y=col, data=monthly_avg, marker='o', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig_line)
        
        st.subheader('Impacto de Bonos')
        avg_pct_bonos = df['Porcentaje Bonos en GGR'].mean()
        st.write(f'Porcentaje promedio de Bonos en GGR: {avg_pct_bonos:.2f}%')
        st.write(f'GGR Ajustado promedio: {df["GGR Ajustado"].mean():.2f}')
        
        st.subheader('Análisis de Varianza: Correlaciones por Mes')
        for mes in sorted(df['Mes'].unique()):
            sub_df = df[df['Mes'] == mes]
            if len(sub_df) > 1:
                r, _ = pearsonr(sub_df['BONOS'], sub_df['GGR TOTAL'])
                st.write(f'Mes {mes}: Correlación Bonos-GGR: {r:.2f}')
            else:
                st.write(f'Mes {mes}: Insuficientes datos.')
        
        st.subheader('Resumen Estadístico')
        st.dataframe(df[numeric_cols + ['GGR Ajustado', 'Porcentaje Bonos en GGR']].describe())
        
        # 6. Resumen Final
        st.subheader('Resumen Interpretativo')
        overall_corr = corr_matrix.loc['BONOS', 'GGR TOTAL']
        if overall_corr > 0.7:
            summary_text = f"La correlación entre Bonos y GGR es {overall_corr:.2f}, lo que sugiere un inflado significativo por bonos."
        elif overall_corr > 0.3:
            summary_text = f"La correlación es moderada ({overall_corr:.2f}), indicando posible inflado en ciertos escenarios."
        else:
            summary_text = f"La correlación es baja ({overall_corr:.2f}), con poco evidencia de inflado."
        
        # Insights adicionales con lógica simple
        high_bonos_months = monthly_avg[monthly_avg['BONOS'] > promedio_bonos]
        if not high_bonos_months.empty:
            avg_ggr_increase = ((high_bonos_months['GGR TOTAL'] - monthly_avg['GGR TOTAL'].mean()) / monthly_avg['GGR TOTAL'].mean()) * 100
            summary_text += f"\nEn meses con bonos altos, el GGR aumenta un {avg_ggr_increase.mean():.2f}% en promedio."
        
        if avg_pct_bonos > 15:
            summary_text += "\nRecomendación: Reducir rollover o limitar bonos ya que el inflado supera el 15%."
        else:
            summary_text += "\nRecomendación: Monitorear bonos, pero no hay inflado crítico detectado."
        
        st.write(summary_text)
        
        # 7. Dashboard Interactivo
        st.subheader('Dashboard Interactivo')
        months = sorted(df['Mes'].unique().astype(str))
        selected_month = st.selectbox('Filtrar por Mes', ['Todos'] + months)
        
        if selected_month != 'Todos':
            filtered_df = df[df['Mes'] == pd.Period(selected_month)]
        else:
            filtered_df = df
        
        st.dataframe(filtered_df)
        
        # Gráfico de dispersión interactivo
        fig_interactive, ax = plt.subplots()
        sns.scatterplot(x='BONOS', y='GGR TOTAL', hue='Categoria_Bonos', data=filtered_df, ax=ax)
        st.pyplot(fig_interactive)
        
        # Opción de Descargar PDF
        if st.button('Descargar Reporte en PDF'):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Reporte de Análisis de Casino Online", ln=1, align='C')
            pdf.multi_cell(0, 10, summary_text)
            
            # Añadir resumen estadístico como texto
            stats_str = df.describe().to_string()
            pdf.multi_cell(0, 10, "Resumen Estadístico:\n" + stats_str)
            
            # Añadir imágenes (heatmap y un scatter como ejemplo)
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Guardar heatmap
                heatmap_path = os.path.join(tmpdirname, 'heatmap.png')
                fig_heatmap.savefig(heatmap_path)
                pdf.image(heatmap_path, x=10, w=180)
                
                # Guardar scatter ejemplo
                scatter_path = os.path.join(tmpdirname, 'scatter.png')
                fig_scatter.savefig(scatter_path)
                pdf.image(scatter_path, x=10, w=180)
            
            pdf_bytes = io.BytesIO()
            pdf.output(pdf_bytes)
            pdf_bytes.seek(0)
            
            st.download_button(
                label="Descargar PDF",
                data=pdf_bytes,
                file_name="reporte_casino.pdf",
                mime="application/pdf"
            )
    
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
