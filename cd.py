import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import r2_score
import streamlit as st

# Configurações Globais de Estilo de Gráficos
plt.style.use('dark_background')
matplotlib.rcParams.update({
    'axes.facecolor': '#1f1f1f',
    'figure.facecolor': '#1f1f1f',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#444444',
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# -----------------------------
# PROCESSAMENTO DE DADOS
# -----------------------------
def process_data():
    df = pd.read_csv('f1.csv')

    # verrificação de nulos
    df = df.dropna(subset=['Compound'])

    # Tratando os outliers
    colunas_tratar = ['LapTime (s)', 'TyreLife']
    for col in colunas_tratar:
        Q1 = df[col].quantile(0.15)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=limite_inferior, upper=limite_superior)

    # tratando cumulative_degradation de forma mais agressiva
    p05 = df['Cumulative_Degradation'].quantile(0.07)
    p95 = df['Cumulative_Degradation'].quantile(0.93)
    df['Cumulative_Degradation'] = df['Cumulative_Degradation'].clip(lower=p05, upper=p95)

    # Corrigindo colunas
    df = df.sort_values(['Race', 'Year', 'Driver', 'LapNumber'])
    df['Position_Change'] = df.groupby(['Race', 'Year', 'Driver'])['Position'].diff().fillna(0) * -1
    df['PitStop'] = df.groupby(['Race','Year','Driver'])['Stint'].diff().fillna(0).gt(0).astype(int)
    df['PitNextLap'] = df.groupby(['Race','Year','Driver'])['PitStop'].shift(-1).fillna(0).astype(int)

    # Removendo corridas de test
    corridas_remover = ['Pre-Season Testing', 'Pre-Season Test', 'Pre-Season Track Session']
    df = df[~df['Race'].isin(corridas_remover)]

    # Novas Features
    df['Lap_vs_Driver_Avg'] = df['LapTime (s)'] - df.groupby(['Race', 'Driver', 'Year'])['LapTime (s)'].transform('mean')
    df['Lap_vs_Race_Avg'] = df['LapTime (s)'] - df.groupby(['Race','Year'])['LapTime (s)'].transform('mean')

    media_stint = df.groupby(['Race','Year','Compound'])['TyreLife'].transform('mean')
    df['TyreLife_Relative'] = df['TyreLife'] - media_stint
    df['Tyre_Efficiency'] = df['TyreLife'] / df['LapTime (s)']
    df['Total_Position_Gain'] = df.groupby(['Race','Year', 'Driver'])['Position_Change'].cumsum()

    max_stint = df.groupby(['Race', 'Year', 'Driver', 'Stint'])['TyreLife'].transform('max')
    df['Tyre_Usage_Percent'] = df['TyreLife'] / max_stint

    # Outliers de novas features
    colunas_novas_tratar = ['LapTime_Delta', 'Lap_vs_Driver_Avg']
    for col in colunas_novas_tratar:
        if col in df.columns:
            Q1 = df[col].quantile(0.15)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

    return df

# -----------------------------
# FUNÇÕES DE GRÁFICOS
# -----------------------------

def plot_tire_usage(df, compounds=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    compound_colors = {'SOFT': '#e10600', 'MEDIUM': '#ffcc00', 'HARD': '#ffffff', 'WET': '#0000ff', 'INTERMEDIATE': '#00ff00'}
    selected_compounds = compounds if compounds else ['SOFT', 'MEDIUM', 'HARD']
    for composto in selected_compounds:
        color = compound_colors.get(composto, '#d1d1d1')
        temp = df[(df['Compound'] == composto) & (df['PitStop'] == 0)]
        if temp.empty: continue
        media = temp.groupby(pd.cut(temp['Tyre_Usage_Percent'], bins=15), observed=False)['LapTime_Delta'].mean()
        x = [i.mid for i in media.index]
        ax.plot(x, media.values, marker='o', label=composto, color=color, linewidth=2)
    ax.set_xlabel('Tyre Usage Percent')
    ax.set_ylabel('Average LapTime Delta')
    ax.legend()
    ax.grid(True)
    return fig

def plot_position_change(df):
    media_original = df.groupby('Position_Change')['LapTime_Delta'].mean().reset_index()
    if media_original.empty: return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(media_original['Position_Change'], media_original['LapTime_Delta'], marker='o', alpha=0.8, color='#e10600')
    ax.set_xlabel('Position Change')
    ax.set_ylabel('Average LapTime Delta')
    ax.grid(True)
    return fig

def plot_degradation_impact(df, compounds=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    compound_colors = {'SOFT': '#e10600', 'MEDIUM': '#ffcc00', 'HARD': '#ffffff', 'WET': '#0000ff', 'INTERMEDIATE': '#00ff00'}
    selected_compounds = compounds if compounds else ['SOFT', 'MEDIUM', 'HARD']
    for composto in selected_compounds:
        color = compound_colors.get(composto, '#d1d1d1')
        temp = df[df['Compound'] == composto]
        if temp.empty: continue
        media = temp.groupby(pd.cut(temp['Cumulative_Degradation'], bins=30), observed=False)['Lap_vs_Driver_Avg'].mean()
        ax.plot(media.values, marker='o', linewidth=2, alpha=0.8, label=composto, color=color)
    ax.axhline(y=0, linestyle='--', color='white', alpha=0.7)
    ax.set_xlabel('Cumulative Degradation')
    ax.set_ylabel('Average Lap vs Driver Avg')
    ax.legend(title='Compound')
    ax.grid(True)
    return fig

def plot_cumulative_degradation(df, compounds=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    compound_colors = {'SOFT': '#e10600', 'MEDIUM': '#ffcc00', 'HARD': '#ffffff', 'WET': '#0000ff', 'INTERMEDIATE': '#00ff00'}
    selected_compounds = compounds if compounds else ['SOFT', 'MEDIUM', 'HARD']

    # Menor para permitir visualização em corridas individuais
    MIN_OBS = 3

    for compound in selected_compounds:
        color = compound_colors.get(compound, '#d1d1d1')
        temp = df[df['Compound'] == compound].copy()
        if temp.empty: continue

        # Remove outliers de vida útil por composto (ex: Softs que duram 40 voltas)
        # Usamos o percentil 95 para cortar os valores absurdos de cada composto individualmente
        tyre_max = temp['TyreLife'].quantile(0.95)
        temp = temp[temp['TyreLife'] <= tyre_max]

        counts = temp.groupby('TyreLife').size().reset_index(name='Count')
        valid_tyrelife = counts[counts['Count'] >= MIN_OBS]['TyreLife']

        # Usamos MEDIANA em vez de média para ser mais robusto contra distorções
        plot_data = (
            temp[temp['TyreLife'].isin(valid_tyrelife)]
            .groupby('TyreLife')['Cumulative_Degradation']
            .median()
            .reset_index()
        )

        if plot_data.empty: continue

        ax.plot(
            plot_data['TyreLife'],
            plot_data['Cumulative_Degradation'],
            linewidth=2,
            label=compound,
            color=color
        )

    ax.set_xlabel("TyreLife (laps)")
    ax.set_ylabel("Median Cumulative Degradation")
    ax.set_title("Cumulative Degradation over Tire Life")
    ax.grid(True)
    ax.legend(title="Compound")
    return fig

def plot_avg_laptime_compound(df, race=None, year=None, compounds=None):
    if race == "Overall":
        df_filtered = df[df['Year'] == year] if year and year != "All" else df
    elif race and year:
        df_filtered = df[(df['Race'] == race) & (df['Year'] == year)]
    else:
        df_filtered = df

    if compounds:
        df_filtered = df_filtered[df_filtered['Compound'].isin(compounds)]

    df_sem_pit = df_filtered[df_filtered['PitStop'] == 0]
    if df_sem_pit.empty:
        return None
    # Agrupando por composto para pegar a média geral da seleção
    media_corrida = df_sem_pit.groupby('Compound')['LapTime (s)'].mean().reset_index()
    medianas = media_corrida.groupby('Compound')['LapTime (s)'].median()

    fig, ax = plt.subplots(figsize=(8, 4))
    df_sem_pit.boxplot(column='LapTime (s)', by='Compound', showfliers=False, ax=ax, patch_artist=True,
                     boxprops=dict(facecolor='#2d2d2d', color='#d1d1d1'),
                     medianprops=dict(color='#e10600', linewidth=2))
    # Ajuste do limite superior do eixo Y para dar espaço aos números
    y_max = df_sem_pit['LapTime (s)'].max()
    ax.set_ylim(bottom=df_sem_pit['LapTime (s)'].min() * 0.99, top=y_max * 1.05)

    for i, composto in enumerate(sorted(df_sem_pit['Compound'].unique()), start=1):
        if composto in medianas.index:
            mediana = medianas.loc[composto]
            ax.text(i, mediana, f'{mediana:.2f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(facecolor='#e10600', alpha=0.6, edgecolor='none', pad=1))

    title = f'Average Lap Time by Compound - {race} ({year})' if race else 'Average Lap Time by Compound'
    ax.set_title(title)
    plt.suptitle('')
    ax.set_ylabel('LapTime (s)')
    return fig

def plot_tyrelife_pit(df):
    compostos_all = ['SOFT', 'MEDIUM', 'HARD', 'WET', 'INTERMEDIATE']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for i, composto in enumerate(compostos_all):
        ax = axes_flat[i]
        temp = df[df['Compound'] == composto]
        if not temp.empty:
            temp.boxplot(column='TyreLife', by='PitNextLap', showfliers=False, ax=ax, patch_artist=True,
                        boxprops=dict(facecolor='#2d2d2d', color='#d1d1d1'),
                        medianprops=dict(color='#e10600', linewidth=2))
            med_pit = temp.groupby('PitNextLap')['TyreLife'].median()
            for pit, m in med_pit.items():
                x_pos = list(sorted(med_pit.index)).index(pit) + 1
                ax.text(x_pos, m, f'{m:.1f}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
        ax.set_title(composto)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Remove o último eixo que sobra (since 2x3=6 and we have 5 compounds)
    fig.delaxes(axes_flat[-1])
    plt.tight_layout()
    return fig

def plot_efficiency_year(df):
    resultado = df.groupby('Driver').mean(numeric_only=True).dropna()
    if len(resultado) < 2: return None

    x = resultado['TyreLife_Relative']
    y = resultado['Tyre_Efficiency']
    if x.empty or y.empty: return None

    try:
        coef = np.polyfit(x, y, 1)
        reta = np.poly1d(coef)
        x_linha = np.linspace(x.min(), x.max(), 100)
        r2 = r2_score(y, reta(x))
    except (ValueError, TypeError, np.RankWarning):
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, s=70, color='#d1d1d1')
    ax.plot(x_linha, reta(x_linha), linewidth=3, color='#e10600', label=f'R² = {r2:.3f}')
    for piloto in resultado.index:
        ax.annotate(piloto, (resultado.loc[piloto, 'TyreLife_Relative'], resultado.loc[piloto, 'Tyre_Efficiency']), fontsize=8, color='white')
    ax.axvline(0, linestyle='--', color='gray')
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_xlabel('Average TyreLife Relative')
    ax.set_ylabel('Average Tyre Efficiency')
    ax.legend()
    return fig

def plot_conservation_gain_year(df):
    tyre = df.groupby(['Year', 'Race', 'Driver'])['TyreLife_Relative'].mean().reset_index().groupby('Driver')['TyreLife_Relative'].mean()
    pos_gain = df.groupby(['Year', 'Race', 'Driver'])['Total_Position_Gain'].last().reset_index().groupby('Driver')['Total_Position_Gain'].mean()
    resultado = pd.concat([tyre, pos_gain], axis=1).dropna()
    resultado.columns = ['TyreLife_Relative', 'Avg_Total_Position_Gain']

    if resultado.empty: return None
    q1 = resultado.quantile(0.01)
    q99 = resultado.quantile(0.99)
    resultado = resultado[(resultado['TyreLife_Relative'] >= q1['TyreLife_Relative']) & (resultado['TyreLife_Relative'] <= q99['TyreLife_Relative']) &
                         (resultado['Avg_Total_Position_Gain'] >= q1['Avg_Total_Position_Gain']) & (resultado['Avg_Total_Position_Gain'] <= q99['Avg_Total_Position_Gain'])]

    if resultado.empty or len(resultado) < 2: return None
    x, y = resultado['TyreLife_Relative'], resultado['Avg_Total_Position_Gain']
    try:
        coef = np.polyfit(x, y, 1)
        reta = np.poly1d(coef)
        x_linha = np.linspace(x.min(), x.max(), 100)
        r2 = r2_score(y, reta(x))
    except (ValueError, TypeError, np.RankWarning):
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, s=80, color='#d1d1d1')
    ax.plot(x_linha, reta(x_linha), linewidth=3, color='#e10600', label=f'R² = {r2:.3f}')
    for piloto in resultado.index:
        ax.annotate(piloto, (resultado.loc[piloto, 'TyreLife_Relative'], resultado.loc[piloto, 'Avg_Total_Position_Gain']), fontsize=8, color='white')
    ax.axvline(0, color='white', linestyle='--')
    ax.set_xlabel('Average TyreLife Relative')
    ax.set_ylabel('Average Position Gain')
    ax.legend()
    return fig

def plot_pace_result_year(df):
    ritmo = df.groupby('Driver')['Lap_vs_Race_Avg'].mean()
    pos_final = df.groupby(['Year', 'Race', 'Driver'])['Position'].last().reset_index().groupby('Driver')['Position'].mean()
    resultado = pd.concat([ritmo, pos_final], axis=1).dropna()
    resultado.columns = ['Lap_vs_Race_Avg', 'Avg_Final_Position']

    if resultado.empty or len(resultado) < 2: return None
    x, y = resultado['Lap_vs_Race_Avg'], resultado['Avg_Final_Position']
    try:
        coef = np.polyfit(x, y, 1)
        reta = np.poly1d(coef)
        x_linha = np.linspace(x.min(), x.max(), 100)
        r2 = r2_score(y, reta(x))
    except (ValueError, TypeError, np.RankWarning):
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, color='#d1d1d1')
    ax.plot(x_linha, reta(x_linha), linewidth=3, color='#e10600', label=f'R² = {r2:.3f}')
    for piloto in resultado.index:
        ax.annotate(piloto, (resultado.loc[piloto, 'Lap_vs_Race_Avg'], resultado.loc[piloto, 'Avg_Final_Position']), fontsize=8, color='white')
    ax.invert_yaxis()
    ax.set_xlabel('Lap vs Race Avg')
    ax.set_ylabel('Average Final Position')
    ax.legend()
    return fig

def plot_podiums_perf_year(df):
    podio = df.groupby(['Year', 'Race', 'Driver']).agg({'Position': 'last', 'Position_Change': 'sum', 'Lap_vs_Race_Avg': 'mean'}).reset_index()
    podio = podio[podio['Position'] <= 3]
    if podio.empty: return None

    qtd_podios = podio.groupby('Driver').size().rename('Qtd_Podios')
    ganho_medio = podio.groupby('Driver')['Position_Change'].mean().rename('Pos_Ganhas')
    lap_perf = podio.groupby('Driver')['Lap_vs_Race_Avg'].mean().rename('Lap_vs_Race_Avg')
    res_pod = pd.concat([ganho_medio, qtd_podios, lap_perf], axis=1).dropna().sort_values('Qtd_Podios', ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    res_pod['Qtd_Podios'].plot(kind='bar', ax=ax1, alpha=0.7, color='#444444')
    ax1.set_ylabel('Number of Podiums')
    for i, valor in enumerate(res_pod['Qtd_Podios']):
        ax1.text(i, valor + 0.1, str(int(valor)), ha='center', fontsize=10, fontweight='bold', color='white',
                 bbox=dict(facecolor='#444444', alpha=0.7, edgecolor='none', pad=1))

    ax2 = ax1.twinx()
    ax2.plot(range(len(res_pod)), res_pod['Pos_Ganhas'], marker='o', color='#e10600', label='Positions Gained', linewidth=3)
    ax2.plot(range(len(res_pod)), res_pod['Lap_vs_Race_Avg'], marker='s', color='#ffffff', label='Lap vs Race Avg', linewidth=3)
    ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Performance')
    for i, valor in enumerate(res_pod['Pos_Ganhas']):
        ax2.annotate(f'{valor:.1f}', (i, valor), textcoords='offset points', xytext=(0, 10), ha='center',
                     fontsize=9, color='#e10600', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    for i, valor in enumerate(res_pod['Lap_vs_Race_Avg']):
        ax2.annotate(f'{valor:.3f}', (i, valor), textcoords='offset points', xytext=(0, -15), ha='center',
                     fontsize=9, color='white', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    ax2.legend(loc='upper right')
    plt.tight_layout()
    return fig

def get_best_overcomer(df):
    # Mesma lógica do gráfico de pódios
    podio = df.groupby(['Year', 'Race', 'Driver']).agg({'Position': 'last', 'Position_Change': 'sum'}).reset_index()
    podio = podio[podio['Position'] <= 3]

    if podio.empty:
        return None, None

    # Agregações por piloto
    res = podio.groupby('Driver').agg({'Position': 'size', 'Position_Change': 'mean'})
    res.columns = ['Qtd_Podios', 'Pos_Ganhas']

    # Cálculo do Índice de Superação: Podios * (1 + Ganho Médio)
    res['Score'] = res['Qtd_Podios'] * (1 + res['Pos_Ganhas'])

    best_driver = res['Score'].idxmax()
    best_score = res.loc[best_driver, 'Score']

    return best_driver, best_score

def plot_consistency_year(df, ano=None):
    if ano and ano != "All":
        df_ano = df[df['Year'] == ano]
    else:
        df_ano = df

    if df_ano.empty:
        return None

    consistencia = df_ano.groupby('Driver')['Lap_vs_Race_Avg'].std()
    pos_media = df_ano.groupby(['Year', 'Race', 'Driver'])['Position'].last().reset_index().groupby('Driver')['Position'].mean()
    res = pd.concat([consistencia, pos_media], axis=1).dropna()
    res.columns = ['Std_Lap_vs_Race_Avg', 'Avg_Final_Position']

    Q1, Q3 = res['Std_Lap_vs_Race_Avg'].quantile(0.25), res['Std_Lap_vs_Race_Avg'].quantile(0.75)
    IQR = Q3 - Q1
    res = res[(res['Std_Lap_vs_Race_Avg'] >= Q1 - 1.5*IQR) & (res['Std_Lap_vs_Race_Avg'] <= Q3 + 1.5*IQR)]
    if res.empty or len(res) < 2: return None

    x, y = res['Std_Lap_vs_Race_Avg'], res['Avg_Final_Position']
    try:
        coef = np.polyfit(x, y, 1)
        reta = np.poly1d(coef)
        x_linha = np.linspace(x.min(), x.max(), 100)
        r2 = r2_score(y, reta(x))
    except (ValueError, TypeError, np.RankWarning):
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, s=80, color='#d1d1d1')
    ax.plot(x_linha, reta(x_linha), linewidth=3, color='#e10600', label=f'R² = {r2:.3f}')
    for piloto in res.index:
        ax.annotate(piloto, (res.loc[piloto, 'Std_Lap_vs_Race_Avg'], res.loc[piloto, 'Avg_Final_Position']), fontsize=8, color='white')
    ax.invert_yaxis()
    ax.set_xlabel('Consistency Standard Deviation')
    ax.set_ylabel('Average Final Position')
    ax.legend()
    return fig

def plot_pace_evolution(df):
    df_temp = df.copy()
    df_temp['Faixa'] = pd.cut(df_temp['RaceProgress'], bins=25)
    media = df_temp.groupby(['Driver', 'Faixa'], observed=False)['Lap_vs_Race_Avg'].mean().reset_index().groupby('Faixa', observed=False)['Lap_vs_Race_Avg'].mean()
    x = [f.mid for f in media.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, media.values, marker='o', linewidth=3, color='#e10600')
    ax.axhline(y=0, linestyle='--', color='white', alpha=0.5)
    ax.set_xlabel('Race Progress (%)')
    ax.set_ylabel('Average Lap vs Race Avg (s)')
    return fig

def plot_pit_comparison(df):
    antes_pit = df[df['PitNextLap'] == 1]['LapTime (s)']
    pit = df[df['PitStop'] == 1]['LapTime (s)']
    df_temp = df.copy()
    df_temp['PitStop_Anterior'] = df_temp.groupby(['Race','Year','Driver'])['PitStop'].shift(1)
    depois_pit = df_temp[df_temp['PitStop_Anterior'] == 1]['LapTime (s)']

    valores = [antes_pit.mean(), pit.mean(), depois_pit.mean()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['Before Pit', 'In Pit', 'After Pit'], valores, color=['#444444', '#e10600', '#d1d1d1'])
    ax.set_ylabel('Average Lap Time (s)')
    for i, v in enumerate(valores):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', color='white')
    return fig

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main App Background */
        .stApp {
            background-color: #1f1f1f;
            color: #ffffff;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 3px solid #e10600;
        }

        /* Header Styling */
        h1, h2, h3 {
            color: #ffffff !important;
            font-family: 'Titillium Web', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Custom KPI Card Styling */
        .kpi-card {
            background-color: #2d2d2d;
            border-left: 5px solid #e10600;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
            transition: transform 0.2s;
            margin-bottom: 10px;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            background-color: #3d3d3d;
        }
        .kpi-label {
            color: #d1d1d1;
            font-size: 14px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .kpi-value {
            color: #ffffff;
            font-size: 28px;
            font-weight: bold;
        }

        /* Button and Input Styling */
        .stButton>button {
            background-color: #e10600 !important;
            color: white !important;
            border-radius: 0px !important;
            font-weight: bold;
            border: none;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d;
            color: white;
            border-radius: 4px 4px 0 0;
            border-bottom: 2px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e10600 !important;
            color: white !important;
            border-bottom: 2px solid white !important;
        }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------
# INTERFACE STREAMLIT
# -----------------------------
def render_kpi_cards(df):
    # Calculate metrics
    metrics = {
        "Drivers": df['Driver'].nunique(),
        "Races": df['Race'].nunique(),
        "Compounds": df['Compound'].nunique(),
        "Avg LapTime": f"{df['LapTime (s)'].mean():.2f}s"
    }

    cols = st.columns(len(metrics))
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)

def render_details_menu(df):
    with st.expander("🔍 View Filter Names (Drivers, Races and Compounds)"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏎️ Drivers**")
            pilotos = sorted(df['Driver'].unique().tolist())
            st.write(f"- {', '.join(pilotos)}" if pilotos else "No drivers found")

        with col2:
            st.markdown("**🏁 Races**")
            corridas = sorted(df['Race'].unique().tolist())
            st.write(f"- {', '.join(corridas)}" if corridas else "No races found")

        with col3:
            st.markdown("**🛞 Compounds**")
            compostos = sorted(df['Compound'].unique().tolist())
            st.write(f"- {', '.join(compostos)}" if compostos else "No compounds found")

def run_streamlit():
    st.set_page_config(page_title="F1 Performance Dashboard", layout="wide")
    inject_custom_css()

    # Logo F1 - Tenta carregar localmente, se não existir, tenta a URL
    try:
        st.image("f1_logo.png", width=400)
    except:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Formula_1_logo.svg/1200px-Formula_1_logo.svg.png", width=400)

    # Header
    st.title("F1 Performance Dashboard")
    st.markdown("Telemetry visualization, tire degradation and driver performance analysis.")

    # Sidebar Filters
    st.sidebar.header("⚙️ Global Filters")

    @st.cache_data
    def get_data():
        return process_data()

    df_all = get_data()

    # Year Filter
    anos_opcoes = ["All"] + sorted(df_all['Year'].unique().tolist())
    ano_sel = st.sidebar.selectbox("Select Year", anos_opcoes, index=len(anos_opcoes)-1)

    # Race Filter (Dynamic)
    if ano_sel == "All":
        corridas_disponiveis = sorted(df_all['Race'].unique().tolist())
    else:
        corridas_disponiveis = sorted(df_all[df_all['Year'] == ano_sel]['Race'].unique().tolist())
    corridas_opcoes = ["Overall"] + corridas_disponiveis
    corrida_sel = st.sidebar.selectbox("Select Race", corridas_opcoes)

    # Global Filtering Logic
    if ano_sel == "All":
        if corrida_sel == "Overall":
            df_filtered = df_all.copy()
        else:
            df_filtered = df_all[df_all['Race'] == corrida_sel]
    else:
        if corrida_sel == "Overall":
            df_filtered = df_all[df_all['Year'] == ano_sel]
        else:
            df_filtered = df_all[(df_all['Year'] == ano_sel) & (df_all['Race'] == corrida_sel)]

    # KPI Section
    render_kpi_cards(df_filtered)
    render_details_menu(df_filtered)
    st.divider()

    def plot_centered(fig):
        if fig is None:
            return
        col1, col2, col3 = st.columns([1, 10, 1])
        with col2:
            st.pyplot(fig)

    tabs = st.tabs(["Tires & Degradation", "Performance & Pace", "Strategy & Podiums"])

    with tabs[0]:
        st.header("Tire Analysis")

        with st.expander("View Filtered Data"):
            st.dataframe(df_filtered[['Compound', 'TyreLife', 'LapTime (s)', 'Cumulative_Degradation', 'Tyre_Usage_Percent']], use_container_width=True)

        # Filtro individual para cada gráfico que usa compostos
        st.subheader("Performance Loss by Tyre use")
        st.caption("Analyzes lap time variation as tire usage percentage increases.")
        comp_usage = st.multiselect("Filter Compounds (Performance)", sorted(df_filtered['Compound'].unique().tolist()), default=sorted(df_filtered['Compound'].unique().tolist()), key="f_usage")
        plot_centered(plot_tire_usage(df_filtered, comp_usage))

        st.subheader("Cumulative Degradation Impact")
        st.caption("Shows how cumulative degradation affects driver pace relative to the average.")
        comp_deg = st.multiselect("Filter Compounds (Degradation)", sorted(df_filtered['Compound'].unique().tolist()), default=sorted(df_filtered['Compound'].unique().tolist()), key="f_deg")
        plot_centered(plot_degradation_impact(df_filtered, comp_deg))

        st.subheader("Cumulative Degradation by Tire Life")
        st.caption("Analyzes the evolution of average tire degradation as lap count increases.")
        comp_cum = st.multiselect("Filter Compounds (Cumulative Degradation)", sorted(df_filtered['Compound'].unique().tolist()), default=sorted(df_filtered['Compound'].unique().tolist()), key="f_cum")
        plot_centered(plot_cumulative_degradation(df_filtered, comp_cum))

        st.subheader("Average Lap Time by Compound")
        st.caption("Compares lap time distribution across different tire compounds.")
        comp_avg = st.multiselect("Filter Compounds (Average Time)", sorted(df_filtered['Compound'].unique().tolist()), default=sorted(df_filtered['Compound'].unique().tolist()), key="f_avg")
        plot_centered(plot_avg_laptime_compound(df_filtered, corrida_sel, ano_sel, comp_avg))

        st.subheader("TyreLife by PitNextLap and Compound")
        st.caption("Analyzes tire life at the moment the driver decides to enter the pit stop.")
        plot_centered(plot_tyrelife_pit(df_filtered))

    with tabs[1]:
        st.header("Performance & Pace")
        with st.expander("View Filtered Data"):
            st.dataframe(df_filtered[['Driver', 'Year', 'Lap_vs_Driver_Avg', 'Lap_vs_Race_Avg', 'TyreLife_Relative', 'Tyre_Efficiency']], use_container_width=True)

        st.subheader("Average Pace Evolution")
        st.caption("Tracks the variation of average pace for all drivers throughout the race.")
        plot_centered(plot_pace_evolution(df_filtered))

        st.subheader("TyreLife Relative vs Tyre Efficiency")
        st.caption("Evaluates the relationship between tire conservation and each driver's tire efficiency.")
        fig_eff = plot_efficiency_year(df_filtered)
        if fig_eff: plot_centered(fig_eff)

        st.subheader("Consistency vs Final Result")
        st.caption("Measures the correlation between pace stability and average final position.")
        fig_cons = plot_consistency_year(df_filtered, ano_sel)
        if fig_cons: plot_centered(fig_cons)

        

    with tabs[2]:
        st.header("Strategy & Results")
        with st.expander("View Filtered Data"):
            st.dataframe(df_filtered[['Driver', 'Year', 'Position', 'Position_Change', 'Total_Position_Gain', 'PitStop', 'PitNextLap']], use_container_width=True)

        st.subheader("Time Before vs After Pit")
        st.caption("Compares lap times immediately before, during and after a pit stop.")
        plot_centered(plot_pit_comparison(df_filtered))

        st.subheader("Tire Conservation vs Position Gain")
        st.caption("Measures the correlation between tire conservation ability and position gain.")
        fig_gain = plot_conservation_gain_year(df_filtered)
        if fig_gain: plot_centered(fig_gain)

        st.subheader("Relative Pace vs Final Result")
        st.caption("Compares driver average pace with average final position achieved.")
        fig_res = plot_pace_result_year(df_filtered)
        if fig_res: plot_centered(fig_res)

        st.subheader("Podiums vs Performance vs Pace")
        st.caption("Cross-references podium count with position gain and pace performance.")
        fig_pod = plot_podiums_perf_year(df_filtered)
        if fig_pod:
            plot_centered(fig_pod)
            # Botão para calcular o piloto que mais se supera
            if st.button("🏆 Who Overcame the Most?"):
                driver, score = get_best_overcomer(df_filtered)
                if driver:
                    st.success(f"**The driver who overcame the most was {driver}!** \n\nOvercoming Score: `{score:.2f}`")
                else:
                    st.warning("Not enough podium data to calculate overcoming.")
            st.caption("Calculation: Podiums × (1 + Avg Positions Gained). Values drivers who reach the podium by recovering positions.")

        

# Launch Streamlit Interface
if st.button("Load Interface") or True: # The 'or True' ensures it runs on startup
    run_streamlit()
