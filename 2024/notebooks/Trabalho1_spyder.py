# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:54:03 2024

@author: 
    José Henrique Hess
    Researcher in LCQAr - Air Quality Control Laboratory
    Student in Sanitary and Environmental Engineering
    Federal University of Santa Catarina
    
"""

#%% Célula de importações

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymannkendall as mk
from scipy.stats import theilslopes as ts

#%% Manipulação dos dados

df_MG1 = pd.read_excel(r"C:\PosGraduacao\ENS410064\dados\aula01\MG.xlsx")
df_MG2 = pd.read_excel(r"C:\PosGraduacao\ENS410064\dados\aula01\MG2.xlsx")
df_MG = pd.concat([df_MG1,df_MG2], axis=0)
df_MG.reset_index(drop=True)

#%% Criando dataframe para as estações em Congonhas

# Criando a lista das estações de Congonhas
lista_estacoes = ['Motas',
                  'Pires',
                  'Novo_Plataforma',
                  'Matriz',
                  'Basílica',
                  'Jardim_Profeta',
                  'Lobo_Leite']

# Criando o dataframe com as estações de Congonhas
df_Congonhas = df_MG[df_MG['Estacao'].isin(lista_estacoes)].reset_index(drop=True)

# Salvando o dataframe de Congonhas em .csv
df_Congonhas.to_csv(r'C:\PosGraduacao\ENS410064\2024\dados\Congonhas.csv')

df_Congonhas = pd.read_csv(r'C:\PosGraduacao\ENS410064\2024\dados\Congonhas.csv')
df_Congonhas = df_Congonhas.drop(columns=['Unnamed: 0'])

#%% Verificando tipos de poluentes pra cada estação

# Criando uma lista com todos os poluentes do dataframe
poluentes = df_Congonhas['Poluente'].unique().tolist()

# Criando lista de resoluções

lista_resolucoes = ['Hora',
                    'Dia',
                    'Mês',
                    'Estação',
                    'Ano']

#%% Criar um index datetime para o dataframe

df_Congonhas['datetime'] = pd.to_datetime(df_Congonhas.rename
                            (columns={'Ano': 'year', 'Mes': 'month', 'Dia': 'day', 'Hora': 'hour', 'Minuto': 'minute'})
                            [['year', 'month', 'day', 'hour', 'minute']])

df_Congonhas.set_index('datetime', inplace=True)

#%% Análise da série temporal com lineplot

def AnaliseSerieTemporalLineplot(df: pd.DataFrame(),estacao: str,poluente: str, resolucao: str):
    '''
    
    Esta função cria uma análise temporal de um dataframe com um lineplot
    Este dataframe deve ter uma estação que avalia a concentração de um poluente

    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe com os valores de concentração dos poluentes em determinada estação 
    estacao : str
        Nome da estação 
    poluente : str
        Nome do poluente
    resolucao : str
        Resolução do tempo, se horário, diário, mensal ou estação

    Returns
    -------
    None.

    '''

    # Cria um dataframe apenas com a estação e poluente selecionados
    df = df[(df['Estacao'] == estacao) & (df['Poluente'] == poluente)]

    if len(df) == 0: # Verifica se existe o poluente naquela estação
        
        print('Não há o poluente ' + poluente + ' para a estação ' + estacao)     
    
    else:
    
        if resolucao != 'Hora':
            if resolucao == 'Dia':
                estatisticas_dia = df.resample('D').agg({'Valor': ['mean', 'min', 'max']})
                df = df[df.index.hour == 0].copy() # Transformando o df em um tamanho compatível as estatísticas
                
            elif resolucao == 'Mês':
                estatisticas_dia = df.resample('M').agg({'Valor': ['mean', 'min', 'max']})
                df = df[(df.index.day == 1) & (df.index.hour == 0)].copy() # Transformando o df em um tamanho compatível as estatísticas
            
            else:
                
                falha = 'Resolução temporal não permitida, tente Hora, Dia ou Mês'
                
                return falha
    
            # Adicionando as colunas de média, mínimo e máximo de cada dia ao DataFrame original
            df['Valor'] = estatisticas_dia['Valor']['mean'].tolist()
            df['Minimo'] = estatisticas_dia['Valor']['min'].tolist()
            df['Maximo'] = estatisticas_dia['Valor']['max'].tolist()
        
        fig, ax = plt.subplots() # Cria uma figura para plotar
        
        if resolucao != 'Hora':
            ax.fill_between(df.index, df.Minimo, df.Maximo, alpha = 0.2, color='yellow', label='Concentração mínima à máxima')
        
        ax.plot(df.index, df.Valor, color = 'orange', label = 'Concentração média') # Cria um lineplot das concentrações

        ax.scatter(df.index[df.Valor.isna()],
                   np.zeros(np.sum(df.Valor.isna())),
                   color='red', 
                   label = 'Dados faltantes: ' + str(np.sum(df.Valor.isna())) + ' (' + str(round(100*np.sum(df.Valor.isna()) / len(df), 2)) + '%)') # Identifica os lugares sem dados
        
        for y in range(25, 1601, 25):
            ax.hlines(
                y=y, xmin=df.index.min(), xmax=df.index.max(),
                color='black' if y % 100 == 0 else 'gray',
                linestyles='--',
                linewidth=1 if y % 100 == 0 else 0.75,
                alpha = 0.5
            )
            
        ax.set_xlim(df.index.min() - pd.Timedelta(minutes=30),df.index.max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        print(df.drop(columns=['Ano']).apply(pd.to_numeric, errors='coerce').max().max()*1.2)
        
        ax.set_ylim(0, df.drop(columns=['Ano','Mes', 'Dia', 'Hora', 'Minuto']).apply(pd.to_numeric, errors='coerce').max().max()*1.2) 
        
        ax.set_xlabel('Data no ano de ' + str(df.Ano[0]) + ' (' + resolucao + ')')
        ax.set_ylabel('Concentração de ' + poluente + ' (' + df.Unidade[0] + ')\nna estação ' + estacao)

        ax.legend(fontsize=7, loc='upper left')
        
        fig.savefig(r'C:\PosGraduacao\ENS410064\2024\figuras\LinePlots\LinePlot_' + poluente + '_' + resolucao + '_' + estacao + '.png')

lineplot = AnaliseSerieTemporalLineplot(df_Congonhas,
                                        lista_estacoes[6],
                                        poluentes[0],
                                        lista_resolucoes[1])

#%% Análise da série temporal através de um boxplot

def AnaliseSerieTemporalBoxplot(df: pd.DataFrame(),estacao: str,poluente: str, resolucao: str):
    '''
    
    Esta função cria uma análise temporal de um dataframe com um boxplot
    Este dataframe deve ter uma estação que avalia a concentração de um poluente

    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe com os valores de concentração dos poluentes em determinada estação 
    estacao : str
        Nome da estação 
    poluente : str
        Nome do poluente
    resolucao : str
        Resolução do tempo, se horário, diário, mensal ou estação

    Returns
    -------
    None.

    '''
    
    # Cria um dataframe apenas com a estação e poluente selecionados
    df = df[(df['Estacao'] == estacao) & (df['Poluente'] == poluente)]

    if len(df) == 0: # Verifica se existe o poluente naquela estação
        
        print('Não há o poluente ' + poluente + ' para a estação ' + estacao)     
    
    else:       
    
        if resolucao == 'Hora':
            data_nome = df.Hora
            
        elif resolucao == 'Dia':
            data_nome = df.index.day_name(locale='pt_BR').str[:3]
            
        elif resolucao == 'Mês':
            data_nome = df.index.month_name(locale='pt_BR').str[:3]
            
        elif resolucao == 'Estação':
            
            df['Estacao'] = pd.cut(df['Mes'], 
                                   bins=[-float('inf'), 3, 6, 9, float('inf')], 
                                   labels=['Verão', 'Outono', 'Inverno', 'Primavera'])
        
            data_nome = df.Estacao
            
        elif resolucao == 'Ano':
            data_nome = df.Ano
            
        else:
            
            falha = 'Resolução temporal não permitida, tente Hora, Dia, Mês ou Ano'
            
            return falha
    
        # Criando uma lista de valores para cada mês
        valores_tempo = [df[data_nome == res].Valor.dropna() 
                         for res in data_nome.unique()]
        
        # Criando o boxplot
        fig, ax = plt.subplots()
        ax.boxplot(valores_tempo, tick_labels=data_nome.unique(), 
            patch_artist=True,  # Habilita o preenchimento do box
            boxprops=dict(facecolor=(1, 1, 0, 0.2), edgecolor=(0, 0, 0, 1)), 
            medianprops=dict(color='orange') # Cor da mediana
            )
        
        for y in range(25, 1301, 25):
            ax.axhline(
                y=y,
                linestyle='--',
                color='black' if y % 100 == 0 else 'gray',
                linewidth=1 if y % 100 == 0 else 0.75,
                alpha = 0.5
            )
        
        ax.set_ylim(0, df.drop(columns=['Ano','Mes', 'Dia', 'Hora', 'Minuto']).apply(pd.to_numeric, errors='coerce').max().max()*1.2) 
        
        ax.set_xlabel('Tempo (' + resolucao + ')')
        ax.set_ylabel('Concentração de ' + poluente + ' (' + df.Unidade[0] + ')\nna estação ' + estacao)

        fig.savefig(r'C:\PosGraduacao\ENS410064\2024\figuras\BoxPlots\BoxPlot_' + poluente + '_' + resolucao + '_' + estacao + '.png')

boxplot = AnaliseSerieTemporalBoxplot(df_Congonhas,
                                        lista_estacoes[2],
                                        poluentes[2],
                                        lista_resolucoes[4])

#%% Estatísticas univariadas

def EstatisticasUnivariadas(df: pd.DataFrame(),estacao: str,poluente: str, resolucao: str):
    '''
    
    Esta função salva um csv para uma determinada estação, resolução temporal e poluição com 
    média, mediana, moda, mínimo, percentil 25%, percentil 75%, máximo, desvio padrão, variância

    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe com os valores de concentração dos poluentes em determinada estação 
    estacao : str
        Nome da estação 
    poluente : str
        Nome do poluente
    resolucao : str
        Resolução do tempo, se horário, diário, mensal ou estação

    Returns
    -------
    None.

    '''
    
    # Cria um dataframe apenas com a estação e poluente selecionados
    df = df[(df['Estacao'] == estacao) & (df['Poluente'] == poluente)]
    
    if len(df) == 0: # Verifica se existe o poluente naquela estação
        
        print('Não há o poluente ' + poluente + ' para a estação ' + estacao)     
    
    else: 
        
        if resolucao == 'Hora':
            estatistica = df.groupby(df.index.hour).Valor
            tempo = df.Hora.unique().tolist()
            
        elif resolucao == 'Dia':
            estatistica = df.groupby(df.index.dayofweek).Valor
            tempo = ['Dom', 'Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb']
            
        elif resolucao == 'Mês':
            estatistica = df.groupby(df.index.month).Valor
            tempo = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            
        elif resolucao == 'Estação':
            df['Estacao'] = pd.cut(df['Mes'], 
                                bins=[-float('inf'), 3, 6, 9, float('inf')], 
                                labels=['Verao', 'Primavera', 'Inverno', 'Outono'])
         
            estatistica = df.groupby(df.Estacao).Valor
            tempo = ['Verão', 'Outono', 'Inverno', 'Primavera']
        
        elif resolucao == 'Ano':
            estatistica = df.groupby(df.index.year).Valor
            tempo = df.Ano.unique().tolist()
    
        else:
            
            falha = 'Resolução temporal não permitida, tente Hora, Dia, Mês, Estação ou Ano'
            
            return falha
    
        stats = {
            'Media': estatistica.mean(),
            'Mediana': estatistica.median(),
            'Minimo': estatistica.min(),
            'Percentil25': estatistica.quantile(0.25),
            'Percentil75': estatistica.quantile(0.75),
            'Maximo': estatistica.max(),
            'Desvio Padrao': estatistica.std(),
            'Variancia': estatistica.var()
        }
        
        # Criando o DataFrame
        df_stats = pd.DataFrame(stats)
        df_stats.index = tempo
        
        df_stats.to_csv(r'C:\PosGraduacao\ENS410064\2024\tabelas\Trabalho1\EstatiscasUnivariadas_' + poluente + '_' + resolucao + '_' + estacao + '.csv')
        
EstatisticasUnivariadas(df_Congonhas,
                        lista_estacoes[2],
                        poluentes[2],
                        lista_resolucoes[4])

#%% Sazonalidade
        
def Sazonalidade(df: pd.DataFrame(),estacao: str,poluente: str):
    '''
    
    Esta função estima a sazonalidade de um determinado poluente em uma estação

    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe com os valores de concentração dos poluentes em determinada estação 
    estacao : str
        Nome da estação 
    poluente : str
        Nome do poluente
        
    Returns
    -------
    None.

    '''
    
    # Cria um dataframe apenas com a estação e poluente selecionados
    df = df[(df['Estacao'] == estacao) & (df['Poluente'] == poluente)]
    
    if len(df) == 0: # Verifica se existe o poluente naquela estação
        
        print('Não há o poluente ' + poluente + ' para a estação ' + estacao) 
    
    else: 

        result = mk.original_test(df.Valor)
        print(result.trend)
        
        dict_resultados = {
            'Indice': poluente + '_' + estacao,
            'Tendência': result.trend,
            'p_valor': result.p,
            'Significativo?': "Sim" if result.h == 1 else "Não",
            'Tau': result.Tau
            }
            
        return dict_resultados

df_saz = pd.DataFrame(columns=[
    'Indice',
    'Tendência',
    'p_valor',
    'Significativo?',
    'Tau'])
    
for pol in poluentes:
    for est in lista_estacoes:
        saz = Sazonalidade(df_Congonhas, est, pol)
        if saz != None:
            df_saz = pd.concat([df_saz, pd.DataFrame([saz])])
            
df_saz.index = df_saz.Indice
df_saz = df_saz.drop(columns = 'Indice')
            
df_saz.to_csv(r'C:\PosGraduacao\ENS410064\2024\tabelas\Trabalho1\Sazonalidade.csv')

#%% Tendência nos dados

def Tendencia(df: pd.DataFrame(),estacao: str,poluente: str, resolucao: str):
    '''
    
    Esta função estima a tendência de um determinado poluente em uma estação

    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe com os valores de concentração dos poluentes em determinada estação 
    estacao : str
        Nome da estação 
    poluente : str
        Nome do poluente
    resolucao : str
        Resolução do tempo, se horário, diário, mensal ou estação

    Returns
    -------
    None.

    '''

    # Cria um dataframe apenas com a estação e poluente selecionados
    df = df[(df['Estacao'] == estacao) & (df['Poluente'] == poluente)]
    
    if len(df) == 0: # Verifica se existe o poluente naquela estação
        
        print('Não há o poluente ' + poluente + ' para a estação ' + estacao) 
        
        return 0
    
    else: 
        
        if resolucao == 'Hora':
            df = df

        elif resolucao == 'Dia':
            df = df.resample('D').mean()

        elif resolucao == 'Mes':    
            df = df.resample('M').mean()   
        
        else:
            
            print('Resolução temporal não permitida, tente Hora, Dia ou Mês')
            
            return 0

        # Aplicando o método Thiel-Sen
        slope, intercept, lower, upper = ts(df['Valor'])

        dict_resultados = {
            'Indice': poluente + '_' + estacao,
            'Inclinação': slope,
            'Intercept': intercept,
            'Menor': lower,
            'Maior': upper
            }
        
        return dict_resultados

for tempo in lista_resolucoes:

    df_ten = pd.DataFrame(columns=[
        'Indice',
        'Inclinação',
        'Intercept',
        'Menor',
        'Maior'])

    for pol in poluentes:
        for est in lista_estacoes:
            ten = Tendencia(df_Congonhas, est, pol, tempo)
            if ten != 0:
                df_ten = pd.concat([df_ten, pd.DataFrame([ten])])
    
    print(1111111111111111111111)
    
    df_ten.index = df_ten.Indice
    df_ten = df_ten.drop(columns = 'Indice')
                
    df_ten.to_csv(r'C:\PosGraduacao\ENS410064\2024\tabelas\Trabalho1\Tendencia_' + tempo + '.csv')

#%%

for pol in poluentes:
    for est in lista_estacoes:
        for tempo in lista_resolucoes:
            AnaliseSerieTemporalLineplot(df_Congonhas, est, pol, tempo)
            AnaliseSerieTemporalBoxplot(df_Congonhas, est, pol, tempo)
            EstatisticasUnivariadas(df_Congonhas, est, pol, tempo)
        
#%%



#%%



#%%