import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
file_path = './modelos_variancia_desvio_padrao_100_000.csv'
data = pd.read_csv(file_path)

# Ajustar os nomes das colunas
data.columns = ['Modelo', 'Acuracia_Media', 'Desvio_Padrao_Acuracia', 'Acuracia_Variancia']

# Converter strings de porcentagem para float
data['Acuracia_Media'] = data['Acuracia_Media'].str.replace('%', '').str.replace(',', '.').astype(float)
data['Desvio_Padrao_Acuracia'] = data['Desvio_Padrao_Acuracia'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
data['Acuracia_Variancia'] = data['Acuracia_Variancia'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)

# Criar o gráfico combinado com o eixo Y secundário
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plotar Acurácia Média como barras
bar_width = 0.4
bars = ax1.bar(data['Modelo'], data['Acuracia_Media'], bar_width, color='b', label='Acurácia Média')

# Definir rótulos e título para o eixo Y primário
ax1.set_xlabel('Modelos')
ax1.set_ylabel('Acurácia Média (%)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticklabels(data['Modelo'], rotation=45)

# Criar o eixo Y secundário para a variância
ax2 = ax1.twinx()
line = ax2.plot(data['Modelo'], data['Acuracia_Variancia'], color='r', marker='o', linestyle='-', label='Acurácia Variância')

# Definir rótulos e título para o eixo Y secundário
ax2.set_ylabel('Acurácia Variância (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adicionar legendas
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9), bbox_transform=ax1.transAxes)

plt.title('Acurácia Média e Variância da Acurácia por Modelo - 10.000 execuções')

# plt.show()
plt.savefig('variancia_100_000.png', bbox_inches='tight')