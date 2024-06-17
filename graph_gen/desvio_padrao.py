import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
file_path = './100_000.csv'
data = pd.read_csv(file_path)

# Limpeza e conversão de dados
data['Acurácia Média'] = data['Acurácia Média'].str.replace('%', '').str.replace(',', '.').astype(float)
data['Acurácia Desvio Padrão'] = data['Acurácia Desvio Padrão'].str.replace('%', '').str.replace(',', '.').astype(float)

# Extrair dados necessários
categories = data['Modelo']
values = data['Acurácia Média']
std_dev = data['Acurácia Desvio Padrão']

# Configuração do gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Criação das barras
bars = ax.bar(categories, values, yerr=std_dev, capsize=5, color='skyblue', edgecolor='black')

# Adição das linhas de desvio padrão
for bar, std in zip(bars, std_dev):
    height = bar.get_height()
    ax.plot([bar.get_x() + bar.get_width() / 2, bar.get_x() + bar.get_width() / 2],
            [height - std, height + std], color='black')

# Rótulos e título
ax.set_ylabel('Acurácia Média (%)')
ax.set_title('Acurácia por modelo - 10.000 execuções')
ax.set_xticks(np.arange(len(categories)))
ax.set_xticklabels(categories, rotation=45, ha='right')

# Mostrar o gráfico
plt.tight_layout()
# plt.show()
plt.savefig('./100_000.png')
