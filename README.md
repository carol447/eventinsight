# EventInsight

Projeto acadêmico de análise de dados e aprendizado de máquina para explorar fatores associados ao impacto econômico de festivais.

## Atividades e tecnologias

- Limpeza e análise exploratória com pandas, NumPy, Matplotlib e Seaborn.
- Pré-processamento com imputação, OneHotEncoder e StandardScaler.
- Comparação de regressão linear, árvore de decisão, Random Forest e Gradient Boosting.
- Seleção de atributos, GridSearchCV e validação cruzada.
- Avaliação por MAE, RMSE e R² e análise de tempo e memória.

## Execução

```bash
python -m pip install -r requirements.txt
python analysis_eventinsight.py
```

Forneça uma base autorizada em `inputs/baseinicial.csv`. A base original e os resultados não acompanham esta versão de portfólio. O script espera as colunas `Festival_Name`, `Location`, `Attendance_Numbers`, `Visitor_Demographics`, `Economic_Impact` e `Music_Genre`; o cabeçalho da fonte original estava deslocado e sua interpretação está documentada no código. Não use uma base com outra semântica sem adaptar `load_clean`.

O notebook `EventInsight_Trabalho_2.ipynb` está disponível sem saídas de execução. A interpretação das métricas depende da base e das limitações metodológicas; valores econômicos em moedas distintas exigem cuidado na comparação.
