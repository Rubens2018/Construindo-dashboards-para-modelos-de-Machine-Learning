from ucimlrepo import fetch_ucirepo
import plotly.express as px
from dash import Dash, dcc, html

heart_disease = fetch_ucirepo(id=45)
dados = heart_disease.data.features
#print(dados.head())

figura_histograma = px.histogram(dados, x='age', title='Histograma')
div_do_histograma = html.Div([
            html.H2('Histograma de idades'),
            dcc.Graph(figure=figura_histograma)
        ])

dados["doenca"] = (heart_disease.data.targets > 0) * 1

# boxplot das idades por doenca, colorido por doenca
figura_boxplot = px.box(dados, x='doenca', y = 'age', color='doenca', title='Boxplot de idades')

#figura_boxplot = px.box(dados, x='doenca', y = 'age', title='Boxplot de idades')
div_do_boxplot = html.Div([
            html.H2('Boxplot de idades'),
            dcc.Graph(figure=figura_boxplot)
        ])

app = Dash(__name__)
app.layout = html.Div([
        html.H1('Análise de dados do UCI Repository Heart Disease'),
        div_do_histograma,
        div_do_boxplot
])
app.run(debug=True, port=8051)