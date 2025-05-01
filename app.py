from ucimlrepo import fetch_ucirepo
import plotly.express as px
from dash import Dash, dcc, html

heart_disease = fetch_ucirepo(id=45)
dados = heart_disease.data.features
#print(dados.head())

figura_histograma = px.histogram(dados, x='age', title='Histograma')
app = Dash(__name__)
app.layout = html.Div([
    html.H1('Análise de dados do UCI Repository Heart Disease'),
    dcc.Graph(figure=figura_histograma)])

app.run(debug=True, port=8051)