"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location. There are two callbacks,
one uses the current location to render the appropriate page content, the other
uses the current location to toggle the "active" properties of the navigation
links.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table
import dash_daq as daq

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

df = px.data.iris() # iris is a pandas DataFrame
fig = px.scatter(df, x="sepal_width", y="sepal_length")


# the style arguments for the sidebar. We use position:fixed and a fixed width
HEADER_STYLE = {
    "position":"fixed",
    "top":0,
    'left':0,
    'bottom':0,
    "background-color": "#f8f9fa"
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "32rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "top":0,
    "margin-left": "34rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

tab1_content = dbc.Card([
    dbc.CardBody(
        [
            html.Div([

                html.P(children='Number of dimensions:',
                       style={'width':'200px'}),

                daq.NumericInput(
                    value = 1,
                    min = 1,
                    max = 20,
                    style={'margin-left':'15px'}
                )

            ],className='row'),

            html.Div([

                html.P(children='Parameter 1:',
                       style={'width':'200px'}),

                daq.NumericInput(
                    value = 1,
                    min = 1,
                    max = 20,
                    style={'margin-left':'15px'}
                )

            ],className='row',
            style={
                'margin-top':'15px'
            })

        ]
    )
])
tab2_content = dbc.Card([
    dbc.CardBody(
        [
            html.P('Number of dimensions'),
            dcc.Slider(
                min=1,
                max=20,
                value=2,
                marks={1:'1',
                       5:'5',
                       10:'10',
                       15:'15',
                       20:'20'}
            )
        ]
    )
])
tab3_content = dbc.Card([
    dbc.CardBody(
        [
            html.P('This is t-SNE Algorithm')
        ]
    )
])

sidebar = html.Div(
    [
        html.Div([
           html.H2(children='Manifold Algorithms',
                   style={'text-align':'center'})
        ]),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    # 'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),

            # Hidden Div inside the app that stores the intermediate data
            html.Div(id='data-storage',
                     style={
                         'display': 'none'
                     })

        ]),
        html.Hr(),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='demo-dropdown',
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': 'Montreal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    multi=True,
                    value='NYC'
                )
            ]),
            html.Div([
                dash_table.DataTable(
                        id='data-table3',
                        data = df.to_dict('records'),
                        columns = [{'name': i, 'id': i} for i in df.columns],
                        style_table={'overflowX': 'scroll',
                                     'maxHeight': '350px',
                                     'overflowY': 'scroll',
                                     },
                        style_cell={
                            'minWidth': '100px',
                            'text-align': 'center',
                            'width': '30%'
                        }
                    )
            ],style={'margin-top':'15px'}),
            html.Hr(),
        ]),
        dbc.Tabs([
            dbc.Tab(tab1_content,label = 'UMAP'),
            dbc.Tab(tab2_content,label = 'PCA'),
            dbc.Tab(tab3_content,label = 't-SNE'),
            dbc.Tab(tab1_content,label = 'KPCA'),
            dbc.Tab(tab2_content,label = 'isoMAP'),
            dbc.Tab(tab3_content,label = 'CUSTOM'),
        ])
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content",
                   style=CONTENT_STYLE,
                   children=[
                       html.Div([
                           dcc.Graph(figure=fig)
                       ])
                   ])



app.layout = html.Div([dcc.Location(id="url"), sidebar, content])




if __name__ == "__main__":
    app.run_server(debug=True)
