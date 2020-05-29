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
import time
import base64
import io
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_table
import dash_daq as daq
from flask_caching import Cache
import os

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
cache = Cache(app.server, config={
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR':'cache-directory'
})

df = px.data.iris() # iris is a pandas DataFrame
df = df.select_dtypes(['number'])
options = [{'label': i, 'value': i} for i in df.columns]
value = df.columns
fig = px.scatter(df, x="sepal_width", y="sepal_length")


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "40%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow-y":"scroll"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "top":0,
    "margin-left": "42%",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "overflow-y":"scroll"
}

collapse = html.Div(
    [
        dbc.Button(
            "Filter Data",
            id="collapse-button",
            className="mb-6",
            color="primary",
            block=True
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody(
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='dropDown',
                            multi=True,
                            options=options,
                            value=value
                        )
                    ]),
                    html.Div([
                        dcc.Loading(id='loading',
                            type='circle',
                            children=[
                                html.Div(
                                    id='data-table',
                                    children=[
                                        dbc.Table.from_dataframe(
                                            df=df,
                                            striped=True,
                                            bordered=True,
                                            hover=True,
                                            size='sm',
                                         )],
                                    style={'margin-top':'15px',
                                         'overflowX':'scroll',
                                         'overflowY':'scroll',
                                         'height':'450px'})
                        ])
                    ]),

                ]),
            )),
            id="collapse",
        ),
    ],style={'margin-top':'15px'}
)


tab_PCA = dbc.Card([
    dbc.CardBody(
        [
            html.Div([

                html.P(children='Number of dimensions:',
                       style={'width':'200px'}),

                daq.NumericInput(
                    id='PCA_NDimensions',
                    value = 3,
                    min = 1,
                    max = 20,
                    style={'margin-left':'15px'}
                )

            ],className='row'),

        ]
    )
])

tab_MDS = dbc.Card([
    dbc.CardBody(
        [
            html.Div([

                html.P(children='Number of dimensions:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='MDS_NDimensions',
                    value=1,
                    min=1,
                    max=20,
                    style={'margin-left': '15px'}
                )

            ], className='row'),

        ]
    )
])

tab_IsoMAP = dbc.Card([
    dbc.CardBody(
        [
            html.P('This is IsoMAP Algorithm')
        ]
    )
])

tab_LLE = dbc.Card([
    dbc.CardBody(
        [
            html.P('This is LLE Algorithm')
        ]
    )
])

tab_KPCA = dbc.Card([
    dbc.CardBody(
        [
            html.P('This is KPCA Algorithm')
        ]
    )
])

tab_tSNE = dbc.Card([
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
            html.Hr(),
            collapse,
            # Hidden Div inside the app that stores the intermediate data
            html.Div([
                dcc.Store(id='data-storage',
                          data=df.to_json(date_format='iso',orient = 'split'))
            ]),
            html.Div([
                dcc.Store(id='filtered-data-storage',
                          data=df.to_json(date_format='iso',orient = 'split'))
            ]),
            html.Div([
              dcc.Store(
                  id='manifold-data-storage'
              )
            ]),
            html.Div(id='prueba'),
            #html.Div(id='data-storage',
            #         style={
            #             'display': 'none'
            #         })
        ]),
        html.Hr(),


        dbc.Tabs(
            id='tabs',
            active_tab='PCA',
            children=[
                dbc.Tab(tab_PCA,label = 'PCA',tab_id='PCA'),
                dbc.Tab(tab_MDS,label = 'MDS',tab_id='MDS'),
                dbc.Tab(tab_IsoMAP,label = 'IsoMAP',tab_id='IsoMAP'),
                dbc.Tab(tab_LLE, label='LLE',tab_id='LLE'),
                dbc.Tab(tab_KPCA, label='KPCA',tab_id='KPCA'),
                dbc.Tab(tab_tSNE,label = 't-SNE',tab_id='t-SNE'),
            ]),

        html.Div([
                dbc.Button(id='Run_Button',
                           children='Run',
                           color='primary',
                           style={
                               'margin-left':'35%',
                               'width':'30%'
                           })
        ],style={
            'margin-top':'15px'
        }),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content",
                   style=CONTENT_STYLE,
                   children=[
                       html.Div([
                           html.P('Select graphic dimension:'),
                           html.P('2D',
                                  style={'margin-left':'15px'}),
                           daq.ToggleSwitch(
                               id='graphSwitch',
                               value=True
                           ),
                           html.P('3D'),
                       ], className='row'),
                       html.Div([
                           html.Div([
                               html.P('X')
                           ],style={'width':'30%',
                                    'text-align':'center'}),

                           html.Div([
                               html.P('Y')
                           ],style={'width':'30%',
                                    'text-align':'center',
                                    'margin-left':'15px'}),
                           html.Div([
                               html.P('Z')
                           ],style={'width':'30%',
                                    'text-align':'center',
                                    'margin-left':'15px'})
                       ], className='row'),
                       html.Div([
                           html.Div([
                               dcc.Dropdown(
                                   id = 'dropdownDimension1',
                                   value='Principal component 0'
                               )
                           ],style={'width':'30%'}),

                           html.Div([
                               dcc.Dropdown(
                                   id = 'dropdownDimension2',
                                   value='Principal component 1'

                               )
                           ],style={'width':'30%',
                                    'margin-left':'15px'}),
                           html.Div([
                               dcc.Dropdown(
                                   id = 'dropdownDimension3',
                                   value='Principal component 2'
                               )
                           ],style={'width':'30%',
                                    'margin-left':'15px'})
                       ], className='row',
                       style={
                           'zIndex':1
                       }),
                       html.Div([
                        dcc.Loading(id='loading-graph',
                            type='circle',
                            children=[
                                    html.Div(
                                        id='manifold-graph',
                                        children=[
                                            dcc.Graph(figure=fig)
                                        ],
                                        style={
                                            'margin-top': '15px',
                                            'zIndex': 900
                                        }
                                    )
                                    ])
                        ]),
                   ])



app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

#-------------------------------FUNCTION------------------------
def parse_contents(contents, filename):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            try:
                # sep = None detects separator
                dff = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')),sep = None,engine='python')
            except:
                dff = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')), sep=',',decimal='.')
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            dff = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
    columns = [{'name': i, 'id': i} for i in dff.columns]

    # Get only numeric variables
    dff = dff.select_dtypes(['number'])
    #print(dff)
    return dff.to_json(date_format='iso',orient = 'split')

def apply_manifold(data, algorithm = 'PCA', ncomponents = 3):
    max_iter = 100
    n_neighbors = 10

    if(data is not None):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        df_scaled_data = pd.DataFrame(data=scaled_data,columns=data.columns)

        if(algorithm == 'PCA'):
            manifold = PCA(n_components=ncomponents)
            principalComponents = manifold.fit_transform(df_scaled_data)
            principalDf = pd.DataFrame(data = principalComponents
                                       , columns = ['Principal component {}'.format(i) for i in range(ncomponents)])
            return principalDf

        elif(algorithm == 'MDS'):
            manifold = MDS(n_components=ncomponents, max_iter=max_iter, n_init=1)
            principalComponents = manifold.fit_transform(df_scaled_data)
            principalDf = pd.DataFrame(data = principalComponents
                                       , columns = ['Principal component {}'.format(i) for i in range(ncomponents)])
            return principalDf

        elif(algorithm == 'IsoMAP'):
            manifold = Isomap(n_components=ncomponents, n_neighbors=n_neighbors)
            principalComponents = manifold.fit_transform(df_scaled_data)
            principalDf = pd.DataFrame(data = principalComponents
                                       , columns = ['Principal component {}'.format(i) for i in range(ncomponents)])
            return principalDf

        elif(algorithm == 'LLE'):
            manifold = LocallyLinearEmbedding(n_components=ncomponents)
            principalComponents = manifold.fit_transform(df_scaled_data)
            principalDf = pd.DataFrame(data = principalComponents
                                       , columns = ['Principal component {}'.format(i) for i in range(ncomponents)])
            return principalDf

        elif(algorithm == 'KPCA'):
            manifold = KernelPCA(n_components=ncomponents)
            principalComponents = manifold.fit_transform(df_scaled_data)
            principalDf = pd.DataFrame(data = principalComponents
                                       , columns = ['Principal component {}'.format(i) for i in range(ncomponents)])
            return principalDf

        elif(algorithm == 't-SNE'):
            manifold = TSNE(n_components=ncomponents, init='pca', random_state=0)
            principalComponents = manifold.fit_transform(df_scaled_data)
            principalDf = pd.DataFrame(data = principalComponents
                                       , columns = ['Principal component {}'.format(i) for i in range(ncomponents)])
            return principalDf

        elif(algorithm == 'UMAP'):
            manifold = PCA(n_components=ncomponents)
            principalComponents = manifold.fit_transform(df_scaled_data)
            principalDf = pd.DataFrame(data = principalComponents
                                       , columns = ['Principal component {}'.format(i) for i in range(ncomponents)])
            return principalDf

        else:
            print('Invalid algorithm {}'.format(algorithm))
    else:
        print('Missing input data')
# -----------------------------------------


# Upload data and store in data-storage
@app.callback(Output('data-storage', 'data'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
@cache.memoize(timeout=60)  # in seconds
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        #print("Parse")
        dff = parse_contents(list_of_contents, list_of_names)
        #print("1. Datos cargados")
        #print(dff)
        return dff

# Update data table and dropdown
@app.callback([Output('dropDown','options'),
               Output('dropDown','value')],
              [Input('data-storage','data')])
def update_data_table(input_data):
    data = []
    columns = []
    options = [{'label': i, 'value': i} for i in df.columns]
    value = df.columns
    if input_data is not None:
        dff = pd.read_json(input_data,orient='split')
        data = dff.to_dict('records')
        columns = [{'name': i, 'id': i} for i in dff.columns]
        options = [{'label': i, 'value': i} for i in dff.columns]
        value = dff.columns
        #print("2. Mostrar datos en tabla")
        #print(dff.head())
    return options, value

# Select columns and store in filtered-data-storage
@app.callback(Output('filtered-data-storage','data'),
              [Input('data-storage','data'),
               Input('dropDown','value')])
def filter_data(input_data,columns):
    if input_data is not None and len(columns)>0:
        dff = pd.read_json(input_data,orient='split')
        #print("3. Filtramos la tabla")
        #print(dff.head())
        #print(columns)
        dff = dff[columns]
        print(dff.head())
        return dff.to_json(date_format='iso',orient = 'split')
    else:
        dff = df[columns]
        return dff.to_json(date_format='iso', orient='split')

# Update filtered data table
@app.callback(Output('data-table','children'),
              [Input('filtered-data-storage','data'),
               Input('dropDown','value')])
def update_filtered_data(input_data, cols):
    #print("4. Cargar datos filtrados a tabla")
    if input_data is not None:
        try:
            dff = pd.read_json(input_data,orient='split')
            #print(dff.head())
            return html.Div(
                            html.Div([
                                dbc.Table.from_dataframe(
                                    df=dff.head(50),
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    size='sm',
                                )
                            ])
                        )

        except:
            return html.Div([
                dbc.Table.from_dataframe(
                    df=df.head(50),
                    striped=True,
                    bordered=True,
                    hover=True,
                    size='sm',
                )
            ])
    else:
        return html.Div([])

@app.callback([Output('manifold-data-storage', 'data'),
               Output('dropdownDimension1','options'),
               Output('dropdownDimension2','options'),
               Output('dropdownDimension3','options')],
              [Input('Run_Button','n_clicks')],
              [State('filtered-data-storage','data'),
               State('PCA_NDimensions','value'),
               State('tabs','active_tab')])
def update_output_div(run_click, input_data, ncomponents, activeTab):
    try:
        print('RUN')
        if(input_data is not None and run_click is not None):
            print('Your have clicked {} times and entered {} manifold algorithm and {} dimensions'.format(run_click,activeTab,ncomponents))
            dff = pd.read_json(input_data,orient='split')
            principalDf = apply_manifold(dff,algorithm=activeTab,ncomponents=ncomponents)
            print(principalDf.head())
            options = [{'label': i, 'value': i} for i in principalDf.columns]
            return principalDf.to_json(date_format='iso',orient = 'split'), options, options, options
        else:
            principalDf = apply_manifold(df, algorithm=activeTab, ncomponents=ncomponents)
            print(principalDf.head())
            options = [{'label': i, 'value': i} for i in principalDf.columns]
            return principalDf.to_json(date_format='iso', orient='split'), options, options, options
    except:
        principalDf = apply_manifold(df, algorithm=activeTab, ncomponents=ncomponents)
        print(principalDf.head())
        options = [{'label': i, 'value': i} for i in principalDf.columns]
        return principalDf.to_json(date_format='iso', orient='split'), options, options, options

@app.callback([Output('dropdownDimension3','disabled'),
               Output('manifold-graph','children')],
              [Input('manifold-data-storage','data'),
               Input('dropdownDimension1','value'),
               Input('dropdownDimension2','value'),
               Input('dropdownDimension3','value'),
               Input('graphSwitch','value')])
def update_graph(input_data, dim1, dim2,dim3, graph3d):
    print('Input data:',input_data)
    if input_data is not None:
        try:
            dff = pd.read_json(input_data,orient='split')
            if(graph3d):
                if(dim1 is not None and dim2 is not None and dim3 is not None):
                    x = dff[dim1].values
                    y = dff[dim2].values
                    z = dff[dim3].values
                    fig = px.scatter_3d(dff, x=x, y=y, z=z, labels={'x':dim1,'y':dim2,'z':dim3} ,height=500,opacity=0.7)
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=20, b=20),
                    )

                    return False, html.Div([
                        dcc.Graph(
                            figure=fig
                        )
                    ],style={
                        'margin-top':'15px',
                        'zIndex':1
                    })
                else:
                    return False, html.Div(
                        [
                            dcc.Graph(
                            )
                        ], style={
                            'margin-top': '15px',
                            'zIndex': 900
                        })
            else:
                if(dim1 is not None and dim2 is not None):
                    print("Update figure")

                    return True, html.Div(
                        [
                            dcc.Graph(
                                figure = {
                                    'data': [dict(
                                        x=dff[dim1].values,
                                        y=dff[dim2].values,
                                        mode='markers',
                                        marker={
                                            'size': 15,
                                            'opacity': 0.5,
                                            'line': {'width': 0.5, 'color': 'white'}
                                        }
                                    )],
                                    'layout': dict(
                                        xaxis={
                                            'title': dim1,
                                            'type': 'linear'
                                        },
                                        yaxis={
                                            'title': dim2,
                                            'type': 'linear'
                                        },
                                        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                                        hovermode='closest'
                                    )
                                }
                            )],style={
                            'margin-top':'15px',
                            'zIndex':900
                                }
                    )
                else:
                    return True, html.Div(
                        [
                            dcc.Graph(
                            )
                        ], style={
                            'margin-top': '15px',
                            'zIndex': 900
                        })
        except:

            return False, html.Div(
                [
                    dcc.Graph(
                        figure = {
                            'data': [dict(
                                x=dff[dim1].values,
                                y=dff[dim2].values,
                                mode='markers',
                                marker={
                                    'size': 15,
                                    'opacity': 0.5,
                                    'line': {'width': 0.5, 'color': 'white'}
                                }
                            )],
                            'layout': dict(
                                xaxis={
                                    'title': 'sepal_length',
                                    'type': 'linear'
                                },
                                yaxis={
                                    'title': 'sepal_width',
                                    'type': 'linear'
                                },
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                                hovermode='closest'
                            )
                        }
                    )
                ],style={
                    'margin-top':'15px',
                    'zIndex':900
                })
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
@cache.memoize(timeout=60)  # in seconds
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=False)
