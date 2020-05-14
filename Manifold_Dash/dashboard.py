import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_daq as daq
import plotly.graph_objs as go
import plotly.express as px

import pandas as pd

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS

print("Version",dash.__version__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions=True


app.layout = html.Div([

    # Title
    html.Div([
        #Header
        html.Div([
            html.H2(
                children='Manifold Learning Analytics',
                style={
                    'text-align':'center'
                }
            ),
            html.H5(
                children='1. Upload a dataset in *.csv format',
                style={
                    'text-align':'left'
                }
            )
        ])
    ], className='row'),

    # Upload file
    html.Div([
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
                    #'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),

            # Hidden Div inside the app that stores the intermediate data
            html.Div(id='data-storage',
                     style={
                         'display':'none'
                     })

        ])
    ],className='row'),

    # Horizontal line
    html.Div([
        html.Hr()
    ]),

    # Text
    html.Div([
       html.H5(
           children='2. Select variables from the dataset to apply dimensionality reduction (manifold)',
           style={
               'text-align':'left'
           }
       )
    ]),

    # Select Variables
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dropDown',
                style={
                    'width': '100%',
                    'margin': '0px'
                },
                value=[],
                multi=True
            )
        ],id='dd'),

        # Horizontal line
        html.Hr(),

        # Filtered data storage
        html.Div(id='filtered-data-storage',
                 style={
                     'display':'none'
                 })
    ],className='row'),

    # Datatable
    html.Div(id='data-table2'),

    # 3. Select manifold and output dimensionality
    html.Div([
        html.H5(
            children='3. Select manifold and its output dimensionality',
            style={
                'text-align':'left'
            }
        )
    ],className='row'),

    # Manifold algorithm and output dimensions
    html.Div([
        html.Div([
            html.Div([
                html.Hgroup(
                    children='Manifold algorithm',
                    style={
                        'text-align':'center'
                    }
                )
            ])
        ],className='four columns'),

        html.Div([
            html.Div([
                html.Hgroup(
                    children='Output dimension',
                    style={
                        'text-align':'center'
                    }
                )
            ])
        ],className='six columns')

    ], className='row'),

    # Manifold algorithm
    html.Div([
        html.Div([
            dcc.Dropdown(
                id = 'dropdownManifold',
                options=[
                    {'label': 'PCA', 'value': 'PCA'},
                    {'label': 'Multidimensional Scaling', 'value': 'MDS'},
                    {'label': 'Isometric Mapping', 'value': 'IsoMAP'},
                    {'label': 'Linear Locally Embedding', 'value': 'LLE'},
                    {'label': 'Kernel PCA', 'value': 'KPCA'},
                    {'label': 't-distributed Stochastic Neighbor Embedding', 'value': 't-SNE'}
                ],
            )
        ],className='four columns'),
        html.Div([
            dcc.Slider(
                id='numberOfFeatures',
                min=1,
                max=1,
                value=2,
                marks={1:'1'}
            )
        ],className='six columns'),
        html.Div([
            daq.NumericInput(
                id='numericInput',
                value = 1,
                min = 1,
                max = 1
            )
        ],className='two columns')
    ], className='row'),

    html.Div([
        html.Hr(),
        html.Button(
            children='Run',
            id='runButton',
            n_clicks=0
        ),

        html.Hr()
    ], className='row'),

    html.Div([

    ]),
    html.Div([
        html.H5(
            children='4. Select dimensions to plot',
            style={
                'text-align':'left'
            }
        )
    ],className='row'),

    html.Div([
        html.Div([
            html.Hgroup(
                children='Dimension 1',
                style={
                    'text-align':'center'
                }
            )
        ],className='six columns'),

        html.Div([
            html.Hgroup(
                children='Dimension 2',
                style={
                    'text-align':'center'
                }
            )
        ],className='six columns')

    ],className='row'),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id = 'dropdownDimension1'
            )
        ],className='six columns'),

        html.Div([
            dcc.Dropdown(
                id = 'dropdownDimension2'
            )
        ],className='six columns'),
    ], className='row'),

    html.Div(id='manifold-data-storage',
             style={
                 'display':'none'
             }
    ),
    html.Div(id='graphHeader'),

    html.Div(id='manifold-graph')


])

# ----------- FUNCIONES ----------------
def parse_contents(contents, filename, date):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            try:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')),sep = ';',decimal=',')
            except:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')), sep=',',decimal='.')
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
    columns = [{'name': i, 'id': i} for i in df.columns]
    return df.to_json(date_format='iso',orient = 'split')

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
@app.callback(Output('data-storage', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        #print("1. Datos cargados")
        #print(df)
        return df

# Update data table and dropdown
@app.callback([Output('dropDown','options'),
               Output('dropDown','value')],
              [Input('data-storage','children')])
def update_data_table(input_data):
    data = []
    columns = []
    options = []
    value = []
    if input_data is not None:
        dff = pd.read_json(input_data[0],orient='split')
        data = dff.to_dict('records')
        columns = [{'name': i, 'id': i} for i in dff.columns]
        options = [{'label': i, 'value': i} for i in dff.columns]
        value = dff.columns
        #print("2. Mostrar datos en tabla")
        #print(dff.head())
    return options, value

# Select columns and store in filtered-data-storage
@app.callback(Output('filtered-data-storage','children'),
              [Input('data-storage','children'),
               Input('dropDown','value')])
def filter_data(input_data,columns):
    if input_data is not None and len(columns)>0:
        dff = pd.read_json(input_data[0],orient='split')
        #print("3. Filtramos la tabla")
        #print(dff.head())
        #print(columns)
        dff = dff[columns]
        #print(dff.head())
        return dff.to_json(date_format='iso',orient = 'split')
    else:
        return []

# Update filtered data table
@app.callback(Output('data-table2','children'),
              [Input('filtered-data-storage','children'),
               Input('dropDown','value')])
def update_filtered_data(input_data, cols):
    #print("4. Cargar datos filtrados a tabla")
    if input_data is not None:
        try:
            dff = pd.read_json(input_data,orient='split')
            #print(dff.head())
            return html.Div([

                dash_table.DataTable(
                    id='data-table3',
                    data = dff.to_dict('records'),
                    columns = [{'name': i, 'id': i} for i in dff.columns],
                    style_table={'overflowX': 'scroll',
                                 'maxHeight': '350px',
                                 'overflowY': 'scroll',
                                 'width':'100%',
                                 },
                    style_cell={
                        'minWidth': '100px',
                        'text-align': 'center',
                        'width': '30%'
                    },
                    fixed_rows={
                        'headers':True
                    }
                ),
                html.Hr(id='hr2')  # horizontal line
            ])
        except:
            return html.Div([])
    else:
        return html.Div([])

@app.callback([Output('numberOfFeatures','max'),
               Output('numberOfFeatures','marks'),
               Output('numericInput','max')],
              [Input('filtered-data-storage','children')])
def update_output_features2(input_data):
    if input_data is not None:
        try:

            dff = pd.read_json(input_data,orient='split')
            max = len(dff.columns)
            steps = int(round(max/10,-1))
            #steps = 1
            marks = {i:'{}'.format(i) for i in range(0,max,steps)}
            print("update features 2")
            print(marks)

            return  max, marks, max
        except:
            return 1,{1:'1'},1

@app.callback(Output('numericInput','value'),
              [Input('numberOfFeatures','value')])
def update_values(input_value):
    if(input_value is not None):
        return input_value
    else:
        return 1

@app.callback(Output('numberOfFeatures','value'),
              [Input('numericInput','value')])
def update_values(input_value):
    if(input_value is not None):
        return input_value
    else:
        return 1

@app.callback([Output('dropdownDimension1','options'),
               Output('dropdownDimension2','options')],
              [Input('manifold-data-storage','children')])
def update_dimensionSelection(input_data):
    ddDim1 = []
    ddDim2 = []
    if input_data is not None:
        try:
            dff = pd.read_json(input_data,orient='split')
            ddDim1 = [{'label':i,'value':i} for i in dff.columns]
            ddDim2 = ddDim1
            return ddDim1, ddDim2
        except:
            return [{'label':'Dimension 0', 'value':0}],[{'label':'Dimension 0', 'value':0}]
    else:
        return [{'label':'Dimension 0', 'value':0}],[{'label':'Dimension 0', 'value':0}]

@app.callback(Output('manifold-graph','children'),
              [Input('manifold-data-storage','children'),
               Input('dropdownDimension1','value'),
               Input('dropdownDimension2','value')])
def update_graph(input_data, dim1, dim2):
    if input_data is not None:
        try:
            dff = pd.read_json(input_data,orient='split')
            if(dim1 is not None and dim2 is not None):
                print("Update figure")

                return dcc.Graph(
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
                )
        except:

            return html.Div([])

@app.callback(Output('manifold-data-storage', 'children'),
              [Input('runButton', 'n_clicks'),
               Input('filtered-data-storage','children')],
              [State('dropdownManifold', 'value'),
               State('numericInput','value')])
def update_output_div(n_clicks, input_data, manifold, ncomponents):
    try:
        if(input_data is not None):
            print('Your have clicked {} times and entered {} manifold algorithm and {} dimensions'.format(n_clicks,manifold,ncomponents))
            dff = pd.read_json(input_data,orient='split')
            principalDf = apply_manifold(dff,manifold,ncomponents)
            return principalDf.to_json(date_format='iso',orient = 'split')
    except:
        a = 1

if __name__ == '__main__':
    app.run_server(debug=True)