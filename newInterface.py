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
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import dash_table
import dash_daq as daq
from flask_caching import Cache
import os
import json
import pandas as pd
import numpy as np
import math

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
cache = Cache(app.server, config={
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR':'cache-directory'
})

df_c = px.data.iris() # iris is a pandas DataFrame
df = df_c.select_dtypes(['number'])
options_c = [{'label': i, 'value': i} for i in df_c.columns]
value_c = df_c.columns[0]
options = [{'label': i, 'value': i} for i in df.columns]
value = df.columns
max = len(df.columns)
first_iter = True

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "40%",
    "padding": "2rem 1rem",
    "background-color": "#F8F8F8",
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
modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("Data loaded"),
                dbc.ModalBody("Data set has been loaded succesfully."),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ml-auto")
                ),
            ],
            id="modal",
        ),
    ]
)

collapse = html.Div(
    [
        modal,
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

collapse2 = html.Div(
    [
        dbc.Button(
            "Visualization Options",
            id="collapse-button2",
            className="mb-6",
            color="primary",
            block=True
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody(
                html.Div([
                    html.Div([
                        html.Div([
                            html.P('Graphic display:')
                        ],style={'width':'50%'}),
                        html.Div([
                            html.P('Colorize by:')
                        ],style={'width':'50%'})
                    ],className='row'),
                    html.Div([
                        html.Div([
                            html.P('2D'),
                            daq.ToggleSwitch(
                                id='graphSwitch',
                                color='#007bff',
                                value=True,
                            ),
                            html.P('3D')
                        ], className='row',
                        style={
                            'width':'50%',
                            'margin-left':'2%'
                        }),

                        html.Div([
                            dcc.Dropdown(
                                id='color_label',
                                value='species',
                                placeholder='Select variable to colorize',
                                style={'verticalAlign':'middle'}
                            )
                        ],style={
                            'width':'45%'
                        })
                    ],className='row'),

                    html.Div([
                        html.Div([
                            html.P('X')
                        ],style={'width':'30%',
                                 'text-align':'center',
                                 'margin-left':'3%'}),

                        html.Div([
                            html.P('Y')
                        ],style={'width':'30%',
                                 'text-align':'center',
                                 'margin-left':'3%'}),
                        html.Div([
                            html.P('Z')
                        ],style={'width':'30%',
                                 'text-align':'center',
                                 'margin-left':'3%'})
                    ], className='row',
                        style={
                            'margin-top':'15px'
                        }),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id = 'dropdownDimension1',
                                value='Principal component 0'
                            )
                        ],style={'width':'32%'}),

                        html.Div([
                            dcc.Dropdown(
                                id = 'dropdownDimension2',
                                value='Principal component 1'

                            )
                        ],style={'width':'33%',
                                 'margin-left':'1%'}),
                        html.Div([
                            dcc.Dropdown(
                                id = 'dropdownDimension3',
                                value='Principal component 2'
                            )
                        ],style={'width':'33%',
                                 'margin-left':'1%'})
                    ], className='row',
                        style={
                            'zIndex':1
                        }),

                    html.Div([
                       html.P('Select Hoverdata to display:')
                    ],className='row',
                    style={
                        'margin-top':'15px'
                    }),
                    html.Div([
                        dcc.Dropdown(
                            id = 'dropdownHoverData',
                            placeholder='Select hoverdata',
                            multi=True,
                            style={
                                'width':'100%'
                            }
                        )
                    ], className='row'),
                    html.Div([
                        html.P('Minimal distance to origin (% of max. distance):')
                    ],className='row',
                    style={
                        'margin-top':'15px'
                    }),

                    html.Div([
                        dcc.Slider(
                            id='min-distance-slider',
                            min=0,
                            max=100,
                            step=1,
                            value=0,
                            marks={i:'{}%'.format(i) for i in range(0,110,10)},
                        )
                    ],
                    style={
                        'width':'100%'
                    })
                ])
            )),
            id="collapse2",
        ),
    ]
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
                    max = max,
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

                html.P(children='Dimensions:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='MDS_NDimensions',
                    value = 3,
                    min = 1,
                    max = max,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

            html.Div([

                html.P(children='Number of initializations:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='MDS_initializations',
                    value = 4,
                    min = 1,
                    max = 10,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

            html.Div([

                html.P(children='Max. iterations:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='MDS_iterations',
                    value = 300,
                    min = 1,
                    max = 1000,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

        ]
    )
])

tab_IsoMAP = dbc.Card([
    dbc.CardBody(
        [
            html.Div([

                html.P(children='Number of dimensions:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='isomap_NDimensions',
                    value = 3,
                    min = 1,
                    max = max,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

            html.Div([

                html.P(children='Number of Neighbors:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='isomap_nneighbors',
                    value = 10,
                    min = 1,
                    max = 100,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

        ]
    )
])

tab_LLE = dbc.Card([
    dbc.CardBody(
        [
            html.Div([

                html.P(children='Number of dimensions:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='LLE_NDimensions',
                    value = 3,
                    min = 1,
                    max = max,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

        ]
    )
])

tab_KPCA = dbc.Card([
    dbc.CardBody(
        [
            html.Div([

                html.P(children='Number of dimensions:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='KPCA_NDimensions',
                    value = 3,
                    min = 1,
                    max = max,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

        ]
    )
])

tab_tSNE = dbc.Card([
    dbc.CardBody(
        [
            html.Div([

                html.P(children='Number of dimensions:',
                       style={'width': '200px'}),

                daq.NumericInput(
                    id='tSNE_NDimensions',
                    value = 3,
                    min = 1,
                    max = max,
                    style={'margin-left':'15px'}
                )

            ], className='row'),

        ]
    )
])


sidebar = html.Div(
    [
        html.Div([
            
           html.H2(children='Manifold Algorithms',
                   style={'text-align':'center'}),
            
            html.Img(src='https://otriuv.es/wp-content/uploads/2014/03/logo-idal.png',
            style={'margin-left': '25%',
                    'width': '50%'}),

            html.P('Salim Chikh & Diego Calvete',
                   style={
                       'text-align':'center'
                   })
            
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
                dcc.Store(id='complete-data-storage',
                          data=df_c.to_json(date_format='iso',orient = 'split'))
            ]),
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
                       collapse2,
                       html.Div([
                           html.Div([
                               html.P('Show principal components on hover data'),
                               daq.ToggleSwitch(
                                   id='hover-components',
                                   color='#007bff',
                                   value=True,
                                   style={
                                       'margin-left':'15px'
                                   }
                               )
                           ],className='row',
                           style={
                               'margin-top':'15px',
                               'margin-left':'1%'
                           }),
                        dcc.Loading(id='loading-graph',
                            type='circle',
                            children=[
                                    html.Div(
                                        id='manifold-graph',
                                        children=[
                                            dcc.Graph(
                                                id='manifold-gr'
                                            )
                                        ],
                                        style={
                                            'margin-top': '15px',
                                            'zIndex': 900
                                        }
                                    )
                                    ])
                        ]),
                       html.Div(
                           id='histogram-div',
                           children=[
                               html.Div(
                                   [
                                       html.Div([
                                           html.P('Histogram'),
                                           daq.ToggleSwitch(
                                               id='hist-distplot',
                                               color='#007bff',
                                               value=False,
                                           ),
                                           html.P('Distplot'),
                                           dcc.Dropdown(
                                               id='dropdown-variable',
                                               placeholder='Select variable for histogram',
                                               options=options_c,
                                               style={
                                                   'margin-left':'15px',
                                                   'width':'60%'
                                               }
                                           ),
                                       ],style={
                                           'width':'100%',
                                           'margin-left':'5%'
                                       }, className='row'),

                                   ],id='select-hist-variable',
                                   className='row',
                                   style={
                                       'margin-top':'15px',
                                       'display':'none',
                                       'zIndex':1
                                   }),
                               dcc.Loading(id='loading-hist',
                                           type='circle',
                                           children=[
                                               html.Div(id='histogram',
                                                        style={
                                                            'zIndex':900,
                                                        })
                                           ]),
                           ]
                       )


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
    print(dff.shape)
    dff = dff.dropna(axis=0)
    dff = dff.reset_index(drop=True)
    return dff

def apply_manifold(data, algorithm = 'PCA', ncomponents = 3, max_iter=100, n_neighbors=10, n_init=1):

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
            manifold = MDS(n_components=ncomponents, max_iter=max_iter, n_init=n_init)
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


def get_outliers(dataFrame,minDistance):
    dist = np.zeros(dataFrame.shape[0])

    for col in dataFrame.columns:
        dist = dist + np.power(dataFrame[col],2)
        #print(dist)

    distance = np.sqrt(dist)

    print("Min distance:",np.min(distance))
    print("Max distance:",np.max(distance))

    minDist = np.max(distance)*minDistance/100

    print("Min distance:",minDist)

    return distance>=minDist

def get_max_distance(dataFrame):
    dist = np.zeros(dataFrame.shape[0])

    for col in dataFrame.columns:
        dist = dist + np.power(dataFrame[col],2)
        #print(dist)

    distance = np.sqrt(dist)

    return np.max(distance)

def create_hover(df, dims, hoverdata, hovercomponents, label):
    nDimsDf = df.shape[1]
    print(nDimsDf)
    hover = dict(Id=df.index)

    for i in range(min(nDimsDf,len(dims))):
        if(dims[i] is not None):
            hover[dims[i]]=hovercomponents

    for component in hoverdata:
        hover[component]=True

    if(label is not None):
        hover['label']=False

    return hover
# -----------------------------------------


# Upload data and store in data-storage
@app.callback([Output('data-storage', 'data'),
               Output('complete-data-storage','data')],
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
        # Get only numeric variables
        dff_numeric = dff.select_dtypes(['number'])
        #print(dff.shape)
        #print(dff_numeric.shape)
        return dff_numeric.to_json(date_format='iso',orient = 'split'), dff.to_json(date_format='iso',orient = 'split')
    else:
        return df.to_json(date_format='iso',orient = 'split'), df_c.to_json(date_format='iso',orient = 'split')

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

@app.callback([Output('PCA_NDimensions','max'),
               Output('MDS_NDimensions','max'),
               Output('isomap_NDimensions','max'),
               Output('KPCA_NDimensions','max'),
               Output('LLE_NDimensions','max'),
               Output('tSNE_NDimensions','max'),
               Output('PCA_NDimensions','value'),
               Output('MDS_NDimensions','value'),
               Output('isomap_NDimensions','value'),
               Output('KPCA_NDimensions','value'),
               Output('LLE_NDimensions','value'),
               Output('tSNE_NDimensions','value')],
              [Input('filtered-data-storage','data')],
              [State('PCA_NDimensions','value'),
               State('MDS_NDimensions','value'),
               State('isomap_NDimensions','value'),
               State('KPCA_NDimensions','value'),
               State('LLE_NDimensions','value'),
               State('tSNE_NDimensions','value')])
def update_max_dimensions(input_data, PCA_Ndim, MDS_Ndim, isomap_Ndim, KPCA_Ndim, LLE_Ndim, tSNE_Ndim):

    nDims = []
    nDims.append(PCA_Ndim)
    nDims.append(MDS_Ndim)
    nDims.append(isomap_Ndim)
    nDims.append(KPCA_Ndim)
    nDims.append(LLE_Ndim)
    nDims.append(tSNE_Ndim)

    if input_data is not None:
        try:
            nDimPCA = PCA_Ndim
            nDimMDS = MDS_Ndim
            nDimIsomap = isomap_Ndim
            nDimKPCA = KPCA_Ndim
            nDimLLE = LLE_Ndim
            nDimtSNE = tSNE_Ndim

            dff = pd.read_json(input_data,orient='split')
            max_iterations = len(dff.columns)

            for i,dim in enumerate(nDims):
                if(dim>max_iterations):
                    nDims[i]=max_iterations

            return max_iterations, max_iterations, max_iterations, max_iterations, max_iterations, max_iterations, nDims[0], nDims[1], nDims[2], nDims[3], nDims[4], nDims[5]
        except:
            nDim = PCA_Ndim
            max_iterations = len(df.columns)

            for i,dim in enumerate(nDims):
                if(dim>max_iterations):
                    nDims[i]=max_iterations
            return max_iterations, max_iterations, max_iterations, max_iterations, max_iterations, max_iterations, nDim, nDim, nDim, nDim, nDim, nDim

    else:
        max_iterations = len(df.columns)
        return max_iterations, max_iterations, max_iterations, max_iterations, max_iterations, max_iterations


@app.callback([Output('manifold-data-storage', 'data'),
               Output('dropdownDimension1','options'),
               Output('dropdownDimension2','options'),
               Output('dropdownDimension3','options'),
               Output('color_label','options'),
               Output('dropdownHoverData','options'),
               Output('dropdownHoverData','value'),
               Output('dropdown-variable','options')],
              [Input('Run_Button','n_clicks')],
              [State('filtered-data-storage','data'),
               State('complete-data-storage','data'),
               State('PCA_NDimensions','value'),
               State('MDS_NDimensions','value'),
               State('isomap_NDimensions','value'),
               State('LLE_NDimensions','value'),
               State('KPCA_NDimensions','value'),
               State('tSNE_NDimensions','value'),
               State('tabs','active_tab'),
               State('MDS_iterations','value'),
               State('MDS_initializations','value'),
               State('isomap_nneighbors','value')])
def update_output_div(run_click, input_data,complete_input_data, PCAncomponents, MDSncomponents, isomapncomponents, LLEncomponents, KPCAncomponents, tSNEncomponents, activeTab, max_iterations, n_init, n_neighbors):
    try:
        print('RUN')
        if(input_data is not None and run_click is not None):
            if(activeTab=='PCA'):
                ncomponents = PCAncomponents
            elif(activeTab=='MDS'):
                ncomponents = MDSncomponents
            elif(activeTab=='IsoMAP'):
                ncomponents = isomapncomponents
            elif(activeTab=='LLE'):
                ncomponents = LLEncomponents
            elif(activeTab=='KPCA'):
                ncomponents = KPCAncomponents
            elif(activeTab=='t-SNE'):
                ncomponents = tSNEncomponents

            print('Your have clicked {} times and entered {} manifold algorithm and {} dimensions'.format(run_click,activeTab,ncomponents))
            dff = pd.read_json(input_data,orient='split')
            dff_c = pd.read_json(complete_input_data,orient='split')
            principalDf = apply_manifold(dff,algorithm=activeTab,ncomponents=ncomponents, max_iter=max_iterations, n_neighbors=n_neighbors, n_init=n_init)
            print(principalDf.head())
            options = [{'label': i, 'value': i} for i in principalDf.columns]
            options_c = [{'label': i, 'value': i} for i in dff_c.columns]
            return principalDf.to_json(date_format='iso',orient = 'split'), options, options, options, options_c, options_c, dff_c.columns[:5], options_c
        else:
            if(activeTab=='PCA'):
                ncomponents = PCAncomponents
            elif(activeTab=='MDS'):
                ncomponents = MDSncomponents
            elif(activeTab=='IsoMAP'):
                ncomponents = isomapncomponents
            elif(activeTab=='LLE'):
                ncomponents = LLEncomponents
            elif(activeTab=='KPCA'):
                ncomponents = KPCAncomponents
            elif(activeTab=='t-SNE'):
                ncomponents = tSNEncomponents

            principalDf = apply_manifold(df, algorithm=activeTab, ncomponents=ncomponents, max_iter=300, n_neighbors=10, n_init=1)
            print(principalDf.head())
            options = [{'label': i, 'value': i} for i in principalDf.columns]
            options_c = [{'label': i, 'value': i} for i in df_c.columns]
            return principalDf.to_json(date_format='iso', orient='split'), options, options, options, options_c, options_c, df_c.columns[:5], options_c
    except:
        principalDf = apply_manifold(df, algorithm=activeTab, ncomponents=ncomponents, max_iter=300, n_neighbors=10, n_init=1)
        print(principalDf.head())
        options = [{'label': i, 'value': i} for i in principalDf.columns]
        options_c = [{'label': i, 'value': i} for i in df_c.columns]
        return principalDf.to_json(date_format='iso', orient='split'), options, options, options, options_c,  options_c, df_c.columns[:5], options_c

@app.callback([Output('min-distance-slider','max'),
               Output('min-distance-slider','marks'),
               Output('min-distance-slider','step')],
              [Input('manifold-data-storage','data')])
def update_slider(input_data):
    dff = pd.read_json(input_data,orient='split')
    max_dist = get_max_distance(dff)

    marks = {i:'{}%'.format(i) for i in range(0,110,10)}
    #print(marks)

    return 100, marks, 1

@app.callback([Output('dropdownDimension3','disabled'),
               Output('manifold-graph','children')],
              [Input('manifold-data-storage','data'),
               Input('dropdownDimension1','value'),
               Input('dropdownDimension2','value'),
               Input('dropdownDimension3','value'),
               Input('graphSwitch','value'),
               Input('color_label','value'),
               Input('dropdownHoverData','value'),
               Input('min-distance-slider','value'),
               Input('hover-components','value')],
              [State('complete-data-storage','data')])
def update_graph(input_data, dim1, dim2,dim3, graph3d, color_label, hoverdata, min_dist, hover_components, complete_input_data):
    if input_data is not None:

        # Read manifold data and complete data set data
        dff = pd.read_json(input_data,orient='split')
        dff_c = pd.read_json(complete_input_data,orient='split')

        dims = [dim1, dim2, dim3]

        nDims = dff.shape[1]


        #print("Dff cols:",dff.shape[1])
        #print(hover_columns)

        # Display only data with distance > minDistance
        outliers = get_outliers(dff,min_dist)

        dff = dff[outliers]
        dff_c = dff_c[outliers]

        # Add the ID to identify the points and display histogram
        hover_columns = hoverdata
        hover_columns.append(dff_c.index)

        # Merge manifold and complete data
        dff_m= pd.merge(dff, dff_c, left_index=True, right_index=True)

        # Set color column
        if(color_label in dff_c.columns):
            dff_m['label']=dff_c[color_label]
            label = 'label'
        else:
            label = None

        #print(dff_m.columns)
        #print("Label",label)

        hover = create_hover(dff,dims,hoverdata[:-1],hover_components,label)
        #print(hover)

        if(graph3d):
            if(dim1 is not None and dim2 is not None and dim3 is not None):
                x = dff[dim1].values
                y = dff[dim2].values
                z = dff[dim3].values

                fig = px.scatter_3d(dff_m, x=x, y=y, z=z, labels={'x':dim1,'y':dim2,'z':dim3} ,opacity=0.7,color=label, template='plotly', hover_data=hover)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    clickmode='event+select'
                )

                return False, html.Div([
                    dcc.Graph(
                        id='manifold-gr',
                        figure=fig,
                        style={
                            'height':'70vh'
                        }
                    )
                ],style={
                    'margin-top':'15px',
                    'zIndex':1,
                })
            else:
                return False, html.Div(
                    [
                        dcc.Graph(
                        )
                    ], style={
                        'margin-top': '15px',
                        'zIndex': 900,
                        'height':'70vh'
                    })
        else:
            if(dim1 is not None and dim2 is not None):

                print("Update figure")
                x = dff[dim1].values
                # If only one dimension set y to 0
                try:
                    y = dff[dim2].values
                except:
                    y = [0 for i in x]

                print(y)

                fig = px.scatter(dff_m, x=x, y=y,labels={'x':dim1,'y':dim2}, color=label, opacity=0.7, hover_data=hover)
                #fig.update_xaxes(showline=True, linewidth = 1, showgrid=True, gridwidth=1, gridcolor='LightBlue', linecolor='black')
                #fig.update_yaxes(showline=True, linewidth = 1, showgrid=True, gridwidth=1, gridcolor='LightBlue', linecolor='black')

                fig.update_xaxes(showline=True, linewidth = 1, linecolor='black')
                fig.update_yaxes(showline=True, linewidth = 1, linecolor='black')
                fig.update_traces(marker=dict(size=25,
                                              line=dict(
                                                  color='White',
                                                  width=1
                                              )),
                                  )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    dragmode='select'
                )
                return True, html.Div(
                    [
                        dcc.Graph(
                            id='manifold-gr',
                            figure = fig
                        )],style={
                        'margin-top':'15px',
                        'zIndex':900,
                        'height':'70vh'
                            }
                )
            else:
                return True, html.Div(
                    [
                        dcc.Graph(
                            id='manifold-gr',
                        )
                    ], style={
                        'margin-top': '15px',
                        'zIndex': 900,
                        'height':'70vh'
                    })

@app.callback(Output('graphSwitch','value'),
              [Input('manifold-data-storage','data')])
def update_switch(input_data):
    if input_data is not None:
        try:
            dff = pd.read_json(input_data,orient='split')

            if(len(dff.columns)>2):
                g3D = True
            else:
                g3D = False

            return g3D
        except:
            return True
    else:
        return True

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

@app.callback(
    Output("collapse2", "is_open"),
    [Input("collapse-button2", "n_clicks")],
    [State("collapse2", "is_open")],
)
@cache.memoize(timeout=60)  # in seconds
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("modal", "is_open"),
    [Input("data-storage", "data"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(input_data, n2, is_open):
    #print('modal')
    global first_iter
    if(not first_iter):
        if input_data is not None or n2:
            return not is_open
        return is_open
    first_iter = False
    return False
# options for the dropdown

@app.callback(Output('histogram','children'),
              [Input('dropdown-variable','value'),
               Input('hist-distplot','value')],
              [State('complete-data-storage','data'),
               State('manifold-gr','selectedData')])
def update_histogram(column, distplot, input_data, selectedData):
    if input_data is not None:
        dff_c = pd.read_json(input_data,orient='split')
    if selectedData is not None:
        print(selectedData)
        points = selectedData['points']
        pointsIndex = [i['customdata'][0] for i in points]


        #print('Length:',len(pointsIndex),pointsIndex)
        #print(column)
        try:
            hist_df = dff_c.iloc[pointsIndex,:]
            distplot_df = hist_df[column]
            #print('Categorical data:',hist_df[column].dtype == 'O')
            if(column is not None):
                if(hist_df[column].dtype == 'O'):
                    fig = px.histogram(hist_df,x=column,nbins=20)
                else:
                    if(distplot):
                        columns = [column]
                        fig = ff.create_distplot([hist_df[i] for i in columns],columns,bin_size=.25)
                    else:
                        fig = px.histogram(hist_df,x=column,nbins=20)

                return html.Div([
                    dcc.Graph(
                        figure=fig
                    )
                ],style={
                    'zIndex':900
                })
            else:
                return html.Div([])
        except:
            print("error histogram")
            return html.Div([])

@app.callback([Output('dropdown-variable','value'),
               Output('select-hist-variable','style')],
              [Input('manifold-gr','selectedData'),
               Input('dropdown-variable','options')])
def enable_disable_dropdown(selected_data, options):
    if(selected_data is not None and options is not None):
        return options[0]['label'], {
            'margin-top':'15px',
            'display':'block'
        }
    else:
        return None, {
            'margin-top':'15px',
            'display':'none'
        }


if __name__ == "__main__":
    app.run_server(debug=True)
