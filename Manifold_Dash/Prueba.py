import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions=True

app.layout = html.Div([
    html.Div([
        #Header
        html.Div([
            html.H2(
                children='Manifold Learning Analytics',
                style={
                    'text-align':'center'
                }
            )
        ])
    ], className='row'),

    html.Div([
        # Upload file
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
    html.Div([
        html.Hr()
    ]),
    html.Div([
        html.Div([
            dash_table.DataTable(
                id='data-table',
                style_table={'overflowX': 'scroll',
                             'maxHeight': '350px',
                             'overflowY': 'scroll',
                             'width':'100%'

                             },
                style_cell={
                    'minWidth': '100px',
                    'text-align': 'center',
                    'width': '30%'
                }
            ),

            html.Hr(id='hr2'),  # horizontal line
        ])
    ],className='row'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dropDown',
                style={
                    'width': '100%',
                    'margin': '5px'
                },
                value=[],
                multi=True
            )
        ],id='dd'),

        html.Div(id='filtered-data-storage',
                 style={
                     'display':'none'
                 })
    ],className='row'),

    html.Div(id='data-table2')
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
                    io.StringIO(decoded.decode('utf-8')),sep = ';')
            except:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')), sep=',')
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
    columns = [{'name': i, 'id': i} for i in df.columns]
    return df.to_json(date_format='iso',orient = 'split')
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
        print("1. Datos cargados")
        print(df)
        return df

# Update data table and dropdown
@app.callback([Output('data-table','data'),
               Output('data-table','columns'),
               Output('dropDown','options'),
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
        #value = dff.columns[0]
        print("2. Mostrar datos en tabla")
        print(dff.head())
    return data, \
           columns, \
           options, \
           value

# Select columns and store in filtered-data-storage
@app.callback(Output('filtered-data-storage','children'),
              [Input('data-storage','children'),
               Input('dropDown','value')])
def filter_data(input_data,columns):
    if input_data is not None and len(columns)>0:
        dff = pd.read_json(input_data[0],orient='split')
        print("3. Filtramos la tabla")
        print(dff.head())
        print(columns)
        dff = dff[columns]
        print(dff.head())
        return dff.to_json(date_format='iso',orient = 'split')
    else:
        return []

# Update filtered data table
@app.callback(Output('data-table2','children'),
              [Input('filtered-data-storage','children'),
               Input('dropDown','value')])
def update_filtered_data(input_data, cols):
    print("4. Cargar datos filtrados a tabla")
    if input_data is not None:
        try:
            dff = pd.read_json(input_data,orient='split')
            print(dff.head())
            return html.Div([

                dash_table.DataTable(
                    id='data-table3',
                    data = dff.to_dict('records'),
                    columns = [{'name': i, 'id': i} for i in dff.columns],
                    style_table={'overflowX': 'scroll',
                                 'maxHeight': '350px',
                                 'overflowY': 'scroll',
                                 'width':'100%'

                                 },
                    style_cell={
                        'minWidth': '100px',
                        'text-align': 'center',
                        'width': '30%'
                    }
                ),
                html.Hr(id='hr2')  # horizontal line
            ])
        except:
            return html.Div([])
    else:
        return html.Div([])



#
if __name__ == '__main__':
    app.run_server(debug=True)