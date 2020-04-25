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


app.layout = html.Div([
    #Header
    html.Div([
       html.H2(
           children='Manifold Learning Analytics',
           style={
            'text-align':'center'
        }
       )
    ]),

    # Upload file
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '98%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),

        # Hidden Div inside the app that stores the intermediate data
        html.Div(id='data-storage',
                 style={
                     'display':'none'
                 }),
        html.Div(id='display-data-table')

    ]),
    html.Div([

    ]),

])


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
    print(df.head())
    return df.to_json(date_format='iso',orient = 'split')


@app.callback(Output('data-storage', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return df

@app.callback(Output('display-data-table','children'),
              [Input('data-storage','children')])
def update_data_table(input_data):
    if input_data is not None:
        print(input_data)
        dff = pd.read_json(input_data[0],orient='split')
        return html.Div([
            dash_table.DataTable(
                data=dff.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in dff.columns],
                style_table={'overflowX': 'scroll',
                             'maxHeight': '350px',
                             'overflowY': 'scroll'

                             },
                style_cell={
                    'minWidth': '100px',
                    'text-align': 'center',
                    'width': '30%'
                }
            ),

            html.Hr(),  # horizontal line

            dcc.Dropdown(
                id='dd_variables',
                style={
                    'width': '99%',
                    'margin': '5px'
                },
                options=[{'label': i, 'value': i} for i in dff.keys()],
                value=dff.columns[0:2],
                multi=True
            )
        ])



if __name__ == '__main__':
    app.run_server(debug=True)