#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:55:45 2020

@author: salim
"""

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
                'width': '100%',
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
        html.Div(id='output-data-upload'),
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
        return html.Div([
            'There was an error processing this file.'
        ])
    columns = [{'name': i, 'id': i} for i in df.columns]
    return html.Div([
        html.H5(filename),
        ##html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX':'scroll',
                         'maxHeight':'350px',
                         'overflowY':'scroll'

            },
            style_cell={
                'minWidth':'100px',
                'text-align':'center',
                'width':'30%'
            }
        ),

        html.Hr(),  # horizontal line

        dcc.Dropdown(
            id='dd_variables',
            style={
                'width': '99%',
                'margin': '5px'
            },
            options=[{'label':i,'value':i} for i in df.keys()],
            value=df.columns[0],
            multi=True
        ),

        html.Div(id='output-selected-data')
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)