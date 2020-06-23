"""Create a Dash app within a Flask app."""
import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from .layout import html_layout
import plotly.graph_objects as go
from app.functions import call_model, generate_table
from dash.dependencies import Input, Output
import datetime
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Possible Updates: Change font to make it easier to read, play with scattor plots instead to give date and link to listen on hover
# Add drop downs for the songs in the set to focus in on just one graph.
# Reset hover text to say "36% of the way into set 1"

def create_dashboard(server):
    #Need a background version of dataframe to set options for the date dropdown
    tracks_bg = pd.read_csv('https://jroefive.github.io/track_length_combined')

    #Set up dataframes for laster
    closer_predict_display = pd.DataFrame()
    score_table = pd.DataFrame()

    #Add Year, Month, and Day columns to the dataframe to make the dropdowns dynamic (You only get options for the months played in the chosen year)
    tracks_bg['year'] = pd.DatetimeIndex(tracks_bg['date']).year
    tracks_bg['month'] = pd.DatetimeIndex(tracks_bg['date']).month
    tracks_bg['day'] = pd.DatetimeIndex(tracks_bg['date']).day

    #Create lists for dropdowns for all shows and all years.
    all_dates = tracks_bg['date'].unique()
    all_years = tracks_bg['year'].unique()

    #Initiate the dashboard
    dash_app = dash.Dash(server=server,
                         routes_pathname_prefix='/dashapp/',
                         external_stylesheets=[dbc.themes.BOOTSTRAP, 'static/style.css']
                         )

    #Pull in defaul html saved in layout.py
    dash_app.index_string = html_layout
    #Create overall layout
    dash_app.layout = html.Div([
            # Input window for dates and graph options.
            html.Div([dbc.Row([
                #Date inputs
                dbc.Col(html.Div([
                    html.Div([html.P('Choose show from top list of all possible shows.  Or:')], style={'height':'35px', 'font-size':'14'}),
                    html.Div([html.P('Choose show by choosing year, month, day separately.')], style={'height':'35px'}),
                    dcc.Dropdown(id='Show_Date',
                        options=[{'label': i, 'value': i} for i in all_dates],
                        value='2020-02-23'),
                    html.Div([
                    dcc.Dropdown(id='Show_Year',
                        options=[{'label': i, 'value': i} for i in all_years],
                        placeholder="Select Year")],
                    style={'width':'33.33%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(id='Show_Month',placeholder="Select Month (after year)")],
                        style={'width':'33.33%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(id='Show_Day',placeholder="Select Day (after month)")],
                        style={'width':'33.33%', 'display': 'inline-block'})],
                style={'width': '90%', 'margin-top':'15px', 'display': 'inline-block', 'font-family': 'Arial'})),
                #Model options inputs
                dbc.Col(html.Div([
                        html.Div([dcc.Link('Link to Blog Post with Explanations', href='https://jroefive.github.io/2020/06/22/Predicting-Phish-Set-Closers.html', style={'color': '#F15A50'})], style={'height':'35px', 'margin-top':'20px'}),
                        html.Div([html.P(' ')], style={'height':'25px','display':'inline-block'}),
                        html.Div([html.P(id='cutoff-msg')], style={'display':'inline-block'}),
                    dcc.Dropdown(id='Cutoff',
                                 options=[{'label': 'Include All Songs', 'value': 0},
                                          {'label': 'Include Songs Started 30 Minutes or More Into Set', 'value': 30},
                                          {'label': 'Include Songs Started 45 Minutes or More Into Set', 'value': 45},
                                          {'label': 'Include Songs Started 60 Minutes or More Into Set', 'value': 60}],
                                 value=45),
                    dcc.Dropdown(id='Number_of_Features',
                                options=[{'label': 'Include 3 Features', 'value': 3},
                                 {'label': 'Include 4 Features', 'value': 4},
                                 {'label': 'Include 5 Features', 'value': 5},
                                 {'label': 'Include 6 Features', 'value': 6}],
                                value=4)],
                style={'width': '90%', 'display': 'inline-block', 'font-family': 'Arial'}))
                ,],
                    no_gutters=True,)],
            style={'width': '90%', 'margin-left':'75px','color': '#F15A50', 'text-align': 'center','backgroundColor':'#2C6E91',}),
            #Table of songs with inputs and prediction value
            html.Div([
                html.Div(id='table-container-songs1', style={'width': '95%', 'font-family': 'Arial'}),
                html.Div(id='table-container-songs2', style={'width': '95%', 'margin-top':'25px', 'font-family': 'Arial'}),
                html.Div(id='table-container-songs3', style={'width': '95%', 'margin-top':'25px', 'font-family': 'Arial'}),
                html.Div(id='table-container-songse', style={'width': '95%', 'margin-top':'25px', 'font-family': 'Arial'})],
            style={'margin-left':'60px', 'margin-top':'25px', 'color': '#F15A50', 'text-align': 'center','backgroundColor':'#2C6E91'}),
            #Background graphs and info
            html.Div(children=[
                html.Div([dcc.Graph(id='graph')],
                         style={'width': '90%', 'display': 'inline-block', 'font-family': 'Arial'}),
                ],
            style={'width': '95%', 'display': 'inline-block', 'color': '#F15A50', 'text-align': 'center','backgroundColor':'#2C6E91'}),
                ],
        style={'text-align': 'center','backgroundColor':'#2C6E91'})


    #Update the month options after a year is chosen
    @dash_app.callback(
        Output('Show_Month', 'options'),
        [Input('Show_Year', 'value')])
    def set_month_options(show_year):
        month_options_pd = tracks_bg[tracks_bg['year']==show_year].copy()
        month_options = month_options_pd['month'].unique()
        return [{'label': i, 'value': i} for i in month_options]

    #Update the day options once a month is chosen.
    @dash_app.callback(
        Output('Show_Day', 'options'),
        [Input('Show_Year', 'value'),
         Input('how_Month', 'value'),])
    def set_day_options(show_year, show_month):
        day_options_pd = tracks_bg[(tracks_bg['year']==show_year) & (tracks_bg['month']==show_month)].copy()
        day_options = day_options_pd['day'].unique()
        return [{'label': i, 'value': i} for i in day_options]

    #Reset show_date once a day is chosen
    @dash_app.callback(
        Output('Show_Date', 'value'),
        [Input('Show_Day', 'value'),
        Input('Show_Year', 'value'),
        Input('Show_Month', 'value')])
    def set_date(day, year, month):
        show_date = datetime.date(year, month, day)
        return show_date

    #Update tables and graphs every time a change is made
    @dash_app.callback(
        [Output('table-container-songs1', 'children'),
         Output('table-container-songs2', 'children'),
         Output('table-container-songs3', 'children'),
         Output('table-container-songse', 'children'),
        Output('cutoff-msg', 'children'),
        Output('graph', 'figure')],
        [Input('Cutoff', 'value'),
        Input('Show_Date', 'value'),
        Input('Number_of_Features', 'value')])
    def update_table(cutoff, show_date, num_features):
        #Call model fucntion
        closer_predict_display, score_table, cutoff_change_msg, feature_names, feature_imps, feature_desc_df = call_model(show_date, cutoff, num_features)

        #Create a figure with three subplots to show feature importance graph, description of features, and accuracy scores
        figure = make_subplots(rows=1, cols=3, column_widths=[0.25, 0.5, 0.25], specs=[[{"type": "bar"},{"type": "table"}, {"type": "table"}]])
        figure.add_trace(go.Bar(x=feature_names, y=feature_imps),row=1,col =1)
        figure.update_yaxes(title_text="Feature Importance", row=1, col=1)
        figure.update_layout(paper_bgcolor="#2C6E91", margin=dict(l=40, r=40, t=40, b=40), height = 400),
        figure.update_layout(font=dict(family="Arial, monospace",size=14,color='#F15A50'))
        figure.update_traces(marker_color='#F15A50', marker_line_color="#2C6E91",
                          marker_line_width=1.5, opacity=0.6)

        figure.add_trace(
            go.Table(
                columnwidth=[1, 3],
                header=dict(
                    values=["Feature Label", "Feature Description"],
                    font=dict(size=14,family="Arial, monospace"),
                    line_color='#2C6E91',
                    align="left"
                ),
                cells=dict(
                    values=[feature_desc_df[k].tolist() for k in feature_desc_df.columns],
                    align="left", font=dict(size=14,family="Arial, monospace"),
                    line_color='#2C6E91',
                    height=40)
            ),
            row=1, col=2)

        figure.add_trace(
            go.Table(
                columnwidth=[2, 1],
                header=dict(
                    values=["Score Type", "Accuracy"],
                    font=dict(size=14,family="Arial, monospace"),
                    line_color='#2C6E91',
                    align="center"
                ),
                cells=dict(
                    values=[score_table[k].tolist() for k in score_table.columns],
                    align="center",
                    font=dict(size=14, family="Arial, monospace"),
                    line_color='#2C6E91',
                    height = 30)
            ),
            row=1, col=3)

        #Split prediction and input table into sets for
        merged_df = closer_predict_display
        merged_df1 = merged_df[merged_df['set']=='1'].copy()
        merged_df2 = merged_df[merged_df['set'] == '2'].copy()
        merged_df3 = merged_df[merged_df['set'] == '3'].copy()
        merged_dfe = merged_df[(merged_df['set'] == 'E') | (merged_df['set'] == 'E2')].copy()

        #Pass all info back into app
        return generate_table(merged_df1), generate_table(merged_df2), generate_table(merged_df3), generate_table(merged_dfe), cutoff_change_msg, figure

    return dash_app.server





