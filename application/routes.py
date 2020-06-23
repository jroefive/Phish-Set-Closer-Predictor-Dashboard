"""Core Flask app routes."""
from flask import render_template
from flask import current_app as app



@app.route('/')
def home():
    return render_template('index.jinja2',
                           title='Phish Show Set Closer Prediction Dashboard',
                           template='home-template',
                           body="This is an interactive dashboard that allows you to input any Phish show and see a the percent chance that each song in the show would close the set.")