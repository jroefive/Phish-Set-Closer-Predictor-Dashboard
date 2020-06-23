"""Plotly Dash HTML layout override."""

html_layout = '''
<!DOCTYPE html>
    <html>
        <head>
            <style>  
                header {  
                    background-image: url(https://github.com/jroefive/jroefive.github.io/blob/master/photos/donut_stripe_fin.png?raw=true);  
                    height: 63px
                        }  
            </style>  
            {%metas%}
            <title>{%title%}</title>
            {%css%}
        </head>
        <header>
        </header>
        <body>
            {%app_entry%}
            <header >
                {%config%}
                {%scripts%}
                {%renderer%}
            </header>
        </body>
    </html>
'''