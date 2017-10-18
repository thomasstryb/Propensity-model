import numpy as np
import pandas as pd

from patsy import dmatrices
from sklearn.linear_model import LogisticRegression

import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def build_model():

    dta = pd.read_csv('data.csv')
    y, X = dmatrices('churn ~ C(product)+ avg_revenue+ revenue_variance+ years+ C(industry)+ cs_queries+ C(geo)',
                 dta, return_type='dataframe')

    X = X.rename(columns = {
        'C(product)[T.Business]':'Business Product',
        'C(product)[T.Economy]':'Economy Product',
        'C(product)[T.Premium]':'Premium Product',
        'C(product)[T.Standard]':'Standard Product',
        'C(industry)[T.Finance]':'Finance Industry',
        'C(industry)[T.Logistics]':'Logistics Industry',
        'C(industry)[T.Manufacturing]':'Manufacturing Industry',
        'C(industry)[T.Public]':'Public Sector',
        'C(industry)[T.Retail]':'Retail Industry',
        'C(industry)[T.Telecome]':'Telecom Industry',
        'C(geo)[T.Australia]':'Australia',
        'C(geo)[T.EastAsia]':'East Asia',        
        'C(geo)[T.Europe]':'Europe',
        'C(geo)[T.MiddleEast]':'Middle East',        
        'C(geo)[T.NorthAfrica]':'North Africa',
        'C(geo)[T.NorthAmerica]':'North America',        
        'C(geo)[T.SouthAmerica]':'South America',
        'C(geo)[T.UK]':'United Kingdom',        
        })

    model = LogisticRegression()
    model.fit(X, y)

    coef_names = X.columns.tolist()
    coef_names = coef_names[1:]
    print(coef_names)

    coef_impact = list(np.transpose(model.coef_))
    coef_impact = [c[0] for c in coef_impact[1:]]

    coef_impact = list(zip(coef_names, coef_impact))
    normalised_coef_impact = normalise_sort(coef_impact)
    graph_coefficients(normalised_coef_impact, './templates/churn-plot.html')
    graph_map('./templates/churn-map.html')

    return model


def normalise_sort(coef_in):
    coef_in.sort(reverse = True, key=lambda x: x[1])
    max_coef = coef_in[0][1]
    norm_coef_in = [(c[0], (c[1] /max_coef) * 100) for c in coef_in]
    return norm_coef_in

    

def graph_coefficients(coef_impact_in, file_path_in):

    
    data = [go.Bar(
        x = [c[0] for c in coef_impact_in],
        y = [c[1] for c in coef_impact_in]
    )]
    plotly.offline.plot(data, filename=file_path_in, auto_open=False)


def predict_churn(model_in, attributes_in):
    attribute_list = [0 for x in range(0, 22)]
    attribute_list[attributes_in[0]] = 1         # product
    attribute_list[4 + attributes_in[1]] = 1     # sector
    attribute_list[10 + attributes_in[2]] = 1    # geo
    
    attribute_list[18] = attributes_in[3]        # avg revenue
    attribute_list[19] = attributes_in[4]        # revenue variance
    attribute_list[19] = attributes_in[5]        # years
    attribute_list[19] = attributes_in[6]        # cs_queries

    attribute_list.insert(0, 1)                  # insert axis value at the beginning of the list.
    
    instance_attribute = np.array(attribute_list).reshape(1,-1)
    
    predict_score = model_in.predict_proba(instance_attribute)
    print(predict_score)
    print(model_in.classes_)
    predict_score = int(predict_score[0][1] * 100)
    
    return predict_score


        
    
def graph_map(file_path_in):

    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

    data = [ dict(
            type = 'choropleth',
            locations = df['CODE'],
            z = df['GDP (BILLIONS)'],
            text = df['COUNTRY'],
            colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
                [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
            autocolorscale = False,
            reversescale = True,
            marker = dict(
                line = dict (
                    color = 'rgb(180,180,180)',
                    width = 0.5
                ) ),
            colorbar = dict(
                autotick = False,
                tickprefix = '$',
                title = 'Churn Score'),
          ) ]

    layout = dict(
        title = '2017 Sept, Customer Churn by Country',
        geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(
                type = 'Mercator'
            )
        )
    )

    fig = dict( data=data, layout=layout )
    #py.iplot( fig, validate=False, filename='' )
    plotly.offline.plot(fig, filename=file_path_in, auto_open=False, validate=False)
    
    
        

