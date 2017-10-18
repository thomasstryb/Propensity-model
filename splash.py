from flask import Flask, render_template, url_for
from flask.ext.wtf import Form
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import Required
import churn_model

global_model = []

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string!'

@app.route('/')
def index():
    global global_model
    global_model = churn_model.build_model()
    return render_template('index.html')


@app.route('/step_dashboard', methods=['GET', 'POST'])

def step_dashboard():
    product = '0'
    avg_revenue = ''
    avg_var = ''
    years = ''
    industry = '0'
    cs_queries = ''
    geo = '0'
    churn_score = ''
    
    
    my_form = ChurnForm()

    
    if my_form.validate_on_submit():

        product = str(my_form.form_product.data)
        avg_revenue = int(my_form.form_avg_revenue.data)
        avg_var = int(my_form.form_avg_var.data)
        years = int(my_form.form_years.data)
        industry = str(my_form.form_industry.data)
        cs_queries = int(my_form.form_cs_queries.data)
        geo = str(my_form.form_geo.data)

        churn_score = churn_model.predict_churn(global_model, [int(product), int(industry), int(geo), avg_revenue, avg_var, years,  cs_queries, ])

        churn_score = 'Propensity score is: ' + str(churn_score) + '%.'

    
    return render_template('step_dashboard.html', form=my_form, form_product=product, form_avg_revenue=avg_revenue, form_avg_var = avg_var, form_years = years, form_industry = industry, form_cs_queries = cs_queries, form_geo = geo, form_score=churn_score)


class ChurnForm(Form):
    form_product = SelectField('Product:', choices=[ ('0', 'Business Products'), ('1', 'Economy Product'), ('2', 'Premium Product'), ('3', 'Standard Product')])
    form_avg_revenue = StringField('Revenue:')
    form_avg_var = StringField('Variance:')
    form_years = StringField('Lifetime:')
    form_industry = SelectField('Industry:', choices= [ ('0', 'Finance'),('1', 'Logistics'), ('2', 'Manufacturing'), ('3', 'Public Sector'), ('4', 'Retail'), ('5', 'Telecom')])
    form_cs_queries = StringField('CS Queries:')
    form_geo = SelectField("Geography", choices = [ ('0','Australia'), ('1','East Asia'), ('2','Europe'), ('3','Middle East'), ('4', 'North Africa'), ('5', 'North America'), ('6','South America'), ('7', 'United Kingdom')])
    submit = SubmitField('Get churn score')
    
    
    

if __name__ == '__main__':
    app.run(debug=True)




