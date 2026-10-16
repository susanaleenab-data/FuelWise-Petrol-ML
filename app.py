import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='FuelWise | Weekly Petrol Planner', page_icon='⛽', layout='wide')
st.title('⛽ FuelWise | Weekly Petrol Planner')
st.caption('A practical petrol budget calculator in Indian rupees (INR) and transparent machine-learning demonstration')

@st.cache_data
def load_data():
    return pd.read_csv('vehicle_petrol_mileage_2000.csv')

@st.cache_resource
def train_model():
    raw = load_data()
    # Uploaded data has distance=1 for every record, so construct explicitly labeled
    # synthetic training scenarios with varied distances. Not field observations.
    rng = np.random.default_rng(42)
    n = 6000
    km = rng.uniform(5, 1000, n)
    mileage = rng.uniform(5, 40, n)
    fuel = km / mileage
    scenarios = pd.DataFrame({'distance_traveled_km': km, 'vehicle_mileage_km_per_liter': mileage, 'petrol_used_liters': fuel})
    X = scenarios[['distance_traveled_km', 'vehicle_mileage_km_per_liter']]
    y = scenarios['petrol_used_liters']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, scenarios, X_test, y_test, pred

with st.sidebar:
    st.header('Your weekly journey')
    km = st.number_input('Total distance travelled per week (km)', min_value=0.0, value=250.0, step=10.0)
    mileage = st.number_input('Vehicle mileage (km per litre)', min_value=0.1, value=20.0, step=0.5)
    price = st.number_input('Petrol price this week (₹ per litre)', min_value=0.0, value=105.0, step=1.0)
    symbol = '₹'  # All budget amounts are displayed in Indian rupees (INR)
    budget = st.number_input('Optional weekly petrol budget', min_value=0.0, value=1500.0, step=100.0)

exact_litres = km / mileage
cost = exact_litres * price
st.subheader('Your budget estimate')
a,b,c,d = st.columns(4)
a.metric('Weekly petrol', f'{exact_litres:.2f} L')
b.metric('Weekly cost', f'{symbol}{cost:,.2f}')
c.metric('Monthly average', f'{symbol}{cost*52/12:,.2f}')
d.metric('Annual estimate', f'{symbol}{cost*52:,.2f}')
if budget and cost > budget:
    st.warning(f'Estimated weekly cost exceeds your budget by {symbol}{cost-budget:,.2f}.')
elif budget:
    st.success(f'Estimated weekly cost is within budget by {symbol}{budget-cost:,.2f}.')
st.info('Budget calculations use the exact fuel formula: litres = distance ÷ mileage. Actual consumption can vary with traffic, weather, driving style and vehicle condition.')

model, scenarios, X_test, y_test, pred = train_model()
t1,t2,t3 = st.tabs(['📊 Budget charts', '🤖 Machine learning lab', '🔎 Dataset audit'])
with t1:
    weeks = pd.DataFrame({'Week': [f'Week {i}' for i in range(1, 5)], 'Estimated cost (INR)': [cost]*4})
    st.plotly_chart(px.bar(weeks, x='Week', y='Estimated cost (INR)', title='Four-week budget projection (₹)'), use_container_width=True)
    sensitivity = pd.DataFrame({'Mileage (km/L)': np.linspace(max(1,mileage*.6), mileage*1.4, 35)})
    sensitivity['Weekly cost (INR)'] = km / sensitivity['Mileage (km/L)'] * price
    st.plotly_chart(px.line(sensitivity, x='Mileage (km/L)', y='Weekly cost (INR)', title='How fuel efficiency affects cost (₹)'), use_container_width=True)
with t2:
    st.subheader('Random Forest regression (educational experiment)')
    st.warning('Training scenarios are SYNTHETIC and derived from the known fuel formula. This is NOT a validated real-world fuel-consumption model. The direct formula is more reliable for the main budget estimate.')
    p = float(model.predict(pd.DataFrame({'distance_traveled_km':[km], 'vehicle_mileage_km_per_liter':[mileage]}))[0])
    st.metric('ML predicted fuel (L)', f'{p:.2f}')
    st.caption(f'Formula baseline: {exact_litres:.2f} L. Difference: {p-exact_litres:+.2f} L.')
    m1,m2,m3=st.columns(3)
    m1.metric('Holdout MAE (L)', f'{mean_absolute_error(y_test,pred):.3f}')
    m2.metric('Holdout RMSE (L)', f'{np.sqrt(mean_squared_error(y_test,pred)):.3f}')
    m3.metric('Holdout R²', f'{r2_score(y_test,pred):.4f}')
    chart = pd.DataFrame({'Actual litres': y_test.to_numpy(), 'Predicted litres': pred})
    st.plotly_chart(px.scatter(chart, x='Actual litres', y='Predicted litres', opacity=.35, title='Unseen synthetic test set: predicted vs actual'), use_container_width=True)
    st.caption('80/20 random split, fixed random seed. These scores measure interpolation on formula-generated scenarios, not performance on real driving trips.')
with t3:
    data = load_data()
    st.write('Uploaded dataset:', len(data), 'rows,', len(data.columns), 'columns')
    st.dataframe(data.head(30), use_container_width=True)
    st.write('Missing values:', data.isna().sum().to_dict())
    st.write('Distinct distances:', data.distance_traveled_km.nunique())
    err = (data.petrol_used_liters - data.distance_traveled_km / data.vehicle_mileage_km_per_liter).abs()
    st.write(f'Maximum deviation from distance/mileage formula: {err.max():.8f} L')
    st.caption('The uploaded data contains only 1-km trips and formula-derived fuel values. It cannot establish real-world accuracy across varied trips.')
    st.download_button('Download uploaded data', data.to_csv(index=False), file_name='vehicle_petrol_mileage_2000.csv', mime='text/csv')
