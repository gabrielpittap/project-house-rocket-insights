import folium
import geopandas
import numpy as np
import pandas as pd![](../Project_ZaoDS_house_rocket_insights/misc/image/real-estate.jpg)
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import date, datetime
from PIL import Image

st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
# functions
def get_data(path):
    data = pd.read_csv(path)
    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def type_transform(data):
    data['id'] = data['id'].astype('str')

    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')  # .date.strftime('%Y-%m-%d')

    data = data.loc[data['bedrooms'] != 33].copy()

    # add new features
    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.093)

    return data


def define_season(date_value, hemisphere='north'):
    year = date_value.year

    if hemisphere == 'north':
        spring_start = date(year, 3, 21)
        spring_end = date(year, 6, 20)

        summer_start = date(year, 6, 21)
        summer_end = date(year, 9, 20)

        autumn_start = date(year, 9, 21)
        autumn_end = date(year, 12, 20)

        seasons = {
            'spring': pd.date_range(start=spring_start, end=spring_end),
            'summer': pd.date_range(start=summer_start, end=summer_end),
            'autumn': pd.date_range(start=autumn_start, end=autumn_end)
        }

        if date.strftime(date_value, format='%Y-%m-%d') in seasons['spring']:
            return 'spring'
        elif date.strftime(date_value, format='%Y-%m-%d') in seasons['summer']:
            return 'summer'
        elif date.strftime(date_value, format='%Y-%m-%d') in seasons['autumn']:
            return 'autumn'
        else:
            return 'winter'

    elif hemisphere == 'south':
        autumn_start = date(year, 3, 21)
        autumn_end = date(year, 6, 20)

        winter_start = date(year, 6, 21)
        winter_end = date(year, 9, 20)

        spring_start = date(year, 9, 21)
        spring_end = date(year, 12, 20)

        seasons = {
            'autumn': pd.date_range(start=spring_start, end=spring_end),
            'winter': pd.date_range(start=winter_start, end=winter_end),
            'spring': pd.date_range(start=autumn_start, end=autumn_end)
        }

        if date.strftime(date_value, format='%Y-%m-%d') in seasons['autumn']:
            return 'spring'
        elif date.strftime(date_value, format='%Y-%m-%d') in seasons['winter']:
            return 'winter'
        elif date.strftime(date_value, format='%Y-%m-%d') in seasons['spring']:
            return 'autumn'
        else:
            return 'summer'

def set_features(data):
    data['season'] = data['date'].apply(define_season)

    # add new features
    data['m2_lot'] = data['sqft_lot'] * 0.093
    data['m2_living'] = data['sqft_living'] * 0.093
    data['m2_above'] = data['sqft_above'] * 0.093
    data['m2_basement'] = data['sqft_basement'] * 0.093
    data['price_m2'] = data['price'] / (data['m2_lot'])

    return data


def create_header():
    # house rocket logo
    header_logo = Image.open('image/header.png')
    st.image(header_logo)

    st.title('Properties Portfolio - Business Dashboard')

    return None

def side_filters(data):
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    # default values for the filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    max_bedrooms = data['bedrooms'].max()
    max_bathrooms = data['bathrooms'].max()
    max_floors = data['floors'].max()

    # Sidebar Filters
    st.sidebar.title('Filters')
    st.sidebar.write('These filters apply to the metrics overview and maps, but not to the recommendations or hypothesis.')

    st.sidebar.subheader('Building attributes')
    f_bedrooms = st.sidebar.selectbox('Maximum bedrooms', data['bedrooms'].sort_values().unique(),
                                      index=(len(data['bedrooms'].sort_values().unique()) - 1), key='f_bedrooms')
    f_bathrooms = st.sidebar.selectbox('Maximum bathrooms', data['bathrooms'].sort_values().unique(),
                                       index=(len(data['bathrooms'].sort_values().unique()) - 1), key='f_bathrooms')
    f_floors = st.sidebar.selectbox('Maximum floors', data['floors'].sort_values().unique(),
                                    index=(len(data['floors'].sort_values().unique()) - 1), key='f_floors')
    f_waterfront = st.sidebar.checkbox('Only waterfront properties', value=False, key='f_waterfront')

    st.sidebar.subheader('Commercial attributes')
    f_price = st.sidebar.slider('Maximum price', min_price, max_price, max_price, key='f_price')

    # st.sidebar.subheader('Ano máximo de construção')
    f_yearbuilt = st.sidebar.slider('Maximum construction year', min_year_built, max_year_built, max_year_built,
                                    key='f_yearbuilt')

    # st.sidebar.subheader('Data máxima desde disponível')
    f_date = st.sidebar.slider('Maximum date since available', min_date, max_date, max_date, key='f_date')
    data['date'] = pd.to_datetime(data['date'])

    # filters the dataset

    data = data.loc[(data['yr_built'] <= f_yearbuilt) &
                    (data['price'] <= f_price) &
                    (data['date'] <= f_date) &
                    (data['bedrooms'] <= f_bedrooms) &
                    (data['bathrooms'] <= f_bathrooms) &
                    (data['floors'] <= f_floors)]
    if f_waterfront:
        data = data.loc[data['waterfront'] == f_waterfront]

    return data
# def side_filtersss(data):
#     # default values for the filters
#     min_year_built = int(data['yr_built'].min())
#     max_year_built = int(data['yr_built'].max())
#     min_date = pd.to_datetime(data['date'].min(), format='%Y-%m-%d')
#     max_date = pd.to_datetime(data['date'].max(), format='%Y-%m-%d')
#     min_price = int(data['price'].min())
#     max_price = int(data['price'].max())
#     max_bedrooms = data['bedrooms'].max()
#     max_bathrooms = data['bathrooms'].max()
#     max_floors = data['floors'].max()
#
#     # Sidebar Filters
#     st.sidebar.title('Filtros')
#
#     st.sidebar.subheader('Atributos construtivos')
#     f_bedrooms = st.sidebar.selectbox('Número máximo de quartos', data['bedrooms'].sort_values().unique(),
#                                       index=(len(data['bedrooms'].sort_values().unique()) - 1), key='f_bedrooms')
#     f_bathrooms = st.sidebar.selectbox('Número máximo de banheiros', data['bathrooms'].sort_values().unique(),
#                                        index=(len(data['bathrooms'].sort_values().unique()) - 1), key='f_bathrooms')
#     f_floors = st.sidebar.selectbox('Número máximo de andares', data['floors'].sort_values().unique(),
#                                     index=(len(data['floors'].sort_values().unique()) - 1), key='f_floors')
#     f_waterfront = st.sidebar.checkbox('Apenas imóveis beira-mar', value=False, key='f_waterfront')
#
#     st.sidebar.subheader('Atributos comerciais')
#     f_price = st.sidebar.slider('Preço máximo', min_price, max_price, max_price, key='f_price')
#
#     # st.sidebar.subheader('Ano máximo de construção')
#     f_yearbuilt = st.sidebar.slider('Ano máximo de construção', min_year_built, max_year_built, max_year_built,
#                                     key='f_yearbuilt')
#
#     # st.sidebar.subheader('Data máxima desde disponível')
#     f_date = st.sidebar.slider('Data máxima desde disponível', min_date, max_date, max_date, key='f_date')
#     data['date'] = pd.to_datetime(data['date'])
#
#     # reset button
#     # reset_filters = st.sidebar.button('Limpar filtros')
#     #
#     # if reset_filters:
#     #     st.session_state.f_bathrooms=max_bedrooms
#     #     st.session_state.f_price=max_price
#     #     st.session_state.f_bedrooms=max_bathrooms
#     #     st.session_state.f_yearbuilt=max_year_built
#     #     st.session_state.f_floors=max_floors
#     #     st.session_state.f_waterfront=False
#     #     st.session_state.f_date=max_date
#
#     # filters the dataset
#
#     data = data.loc[(data['yr_built'] <= f_yearbuilt) &
#                     (data['price'] <= f_price) &
#                     (data['date'] <= f_date) &
#                     (data['bedrooms'] <= f_bedrooms) &
#                     (data['bathrooms'] <= f_bathrooms) &
#                     (data['floors'] <= f_floors)]
#     if f_waterfront:
#         data = data.loc[data['waterfront'] == f_waterfront]
#
#     return data

def overview_data(data):
    with st.expander('Original dataset'):
        ov = st.container()

        ov.header('Dataset overview')
        ov.dataframe(data.head(50))

        c1, c2 = st.columns(2, gap='small')

        # Average metrics
        df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
        df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        df3 = data[['m2_lot', 'zipcode']].groupby('zipcode').mean().reset_index()
        df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

        # merge
        m1 = pd.merge(df1, df2, on='zipcode', how='inner')
        m2 = pd.merge(m1, df3, on='zipcode', how='inner')
        df = pd.merge(m2, df4, on='zipcode', how='inner')
        # c1.dataframe(df)
        # df.style.format({
        #    'price': '{:,.2f}',
        #    'm2_lot': '{:,.2f}',
        #    'price_m2': '{:,.2f}'
        # })
        df.columns = ['ZIPCODE', 'TOTAL PROPERTIES', 'MEAN PRICE', 'SIZE (m²)', 'MEAN PRICE/m²']
        df = df.style.format({
            'MEAN PRICE': 'US$ {:,.2f}',
            'SIZE (m²)': '{:,.2f}',
            'MEAN PRICE/m²': 'US$ {:,.2f}'
        })
        c1.subheader('Zipcode analysis')

        c1.dataframe(df, height=600)

        # Descriptive statistics

        # slices the dataset into a exclusive numeric one
        num_attributes = data.select_dtypes(include=['int64', 'float64'])

        # metrics
        media = pd.DataFrame(num_attributes.apply(np.mean))
        # mediana = pd.DataFrame(num_attributes.apply(np.median))
        std = pd.DataFrame(num_attributes.apply(np.std))

        min_value = pd.DataFrame(num_attributes.apply(np.min))
        max_value = pd.DataFrame(num_attributes.apply(np.max))

        dscstat_df = pd.concat([min_value, media, max_value, std], axis=1).reset_index()

        dscstat_df.columns = ['ATTRIBUTE', 'MIN', 'MEAN', 'MAX', 'STANDARD DEVIATION']
        c2.subheader('Descriptive Statistics')
        dscstat_df = dscstat_df.style.format({'MIN': '{:,.2f}',
                                              'MAX': '{:,.2f}',
                                              'MEAN': '{:,.2f}',
                                              'STANDARD DEVIATION': '{:,.2f}'}) \
            .hide_index()
        c2.dataframe(dscstat_df, height=600)

    st.markdown('''---''')

    return None

def portfolio_density(data, geofile):
    st.title('Metrics overview')

    c1, c2, c3, c4, c5, c6 = st.columns([2, 3, 2, 3, 2, 2])

    c1.metric('Total properties', data['id'].count())
    c2.metric('Mean price (US$)', ' {:20,.2f}'.format(data['price'].mean()))
    c3.metric('Average size (m²)', '{:.2f}'.format(data['m2_lot'].mean()))
    c4.metric('Mean Price/m²', f" US$ {data['price_m2'].mean():,.2f}")
    c5.metric('Average grade', f"{data['grade'].mean():,.2f}")
    c6.metric('Waterfront properties', data.loc[data['waterfront'] == True, 'waterfront'].count())

    st.markdown('''---''')

    # c1, gap, c2 = st.columns([8, 2, 8])
    c1, c2 = st.columns(2, gap='large')

    c1.subheader('Regional density')

    df = data

    # basemap - folium

    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                             zoom_start=10)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Availabçe for US$ {0} since {1}. \n\r Features: {2} m²,\n\r {3} bedrooms,\n\r {4} bathroom,\n\r built in {5}'.format(
                          row['price'], row['date'], row['m2_lot'], row['bedrooms'], row['bathrooms'], row['yr_built']
                      )).add_to(marker_cluster)

    # with c1:
    #     folium_static(density_map)

    # Region Price Map
    c2.subheader('Price density per region')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                  zoom_start=10)

    folium.Choropleth(data=df,
                      geo_data=geofile,
                      name='Price density',
                      columns=['ZIP', 'PRICE'],
                      key_on='feature.properties.ZIP',
                      fill_color='YlOrBr',
                      fill_opacity=0.7,
                      line_opacity=0.2,
                      highlight=True,
                      legend_name='MEAN PRICE IN US$').add_to(region_price_map)

    # folium.LayerControl().add_to(density_map)

    with c1:
        st_folium(density_map)

    with c2:
        st_folium(region_price_map)

    return None


def recommend_properties(data, min_condition=3):
    # groups properties by zipcode and adds the median price
    median_price_byzipcode = data[['price', 'zipcode']].copy().groupby('zipcode').median().reset_index()
    median_price_byzipcode.columns = ['zipcode', 'zip_median_price']

    # merges with original dataset
    prop_with_zip_median = data.copy().sort_values('zipcode').merge(median_price_byzipcode, on='zipcode', how='inner')

    # creates the recommendation column
    prop_recommendation = prop_with_zip_median.copy()
    prop_recommendation['recommendation'] = 'not recommended'

    prop_recommendation.loc[(prop_recommendation['condition'] >= 3) &
                            (prop_recommendation['price'] <= prop_recommendation['zip_median_price']),
                            'recommendation'] = 'recommended'

    return prop_recommendation

def create_recomm_table(data):
    recommendation_table = data[['id', 'zipcode', 'condition', 'price', 'recommendation']].copy()

    return recommendation_table

def set_price_and_season(recommended_properties):
    data = recommended_properties.loc[recommended_properties['recommendation'] == 'recommended'].copy()

    grouped_median_price = data[['zipcode', 'season', 'price']].copy().groupby(
        ['zipcode', 'season']).median().reset_index()
    grouped_median_price.rename(columns={'price': 'median_price'}, inplace=True)

    grouped_max_grade = data[['zipcode', 'season', 'grade']].copy().groupby(['zipcode', 'season']).max().reset_index()
    grouped_max_grade.rename(columns={'grade': 'max_grade'}, inplace=True)

    df_aux = pd.merge(data[['id', 'zipcode', 'season', 'price', 'grade']].copy(),
                      grouped_median_price,
                      on=['zipcode', 'season'],
                      how='inner')
    prop_price_time_recommendation = pd.merge(df_aux.copy(),
                                              grouped_max_grade,
                                              on=['zipcode', 'season'],
                                              how='inner')
    prop_price_time_recommendation['grade_ratio'] = np.round(prop_price_time_recommendation['grade'] /
                                                             prop_price_time_recommendation['max_grade'], 2)

    prop_price_time_recommendation['price_to_sell'] = (1 + (prop_price_time_recommendation['grade_ratio'] * 0.5)) * \
                                                      prop_price_time_recommendation['price']

    prop_price_time_recommendation.loc[prop_price_time_recommendation['price'] >
                                       prop_price_time_recommendation['median_price'], 'price_to_sell'] = \
        (1 + (prop_price_time_recommendation['grade_ratio'] * 0.2)) * prop_price_time_recommendation['price']

    prop_price_time_recommendation['profit'] = prop_price_time_recommendation['price_to_sell'] - \
                                               prop_price_time_recommendation['price']

    prop_price_time_recommendation.rename(columns={'season': 'season_to_sell'}, inplace=True)

    return prop_price_time_recommendation[['id', 'zipcode', 'price', 'season_to_sell', 'price_to_sell', 'profit']]


def show_recommendations():
    total_price_recommended = recommendation_table.loc[recommendation_table['recommendation'] == 'recommended'][
        'price'].sum()
    number_recommended_prop = recommendation_table.loc[recommendation_table['recommendation'] == 'recommended'][
        'id'].count()
    maximum_profit = price_and_season_to_sell['profit'].sum()

    st.title('Buy/Sell Recommendations')
    st.write(
        'Here are two tables that, respectively, show all the properties that are recommended to buy, according to specified conditions mentioned on the report, and the best season to sell them and a price suggestion. ')

    c1, c2 = st.columns(2, gap='small')

    with c1.expander('Best properties to buy'):
        st.dataframe(recommendation_table, height=600)

    with c2.expander('Selling price and time recommendation'):
        st.dataframe(price_and_season_to_sell, height=600)

    st.header('Results')

    st.write(
        f'The number of recommended properties is {number_recommended_prop} and the total amount invested for buying all of them is US$ {total_price_recommended:,.2f}')
    st.write(f'Following the recommendation, the maximum profit is US$ {maximum_profit:,.2f}')

    return None

def hypothesis():

    # Hypothesis testing

    # - H1: Waterfront properties are on average 30% more expensive.

    average_price_waterfront = data[['waterfront', 'price']].groupby('waterfront').mean().reset_index()

    waterfront_price_percentage = (average_price_waterfront.loc[average_price_waterfront['waterfront'] == 1][
                                       'price'].mean() /
                                   average_price_waterfront.loc[average_price_waterfront['waterfront'] == 0][
                                       'price'].mean()) * 100

    # - H2: Properties constructed before 1955 are 50% cheaper on average.

    meanprice_older_1955 = data.loc[data['yr_built'] < 1955]['price'].mean()
    meanprice_newer_1955 = data.loc[data['yr_built'] >= 1955]['price'].mean()

    # - H3: On average, properties with no basement have a total area 40% larger than properties with a basement.

    meanarea_basement = data.loc[data['sqft_basement'] > 0]['sqft_lot'].mean()
    meanarea_no_basement = data.loc[data['sqft_basement'] == 0]['sqft_lot'].mean()

    # - H4: The properties' average price year over year growth is 10%.

    prop_may_2014 = data.loc[
        (data['date'] >= data['date'].min()) & (data['date'] < pd.to_datetime(date(2014, 6, 1)))].copy()
    prop_may_2015 = data.loc[
        (data['date'] >= pd.to_datetime(date(2015, 5, 1))) & (data['date'] < pd.to_datetime(date(2015, 6, 1)))].copy()

    total_price_0514 = prop_may_2014['price'].mean()
    total_price_0515 = prop_may_2015['price'].mean()

    yoy_price = 100 * total_price_0515 / total_price_0514

    # - H5: The mean month-over-month price growth of 3-bathroom-properties is 15% throughout all the months.

    prop_3bath = data.loc[data['bathrooms'] == 3].copy()

    prop_3bath['year-month'] = data['date'].dt.to_period('M')

    prop_3bath_avgprice_month = prop_3bath[['year-month', 'price']].groupby(
        'year-month').mean().reset_index().sort_values('year-month')

    for row in range(len(prop_3bath_avgprice_month)):

        if row == 0:
            #         st.write('aloha')
            prop_3bath_avgprice_month.loc[row, 'mom_growth'] = 0
            continue

        prop_3bath_avgprice_month.loc[row, 'mom_growth'] = 100 * (
                prop_3bath_avgprice_month.loc[row, 'price'] / prop_3bath_avgprice_month.loc[row - 1, 'price'] - 1)

    mean_mom_price = prop_3bath_avgprice_month['mom_growth'].mean()

    # - H6: Properties with 2 or more floors have an average grade 50% higher than the other properties.

    meangrade_2plusfloors = data.loc[data['floors'] >= 2]['grade'].mean()
    meangrade_2minusfloors = data.loc[data['floors'] < 2]['grade'].mean()

    # - H7: The 'condition' attribute of a property is, on average, 30% higher on renovated properties.

    meancondition_renovated = data.loc[data['yr_renovated'] > 0]['condition'].mean()
    meancondition_notrenovated = data.loc[data['yr_renovated'] == 0]['condition'].mean()

    # - H8: Properties with 2 or more bedrooms that have less than 2 bathrooms are 40% cheaper on average.

    meanprice_2plusbed_less2bath = data.loc[(data['bedrooms'] >= 2) & (data['bathrooms'] < 2)]['price'].mean()
    meanprice_2plusbed_2plusbath = data.loc[(data['bedrooms'] >= 2) & (data['bathrooms'] >= 2)]['price'].mean()

    # - H9: Less than 15% of the properties have a basement and more than one floor above ground, and their average price is at least 20% higher than the average price of the whole portfolio.

    numprop_basement_floors = data.loc[(data['sqft_basement'] > 0) & (data['floors'] > 1)]['id'].count()
    total_prop = data['id'].count()

    dataset_numprop_basement_floors = data.loc[(data['sqft_basement'] > 0) & (data['floors'] > 1)].copy()
    dataset_meanprice = dataset_numprop_basement_floors['price'].mean()
    portfolio_mean_price = data['price'].mean()

    # - H10: Pareto: 80% of the portfolio's total cost lies in 20 to 25% of the properties.

    total_cost = data['price'].sum()

    data_sort_price = data.sort_values(by='price', ascending=False).copy()

    pareto_sum = 0
    pareto_count = 0

    for row in range(len(data_sort_price)):

        if pareto_sum <= 0.8 * total_cost:
            pareto_sum = pareto_sum + data_sort_price.iloc[row]['price']
            pareto_count = pareto_count + 1

        else:
            break

    st.title('Business Hypothesis')
    c1, c2 = st.columns(2, gap='small')


    with c1.expander('H1: Waterfront properties are on average 30% more expensive.'):
        st.write(f'False. Waterfront properties are {waterfront_price_percentage:,.2f}% more expensive.')
        #test

    with c1.expander('H2: Properties constructed before 1955 are 50% cheaper on average.'):
        st.write(
            f'False. Properties built before 1955 are {100 - (100 * meanprice_older_1955 / meanprice_newer_1955):,.2f}% cheaper.')

    with c1.expander('H3: Properties with no basement have a total area 40% larger than properties with a basement.'):
        st.write(
            f'False. Properties without a basement have a {(100 * (meanarea_no_basement / meanarea_basement) - 100):,.2f}% larger total area on average then properties with a basement.')

    with c1.expander('H4: The properties\' average price year over year growth is 10%.'):
        st.write(f'False. The Year-over-Year price growth between May/2014 and May/2015 is {yoy_price:.2f}%')
        #test

    with c1.expander('H5: The mean month-over-month price growth of 3-bathroom-properties is 15% throughout all the months.'):
        st.write(
            f'False. The mean month-over-month growth for the average price of 3-bathroom-properties is {mean_mom_price:,.2f}%')

    with c2.expander('H6: Properties with 2 or more floors have an average grade 50% higher than the other properties.'):
        st.write(
            f'False. The average grade for 2+ floors properties is {((100 * meangrade_2plusfloors / meangrade_2minusfloors) - 100):,.2f}% higher.')

    with c2.expander("H7: The 'condition' attribute of a property is, on average, 30% higher on renovated properties."):
        st.write(
            f'False. The average condition on renovated properties is {100 - (100 * meancondition_renovated / meancondition_notrenovated):,.2f}% lower.')

    with c2.expander('H8: Properties with 2 or more bedrooms that have less than 2 bathrooms are 40% cheaper on average.'):
        st.write(
            f'True. Properties with 2 or more bedrooms that have less than 2 bathrooms are {100 - (100 * meanprice_2plusbed_less2bath / meanprice_2plusbed_2plusbath):,.2f}% cheaper on average, wich is almost 40%.')

    with c2.expander('H9: Less than 15% of the properties have a basement and more than one floor above ground, and their average price is at least 20% higher than the average price of the whole portfolio.'):
        st.write(
            f'True. Only {(100 * numprop_basement_floors / total_prop):,.2f}% of the properties have a basement and more than one floor above ground Their average price is {(100 * (dataset_meanprice / portfolio_mean_price - 1)):,.2f}% higher than the average price of the portfolio.')

    with c2.expander('H10: Pareto: Approximately 80% of the portfolio\'s total cost lies in 20 to 25% of the properties.'):
        st.write(
            f'HFalse. {(100 * pareto_sum / total_cost):,.2f}% of the portfolio total cost lies in {(100 * pareto_count / total_prop):,.2f}% of the properties.')

    return None

#here is old function test hypothesis
# def test_hypothesis():
#     # Hypothesis testing
#
#     # - H1: Waterfront properties are on average 30% more expensive.
#
#     average_price_waterfront = data[['waterfront', 'price']].groupby('waterfront').mean().reset_index()
#
#     waterfront_price_percentage = (average_price_waterfront.loc[average_price_waterfront['waterfront'] == 1][
#                                        'price'].mean() /
#                                    average_price_waterfront.loc[average_price_waterfront['waterfront'] == 0][
#                                        'price'].mean()) * 100
#
#
#     # - H2: Properties constructed before 1955 are 50% cheaper on average.
#
#     meanprice_older_1955 = data.loc[data['yr_built'] < 1955]['price'].mean()
#     meanprice_newer_1955 = data.loc[data['yr_built'] >= 1955]['price'].mean()
#
#
#     # - H3: On average, properties with no basement have a total area 40% larger than properties with a basement.
#
#     meanarea_basement = data.loc[data['sqft_basement'] > 0]['sqft_lot'].mean()
#     meanarea_no_basement = data.loc[data['sqft_basement'] == 0]['sqft_lot'].mean()
#
#
#     # - H4: The properties' average price year over year growth is 10%.
#
#     prop_may_2014 = data.loc[
#         (data['date'] >= data['date'].min()) & (data['date'] < pd.to_datetime(date(2014, 6, 1)))].copy()
#     prop_may_2015 = data.loc[
#         (data['date'] >= pd.to_datetime(date(2015, 5, 1))) & (data['date'] < pd.to_datetime(date(2015, 6, 1)))].copy()
#
#     total_price_0514 = prop_may_2014['price'].mean()
#     total_price_0515 = prop_may_2015['price'].mean()
#
#     yoy_price = 100 * total_price_0515 / total_price_0514
#
#
#     # - H5: The mean month-over-month price growth of 3-bathroom-properties is 15% throughout all the months.
#
#     prop_3bath = data.loc[data['bathrooms'] == 3].copy()
#
#     prop_3bath['year-month'] = data['date'].dt.to_period('M')
#
#     prop_3bath_avgprice_month = prop_3bath[['year-month', 'price']].groupby(
#         'year-month').mean().reset_index().sort_values('year-month')
#
#     for row in range(len(prop_3bath_avgprice_month)):
#
#         if row == 0:
#             #         st.write('aloha')
#             prop_3bath_avgprice_month.loc[row, 'mom_growth'] = 0
#             continue
#
#         prop_3bath_avgprice_month.loc[row, 'mom_growth'] = 100 * (
#                 prop_3bath_avgprice_month.loc[row, 'price'] / prop_3bath_avgprice_month.loc[row - 1, 'price'] - 1)
#
#     mean_mom_price = prop_3bath_avgprice_month['mom_growth'].mean()
#
#     # - H6: Properties with 2 or more floors have an average grade 50% higher than the other properties.
#
#     meangrade_2plusfloors = data.loc[data['floors'] >= 2]['grade'].mean()
#     meangrade_2minusfloors = data.loc[data['floors'] < 2]['grade'].mean()
#
#
#     # - H7: The 'condition' attribute of a property is, on average, 30% higher on renovated properties.
#
#     meancondition_renovated = data.loc[data['yr_renovated'] > 0]['condition'].mean()
#     meancondition_notrenovated = data.loc[data['yr_renovated'] == 0]['condition'].mean()
#
#
#     # - H8: Properties with 2 or more bedrooms that have less than 2 bathrooms are 40% cheaper on average.
#
#     meanprice_2plusbed_less2bath = data.loc[(data['bedrooms'] >= 2) & (data['bathrooms'] < 2)]['price'].mean()
#     meanprice_2plusbed_2plusbath = data.loc[(data['bedrooms'] >= 2) & (data['bathrooms'] >= 2)]['price'].mean()
#
#
#     # - H9: Less than 15% of the properties have a basement and more than one floor above ground, and their average price is at least 20% higher than the average price of the whole portfolio.
#
#     numprop_basement_floors = data.loc[(data['sqft_basement'] > 0) & (data['floors'] > 1)]['id'].count()
#     total_prop = data['id'].count()
#
#     dataset_numprop_basement_floors = data.loc[(data['sqft_basement'] > 0) & (data['floors'] > 1)].copy()
#     dataset_meanprice = dataset_numprop_basement_floors['price'].mean()
#     portfolio_mean_price = data['price'].mean()
#
#
#     # - H10: Pareto: 80% of the portfolio's total cost lies in 20 to 25% of the properties.
#
#     total_cost = data['price'].sum()
#
#     data_sort_price = data.sort_values(by='price', ascending=False).copy()
#
#     pareto_sum = 0
#     pareto_count = 0
#
#     for row in range(len(data_sort_price)):
#
#         if pareto_sum <= 0.8 * total_cost:
#             pareto_sum = pareto_sum + data_sort_price.iloc[row]['price']
#             pareto_count = pareto_count + 1
#
#         else:
#             break
#
#     return None

# def commercial_distribution(data):
#     # Filters
#     st.sidebar.title('Commercial Options')
#     st.title('Commercial Attributes')
#
#     min_year_built = int(data['yr_built'].min())
#     max_year_built = int(data['yr_built'].max())
#
#     st.sidebar.subheader('Select Max Year Built')
#     f_yearbuilt = st.sidebar.slider('Year Built', min_year_built, max_year_built, max_year_built)
#
#     min_date = date.strptime(data['date'].min(), '%Y-%m-%d')
#     max_date = date.strptime(data['date'].max(), '%Y-%m-%d')
#
#     st.sidebar.subheader('Select max date')
#     f_date = st.sidebar.slider('Date', min_date, max_date, max_date)
#
#     # Average price per year
#     st.header('Average price per year built')
#     df = data.loc[data['yr_built'] < f_yearbuilt]
#     df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
#     fig = px.line(df, x='yr_built', y='price')
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Average price per day
#     st.header('Average price per day')
#     data['date'] = pd.to_datetime(data['date'])
#     df = data.loc[data['date'] < f_date]
#     df = df[['price', 'date']].groupby('date').mean().reset_index()
#     fig = px.line(df, x='date', y='price')
#     st.plotly_chart(fig, use_container_width=True)
#
#     return None
#
#
# def attributes_distribution(data):
#     # Histograms
#
#     # Filters
#     min_price = int(data['price'].min())
#     max_price = int(data['price'].max())
#     mean_price = int(data['price'].mean())
#
#     st.sidebar.subheader('Select Maximum Price')
#     f_price = st.sidebar.slider('Price', min_price, max_price, mean_price)
#
#     st.sidebar.subheader('Select Attributes')
#     f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', data['bedrooms'].sort_values().unique(),
#                                       index=(len(data['bedrooms'].sort_values().unique()) - 1))
#     f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', data['bathrooms'].sort_values().unique(),
#                                        index=(len(data['bathrooms'].sort_values().unique()) - 1))
#     f_floors = st.sidebar.selectbox('Max number of floors', data['floors'].sort_values().unique(),
#                                     index=(len(data['floors'].sort_values().unique()) - 1))
#     f_waterfront = st.sidebar.checkbox('Waterfront', value=False)
#
#     # Price Histogram
#     st.header('Price Distribution')
#     df = data.loc[data['price'] <= f_price]
#     fig = px.histogram(df, x='price', nbins=50)
#     st.plotly_chart(fig, use_container_width=True)
#
#     st.header('Attributes')
#     c1, c2 = st.columns(2)
#
#     # Bedrooms
#     c1.subheader('Bedrooms')
#     df = data.loc[data['bedrooms'] <= f_bedrooms]
#     fig = px.histogram(df, x='bedrooms', nbins=33)
#     c1.plotly_chart(fig, use_container_width=True)
#
#     # Bathrooms
#     c2.subheader('Bathrooms')
#     df = data.loc[data['bathrooms'] <= f_bathrooms]
#     fig = px.histogram(df, x='bathrooms', nbins=10)
#     c2.plotly_chart(fig, use_container_width=True)
#
#     # Floors
#     c1.subheader('Floors')
#     df = data.loc[data['floors'] <= f_floors]
#     fig = px.histogram(df, x='floors', nbins=10)
#     c1.plotly_chart(fig, use_container_width=True)
#
#     # Waterfront
#     c2.subheader('Waterfront')
#     if f_waterfront:
#         df = data.loc[data['waterfront'] == 1]
#     else:
#         df = data.loc[data['waterfront'] == 0]
#     fig = px.histogram(df, x='waterfront', nbins=2)
#     c2.plotly_chart(fig, use_container_width=True)
#
#     return None


# Dashboard requirements:
#
# - Table visualization and cards summarizing the original dataset
# - A map with the geographic location of the properties.
# - A set of filters that allows personalized analysis.
# - A table with the buy recommendations, considering the region and the median price.
# - A table with the buy recommendations, considering the region, the season, and the median price.
# - The total amount of recommended properties, maximum investment, and maximum profit made from it.

# EU PAREI NA INSERÇÃO DO OVERVIEW E DOS MAPAS, FALTA MOSTRAR OS RESULTADOS NO DASHBOARD E MOSTRAR AS HIPOTESES E RESULTADOS DELAS

if __name__ == '__main__':
    # data extraction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    data = get_data(path)
    geofile = get_geofile(url)

    # data transformation
    data = type_transform(data)
    data = set_features(data)

    # dashboard layout
    create_header()
    filtered_data = side_filters(data)

    # dataset analysis
    overview_data(filtered_data)
    portfolio_density(filtered_data, geofile)

    # set recommendation
    recommended_properties = recommend_properties(data, min_condition=3)
    # creates a visually better table
    recommendation_table = create_recomm_table(recommended_properties)

    # set prices on the recommended properties
    price_and_season_to_sell = set_price_and_season(recommended_properties)

    show_recommendations()

    # business hypothesis
    hypothesis()
