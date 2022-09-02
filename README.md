![header-image](/image/real-estate.jpg)


> Disclaimer: All the processed data here comes from a public dataset found on [Kaggle](https://www.kaggle.com/datasets/shivachandel/kc-house-data), and the House Rocket Company is a fictional company. Any similarity with the real world should be ignored.

#  Business problem

The House Rocket company has a portfolio of available properties to buy and the CEO must be able to explore said dataset and also requires selling recommendations to assist the sales team.

## Delivery

To satisfy the CEO's needs, an interactive dashboard should be presented, containing the data visualizations and the buy/sell recommendations regarding the features on the dataset. 

Dashboard requirements:

- Table visualization and cards summarizing the original dataset
- A map with the geographic location of the properties.
- A set of filters that allows personalized analysis.
- A table with the buy recommendations, considering the region and the median price.
- A table with the buy recommendations, considering the region, the season, and the median price.
- The total amount of recommended properties, maximum investment, and maximum profit made from it.


## Strategy

### Business problem assumptions

- The dataset's 'date' column is the date the property was made available on the market.

- Only the properties within the dataset were available from (min_date) to (max_date)
- One of the properties has the value of 33 bedrooms but wasn't the largest on the dataset, so it was considered an outlier and removed.
- Duplicated properties were not removed as that was considered a second attempt of selling the house, during a different season of the year.
- Seasons were considered as follows:
    - Spring: from March 21st to June 20th
    - Summer: from June 21st to September 20th
    - Autumn: from September 21st to December 20th
    - Winter: from December 21st to March 20th

- Here's an overview of the original dataset and a short description for each column.


|column|column description|
|:--------------|:------------------------------------------------------------------|
| id            | identification number for each property                           |
| date          | the date the property was available                               |
| price         | sale price                                                        |
| bedrooms      | number of bedrooms                                                |
| bathrooms     | number of bathrooms                                               |
| sqft_living   | living room size in square feet                                   |
| sqft_lot      | property size in square feet                                      |
| floors        | number of floors                                                  |
| waterfront    | wether or not the property is waterfront                          |
| view          | how good is the view from the property, ranked from 0 to 4        |
| condition     | condition of the property, ranked from 1 to 5                     |
| grade         | grade given to the property, from 1 to 13                         |
| sqft_above    | size of the floors above ground in square foot                    |
| sqft_basement | size of the basement in square foot                               |
| yr_built      | year of construction                                              |
| yr_renovated  | year of renovation                                                |
| zipcode       | region zipcode                                                    |
| lat           | latitude value                                                    |
| long          | longitude value                                                   |
| sqft_living15 | mean living room size of the 15 nearest properties in square foot |
| sqft_lot15    | mean property size of the 15 nearest properties in square foot    |

### Business questions and solving method

#### Given the different regions, which properties should House Rocket buy?

To guarantee profit for the company, one can compare each property's price with the median price in its region and discard it, if higher. Another important feature to consider is its condition. As it goes from 1 to 5, it should be at least 3.


#### Once bought, for how much and when should the company sell each property?

In what refers to the season to sell, the recommendation is to sell the property in the same season it has been put for sale. This should decrease the influence of this feature on the price. For this, the recommended properties can now be grouped by the region (zip code) and offering season, finding once again each median price.

The selling price can be determined as a function of its grade, as shown below.

$$ pricetosell = price * \bigg(1 + \big(\frac{propertygrade}{maxgraderegion} * R\big)\bigg), R = \begin{cases}
      \text{0.2 if the property price is higher than the group's median price}\\
      \text{0.5 if the property price is lower than the group's median price} 
      \end{cases}    $$
    
    
Example:
   
   |        id |   zipcode |   price | region median price|   grade/region_max_grade |    R | result|  price_to_sell |   profit |
   |----------:|----------:|--------:|:-------------------|-------------------------:|-----:|------:|---------------:|---------:|
   | 123456789 |     91234 |  100000 | 90000              |                      3/10|  0.2 |   1.06|         106000 |    6000  |
   | 987654321 |     94321 |  100000 | 200000             |                       4/5|  0.5 |   1.4 |         140000 |    40000 |


## Results

### Numbers

With the specified conditions, from all the 21,612 total properties, **10,728** of them are recommended to buy and the total amount invested would be US$ 4,159,218,287.00

Following this recommendation and applying the pricing rule above, the maximum profit is **US$ 1,165,681,056.16**

Samples of the recommendation table and the pricing table are shown below, and the full dashboard can be seen [here]([https://google.com](https://project-house-rocket-insights.herokuapp.com/)).

<p style="text-align: center;">  <b>Recommendation Table</b> </p>

|         id |   zipcode |   condition |   price | recommendation   |
|:-----------|:----------|-----------:|--------:|:-----------------:|
| 9543000205 |     98001 |           4 |  US\$ 139,950.00 | recommended      |
| 3353401070 |     98001 |           3 |  US\$ 260,000.00 | recommended      |
| 1311300100 |     98001 |           3 |  US\$ 221,000.00 | recommended      |
| 3751604895 |     98001 |           4 |  US\$ 165,000.00 | recommended      |
| 3751600146 |     98001 |           3 |  US\$ 166,000.00 | recommended      |
| 3328500250 |     98001 |           3 |  US\$ 285,000.00 | not recommended  |
| 3275910020 |     98001 |           3 |  US\$ 340,000.00 | not recommended  |
| 8856000545 |     98001 |           3 |  US\$ 100,000.00 | recommended      |
| 3356403820 |     98001 |           3 |  US\$ 115,000.00 | recommended      |
| 2895550330 |     98001 |           3 |  US\$ 290,000.00 | not recommended  |


<p style="text-align: center;">  <b>Pricing Table</b> </p>

|         id |   zipcode |       price | season_to_sell   |   price_to_sell |   profit |
|:-----------|:----------|------------:|:-----------------|----------------:|---------:|
| 3738000070 | 98039 | US\$ 1,712,750.00 |winter | US\$ 2,414,977.50 | US\$ 702,227.50 |
| 3262301355 | 98039 | US\$ 1,320,000.00 |summer | US\$ 1,914,000.00 | US\$ 594,000.00 |
| 3835502815 | 98039 | US\$ 1,260,000.00 |autumn | US\$ 1,820,700.00 | US\$ 560,700.00 |
| 5427110040 | 98039 | US\$ 1,225,000.00 |spring | US\$ 1,776,250.00 | US\$ 551,250.00 |
| 2525049133 | 98039 | US\$ 1,398,000.00 |spring | US\$ 1,887,300.00 | US\$ 489,300.00 |
| 3262300920 | 98039 | US\$ 1,200,000.00 |spring | US\$ 1,680,000.00 | US\$ 480,000.00 |
| 3625049079 | 98039 | US\$ 1,350,000.00 |summer | US\$ 1,822,500.00 | US\$ 472,500.00 |
| 6447300345 | 98039 | US\$ 1,160,000.00 |spring | US\$ 1,624,000.00 | US\$ 464,000.00 |
| 5426300060 | 98039 | US\$ 1,000,000.00 |autumn | US\$ 1,445,000.00 | US\$ 445,000.00 |
| 3262301610 | 98039 | US\$   865,000.00 |autumn | US\$ 1,249,925.00 | US\$ 384,925.00 |

### Insights

To find insights into this problem, 10 hypotheses were defined and tested on whether or not they were true. One can check all of them out on the dashboard, but here are some of them, which are more impactful on the decision-making process.

| Hypothesis | Validity | Actual result | Business meaning   |
|:-----------|:----------|:------------|:-----------------|
| H1: Waterfront properties are on average <b>30%</b> more expensive. | False | Waterfront properties are <b>312.64%</b> more expensive. | Buying more non-waterfront properties than waterfront ones should bring more significant profits, once less money is spent and risked. |
| H7: The 'condition' attribute of a property is, on average, <b>30% higher</b> on renovated properties. | False | The average condition on renovated properties is <b>5.69% lower</b>. | The renovation process has little to no effect on the condition of the properties, therefore this attribute shouldn't affect the prices. |
| H9: Less than 15% of the properties have a basement and more than one floor above ground, and their average price is at least 20% higher than the average price of the whole portfolio. | True | Only 13.71% of the properties have a basement and more than one floor above ground. Their average price is 51.97% higher than the average price of the portfolio. | This type of property should be avoided since its price is usually higher than average, which could diminish profits. |
| H10: Pareto: 80% of the portfolio's total cost lies in <b>20 to 25%</b> of the properties | False | 80.00% of the portfolio <b>total cost lies in 62.21%</b> of the properties. | As the portfolio's total price is well distributed among the properties, the investment risk drops significantly as it should be easier to sell cheaper properties rather than more expensive ones. |

