
# SpotifySongPopularity
Most Popular Songs on the Spotify Platform.

## Data Collection:  
Utilized Two Data sets available from Kaggle.

## Data Cleaning:  
Cleaned up the data and then proceeded to run EDA on the data to produce some visualizations.

## EDA:  
<h3> To better understand the first dataset I took a look at the top 10 genres represented in the data</h3>  
<img src= "/Top 10 Genres Percentages NonTransparent.png" height=500 width=700>  
<h3>From this we can see that a large percentage of the songs in this dataset are Pop Songs and variants of Pop. Therefore training a machine learning model with this dataset would be a little challenging as it would only predict popularity fairly well for pop songs.</h3>  
  <br>
<h3>Checking the top artists in the data set to better understand the data:</h3>  
   <ul><li>Katy Perry, Justin Bieber and Maroon 5 & Rihanna round out the Top 3</li></ul>
      <img src='/Top 10 Artists NonTransparent.png'>
      
## Model Building:  
### Ran a number of models:Multi Variable Linear Regression, Gradient Boosting, Decision Tree, Random Forest and LightGBM Models to predict popularity given the other features of the data:  
## Model Performance:
### The LightGBM Model proved the most efficient. Therefore chose to package the LightGBM for production due to lighter computing speed. Loss function: MAE

* Multi Variable Linear Regression : 12.96
* Gradient Boosting: 9.82
* Decision Tree: 11.47
* Random Forest: 8.55
* Light GBM: 8.22

 ## Productionalizing:
 Established a web interface which allows a user to input a song name and utilizes the Spotify API to pull the characteristics for the song then utilizes the optimized LightGBM to create a prediction for the input.  
(Can be found on my Website - <a href="https://www.artemmoshkin.com">Visit ArtemMoshkin.com!</a>)
 
