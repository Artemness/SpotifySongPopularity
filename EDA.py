import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark')
plt.style.use('dark_background')
import plotly.express as px
import plotly
from cycler import cycler

df = pd.read_csv('top10s.csv', encoding='ISO-8859-1')
# Removing the index (unnamed 0 column):
df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())

#renaming colmuns so that they are more understandable
df.rename(columns= {'title':'song', 'artist':'artist', 'top genre':'genre', 'year':'year', 'bpm':'beats_per_minute','nrgy':'energy',
                    'dnce':'danceability','dB':'volume','live':'liveness', 'val':'valence', 'dur':'length', 'acous':'acousticness',
                    'spch':'speechiness','pop':'popularity'}, inplace=True)

describ = df.describe()

#Creating a piechart for Genres and % of Songs in Top 10 Genre bucket for each
genrescount = df['genre'].value_counts().head(10)
fig1, ax1 = plt.subplots()
custom_cycler = (cycler(color=['#FF0000', '#FF7400', '#208051', '#AB5A09', '#0DA006', '#009BFF', '#02243A', '#000000', '#550083','#83006B']))
ax1.set_prop_cycle(custom_cycler)
ax1.pie(genrescount, labels=genrescount.index, autopct='%1.1f%%', startangle=0)
plt.title('Top 10 Genres Percentages')
plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "grey",
    "axes.facecolor": "white",
    "axes.edgecolor": "grey",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white"})
plt.show()
fig1 = ax1.get_figure()
fig1.savefig('Top 10 Genres Percentages.png', transparent=True)


#Dissecting the Top Artists from the data:
artists = df['artist'].value_counts().head(10)
plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "grey",
    "axes.facecolor": "black",
    "axes.edgecolor": "grey",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white"})
fig2, ax2 = plt.subplots()
custom_cycler = (cycler(color=['#04F907']))
ax2.set_prop_cycle(custom_cycler)
ax2.bar(x=artists.index, height=artists)
plt.title('Top 10 Artists')
plt.xticks(rotation='45')
plt.show()
fig2 = ax2.get_figure()
fig2.savefig('Top 10 Artists.png', transparent=True)

#Creating a chart for danceability by year:
fig3 = px.violin(df, y="danceability", color="year", points='all', hover_name='song', hover_data=['artist'])
with open('danceability.html', 'w') as f:
    f.write(fig3.to_html(include_plotlyjs='True'))

#Creating a chart for popularity by year:
fig4 = px.violin(df, y="popularity", color="year", points='all', hover_name='song', hover_data=['artist'])
plotly.offline.plt(fig4, filename='popularity.html')

#Creating a chart for popularity of all songs in year 2016:
fig5 = px.scatter(df.query("year==2016"), y="popularity", x="artist", hover_name='song', color='popularity')
plotly.offline.plt(fig5, filename='MostPopular2016.html')

#Creating a correlation heatmap
correlations = df.corr()
fig7, ax7 = plt.subplots()
fig7 = plt.figure(figsize=(18, 12))
heatmap = sns.heatmap(correlations, annot=True, cmap='BuPu', center=1)
fig7 = heatmap.get_figure()
fig7.savefig('heatmap.png', transparent=True)

df.to_csv('Cleaneddata.csv', index=False)