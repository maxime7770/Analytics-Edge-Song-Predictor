import pandas as pd

data = pd.read_csv('data/spotify_data.csv')

# number of rows with 'genre' is 'singer-songwriter'

print(data['genre'].value_counts())

# print unique values of genre
print(data['genre'].unique())