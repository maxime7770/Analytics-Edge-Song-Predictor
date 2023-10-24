import pandas as pd

data = pd.read_csv('data/spotify_data_processed.csv')

index = 879020

# drop na 
data = data.dropna()

print(data.iloc[index]['track_name'])


# check is every track name is a string
for track_name in data['track_name']:
    if type(track_name) != str:
        print(track_name)
        print(type(track_name))
        break