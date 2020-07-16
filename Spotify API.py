import pandas as pd
import requests
import os
import datetime
import base64
from urllib.parse import urlencode
import json
import pickle

client_id= '58d8eea6f9a84b21a79450019b32eb53'
secretdir = ('./apisecret.txt')
with open(secretdir) as f:
    client_secret = f.read().strip()
print(client_secret)


class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"

    def __init__(self, client_id, client_secret, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret

    def get_client_credentials(self):
        """
        Returns a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()

    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization": f"Basic {client_creds_b64}"
        }

    def get_token_data(self):
        return {
            "grant_type": "client_credentials"
        }

    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            raise Exception("Could not authenticate client.")
            # return False
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in']  # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True

    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token()
        return token

    def get_resource_header(self):
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        return headers

    def get_resource(self, lookup_id, resource_type='albums', version='v1'):
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def get_album(self, _id):
        return self.get_resource(_id, resource_type='albums')

    def get_artist(self, _id):
        return self.get_resource(_id, resource_type='artists')

    def base_search(self, query_params):  # type
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"
        lookup_url = f"{endpoint}?{query_params}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def search(self, query=None, operator=None, operator_query=None, search_type='artist'):
        if query == None:
            raise Exception("A query is required")
        if isinstance(query, dict):
            query = " ".join([f"{k}:{v}" for k, v in query.items()])
        if operator != None and operator_query != None:
            if operator.lower() == "or" or operator.lower() == "not":
                operator = operator.upper()
                if isinstance(operator_query, str):
                    query = f"{query} {operator} {operator_query}"
        query_params = urlencode({"q": query, "type": search_type.lower()})
        print(query_params)
        return self.base_search(query_params)

spotify = SpotifyAPI(client_id, client_secret)
track_name = input('Enter Name of Song to pull some stats from Spotify:')
track_name.replace(' ', '%20')

#sample = {}
sampledf = pd.DataFrame(columns = ['artist', 'genre', 'beats_per_minute', 'energy', 'danceability', 'volume', 'liveness', 'valence', 'length', 'acousticness', 'speechiness'])
sampledf.loc[0] = ['NA', 'NA','0','0','0','0','0','0','0','0','0']
songs = spotify.search(track_name, search_type="track")
artist_id = songs['tracks']['items'][0]['artists'][0]['id']
artist = songs['tracks']['items'][0]['artists'][0]['name']

track_id = songs['tracks']['items'][0]['id']

track_url = 'https://api.spotify.com/v1/audio-features/'+track_id
r = requests.get(track_url, headers = spotify.get_resource_header())
track_stats = r.json()

artist_url = 'https://api.spotify.com/v1/artists/'+artist_id
r2 = requests.get(artist_url, headers = spotify.get_resource_header())
artist_stats = r2.json()

tracks2_url = 'https://api.spotify.com/v1/tracks/'+track_id
r3 = requests.get(tracks2_url, headers = spotify.get_resource_header())
trackpop = r3.json()

#sample.update({'artist' : artist_stats['name']})
sampledf['artist'].iloc[0] = artist_stats['name']

#updating Genres of
genres = artist_stats['genres']

#sample.update({'genre' : genre})
sampledf['genre'].iloc[0] = str(genres)

#adding Beats per minute to the Sample:
#sample.update({'beats_per_minute': round(track_stats['tempo'],0)})
sampledf['beats_per_minute'] = round(track_stats['tempo'],0)

#Adding Energy to the Sample:
#sample.update({'energy': track_stats['energy']})
sampledf['energy'].iloc[0] = track_stats['energy']

#Adding danceability to the Sample:
#sample.update({'danceability' : track_stats['danceability']})
sampledf['danceability'].iloc[0] = track_stats['danceability']

#Adding Volume to the Sample:
#sample.update({'volume' : track_stats['loudness']})
sampledf['volume'].iloc[0] = track_stats['loudness']

#Adding liveness to the Sample:
#sample.update({'liveness' : track_stats['liveness']})
sampledf['liveness'].iloc[0] = track_stats['liveness']

#adding valence to Sample:
sampledf['valence'].iloc[0] = track_stats['valence']

#Adding length to the Sample:
lentrack = round(track_stats['duration_ms'])
#sample.update({'length' : lentrack})
sampledf['length'].iloc[0] = lentrack

#Adding accousticness to the Sample:
#sample.update({'acousticness' : track_stats['acousticness']})
sampledf['acousticness'].iloc[0] = track_stats['acousticness']

#Adding speechiness to the Sample:
#sample.update({'speechiness' : track_stats['speechiness']})
sampledf['speechiness'].iloc[0] = track_stats['speechiness']

samplepopularity = trackpop['popularity']
sampleyear = trackpop['album']['release_date'][:4]


pickle_in = open('gbm.pickle','rb')
gbm = pickle.load(pickle_in)

file = open("artistexpanded.obj",'rb')
artistle = pickle.load(file)
file.close()
try:
    sampledf['artist'] = artistle.transform(sampledf['artist'])
except:
    sampledf['artist'] = 0

file = open("genreexpanded.obj",'rb')
genristle = pickle.load(file)
file.close()
try:
    sampledf['genre'] = genristle.transform(sampledf['genre'])
except:
    sampledf['genre'] = 0

print(sampledf)
print(track_name)
print(artist)
print(sampleyear)
print(samplepopularity)
print(gbm.predict(sampledf.values))