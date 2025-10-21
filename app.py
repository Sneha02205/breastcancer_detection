from flask import Flask, render_template, url_for, request, redirect
from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import tweepy  # moved import up for clarity

app = Flask(__name__)

# ==============================
# YouTube Section
# ==============================
api_key_youtube = 'AIzaSyCFSEd7xI7IAh4X67Z07dagb0z5odMWjUA'
youtube = build('youtube', 'v3', developerKey=api_key_youtube)

def get_channel_stats(youtube, ch_username):
    all_data = []
    try:
        request = youtube.channels().list(
            part="snippet,contentDetails,statistics",
            forUsername=ch_username
        )
        response = request.execute()

        if 'items' not in response or len(response['items']) == 0:
            return [{"Channel_name": "Not Found", "Subscribers": "0", "Description": "N/A", "Views": "0", "Total_videos": "0"}]

        for i in range(len(response['items'])):
            data = dict(
                Channel_name=response['items'][i]['snippet']['title'],
                Subscribers=response['items'][i]['statistics'].get('subscriberCount', '0'),
                Description=response['items'][i]['snippet'].get('description', 'No description'),
                Views=response['items'][i]['statistics'].get('viewCount', '0'),
                Total_videos=response['items'][i]['statistics'].get('videoCount', '0')
            )
            all_data.append(data)
    except Exception as e:
        all_data.append({
            "Channel_name": "Error",
            "Subscribers": "0",
            "Description": str(e),
            "Views": "0",
            "Total_videos": "0"
        })

    return all_data


# ==============================
# Twitter Section
# ==============================
api_key_twitter = 'z9CREzdGc3vxdmHjFVZ724XeK'
api_key_secret = '5JAED00x8ohda2jn6Bm15Jhb8HQs2KMGOGfLnC8hkLMhnZqpu2'
access_token = '1208139911766925312-6ka8h7pjEdaEWbkSnp6VsTL6GxIHDe'
access_token_secret = 'WKSmnEHYVKXzbvueGlvDMvDjVdm4oj60sV0jlBxmhu2lu'

# Authenticate with Tweepy
auth = tweepy.OAuthHandler(api_key_twitter, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# ==============================
# Trending Lists Section
# ==============================
def fetch_table(url):
    """Helper to fetch a single Wikipedia table safely."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        df = pd.read_html(str(table))[0]
        return df
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return pd.DataFrame()


df = fetch_table("https://en.wikipedia.org/wiki/List_of_most-subscribed_YouTube_channels")
df1 = fetch_table("https://en.wikipedia.org/wiki/List_of_most-followed_Instagram_accounts")
df2 = fetch_table("https://en.wikipedia.org/wiki/List_of_most-followed_Facebook_pages")
df3 = fetch_table("https://en.wikipedia.org/wiki/List_of_most-followed_Twitter_accounts")
df4 = fetch_table("https://en.wikipedia.org/wiki/List_of_most-followed_TikTok_accounts")
df5 = fetch_table("https://en.wikipedia.org/wiki/List_of_most-followed_Twitch_channels")

# Combine data safely
try:
    data = df[['Rank', 'Name']]
    data1 = df1[['Owner']]
    data2 = df2[['Page name']]
    dattt = df4.rename(columns={"Owner": "sp"})
    data4 = dattt[['sp']]
    data5 = df5[['Channel']]

    da = pd.concat([data, data1, data2, data4, data5], axis=1)
    dst = da.rename(columns={"Name": "ytaccts", "Owner": "igaccts",
                             "Page name": "fbaccts", "Channel": "twaccts", "sp": "tikaccts"})

    ytaccounts = dst['ytaccts'].head(10)
    igaccounts = dst['igaccts'].head(10)
    fbaccounts = dst['fbaccts'].head(10)
    twaccounts = dst['twaccts'].head(10)
    tikaccounts = dst['tikaccts'].head(10)
except Exception as e:
    print("Error processing trending data:", e)
    ytaccounts = igaccounts = fbaccounts = twaccounts = tikaccounts = []


# ==============================
# Routes
# ==============================
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        platform = request.form.get('platform')
        channelname = request.form.get('channelname')

        if platform == 'youtubedetails':
            return redirect(url_for('youtubedetails', channelname=channelname))
        elif platform == 'twitterdetails':
            return redirect(url_for('twitterdetails', accountname=channelname))
        else:
            return "Wrong input"
    return render_template('indexmain.html')


@app.route('/youtubedetails/<channelname>', methods=['GET'])
def youtubedetails(channelname):
    youtubedata = get_channel_stats(youtube, channelname)
    yt = youtubedata[0]
    return render_template(
        'youtubedetails.html',
        channelname=channelname.upper(),
        subcount=str(yt['Subscribers']).upper(),
        viewcount=str(yt['Views']).upper(),
        channeldesc=yt['Description'],
        vidcount=str(yt['Total_videos']).upper()
    )


@app.route('/twitterdetails/<accountname>', methods=['GET'])
def twitterdetails(accountname):
    try:
        user2 = api.get_user(screen_name=accountname)
        return render_template(
            'twitterdetails.html',
            handlename=accountname.upper(),
            acc_desc_tt=user2.description,
            followcount_tt=user2.followers_count,
            createdate=(user2.created_at).strftime("%d/%m/%Y"),
            followingcount_tt=user2.friends_count,
            contact=user2.url
        )
    except Exception as e:
        return f"Error fetching Twitter data: {str(e)}"


@app.route('/about')
def aboutpage():
    return render_template('about.html')


@app.route('/livecount')
def livecounter():
    return render_template('livecount.html')


@app.route('/top')
def toppage():
    return render_template('top.html',
                           ytaccounts=ytaccounts,
                           igaccounts=igaccounts,
                           tikaccounts=tikaccounts,
                           twaccounts=twaccounts,
                           fbaccounts=fbaccounts)


@app.route('/compare')
def comparepage():
    return render_template('compare.html')


if __name__ == "__main__":
    app.run(debug=True)
