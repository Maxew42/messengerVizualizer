import numpy as np
import re
import pandas as pd
import json
from functools import partial
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

### Configuration constants ###
# DEV : should be fetched from text file

posWord = {'😇', '😘', '🙂', '💃', '🍻', '😏', '🥰', '😋', '🤣', '😍', '😁', '😆', '❤️', '😉', '😂', 'merci', 'cool', 'super', 'wouaw', 'j\'adore', 'bien', 'bon',
           'chaud', 'intéressé', 'craque', 'ahah', 'hihi', 'haha', 'ahahah', 'hahaha', 'excellente', 'saucé', 'perfect', 'mdr', 'enormous', 'epic', 'hahahaha'}
negWord = {'😒', '😠', '🥺', '😬', '😤', '🙄', '🥵', '😥', '😰', 'hélas', 'zoo', 'ridicule', 'honte',
           'honteux', 'fous', 'putain', 'merde', 'relou', 'flemme', 'mauvaise', 'déteste', 'mais', 'triste'}
emojiSet = {'😇', '🤔', '😘', '🍻', '😳', '🤝', '🌚', '🙂', '✅', '😈', '😠', '😒', '🙃', '👋', '🥺', '💃', '👀', '😏', '🤷🏻', '🥬', '🐼', '😬', '😴', '🥰', '🤪', '😋',
            '👌', '🙌', '🤣', '😍', '😤', '😁', '😆', '🤞', '🎃', '❣️', '🌙', '💕', '👍', '🙄', '🤡', '❤️', '😉', '😮', '🙈', '😅', '😂', '🤓', '😰', '😊', '🥵', '😥', '☺️', '😆'}
TIME_AREA = {'Late Night (0-6)': [0, 6], 'Morning (6-10)': [6, 10], 'Noon (10-14)': [
    10, 14], 'Afternoon (14-18)': [14, 18], 'Evening (18-22)': [18, 22], 'Night (22-24)': [22, 24]}

### Function definition ###

def get_dom(dt):
    return dt.day

def get_dow(dt):
    return dt.weekday()

def get_hour(dt):
    return dt.hour

def count_rows(rows):
    return len(rows)

def scaled_count_rows(rows):
    return np.log(len(rows))**2.4

def getMinutesDuration(time1,time2):
    return (time2-time1) / pd.Timedelta(minutes=1)

def light_scaled_count_rows(rows):
    return np.log(len(rows))**1.8

def sigmoid(x,multiScale=1,offset=0):
    '''Easy to tune sigmoid function'''
    return 1/(1+ np.exp(-(x+offset)*multiScale))

def grade(value,bornInf=1,bornSup=6,multiScale=1,offset = 0):
    '''Tunable notation tool'''
    return bornInf + (bornSup-bornInf)*sigmoid(value,multiScale=multiScale,offset = offset)

def loadFacebookJson(filePath):
    # Kindly provided by Luksan on StackOverflow (https://stackoverflow.com/users/1889274/luksan)
    fix_mojibake_escapes = partial(
        re.compile(rb'\\u00([\da-f]{2})').sub,
        lambda m: bytes.fromhex(m.group(1).decode()))

    with open(filePath, 'rb') as binary_data:
        repaired = fix_mojibake_escapes(binary_data.read())
    return json.loads(repaired.decode('utf8'))


def getEmojiCount(df, emojiSet=emojiSet):
    emojiCountDict = {i: 0 for i in emojiSet}
    for index, row in df.iterrows():
        if row.type == "text":
            for emoji in emojiSet:
                if emoji in row.message:
                    emojiCountDict[emoji] += 1
    return emojiCountDict


def getTimeArea(time, timeArea): #DEV : Bad function, redo it using set and `in` keyword.
    for key in timeArea.keys():
        if time >= timeArea[key][0] and time < timeArea[key][1]:
            return key
    return None


def plotTimeArea(df, timeArea, ax):

    df['timeArea'] = df['hour'].map(lambda x: getTimeArea(x, timeArea))
    a = df.groupby('timeArea').count().sort_values('sender', ascending=False)

    ax.bar(a.index.to_list(), a['sender'].to_list())
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel("Part of the day")
    ax.set_ylabel("Number of messages")
    return ax


def loadFacebookDf(dataDict):
    df = pd.DataFrame(columns={"sender", "message", "time",
                      "messageWordLength", "messageCharacterLength", "type"})
    dfReactions = pd.DataFrame(
        columns={"reaction", "sender", "messageIndex", "messageSender"})
    threadInfo = {}

    cpt = 0
    #path_to_clean_message = "../cleanData/"
    #paths_to_files = glob.glob(
    #    path_to_clean_message + "/" + filename + "/" + "message_*.json")

    for key in sorted(dataDict):
        data = dataDict[key]
        print(key)
        if cpt == 0:

            nbParticipants = len(data['participants'])
            nbMessages = len(data['messages'])
            title = data['title']
            threadType = data['thread_type']
            threadInfo = {'nbParticipants': nbParticipants,
                          'nbMessages': nbMessages, 'title': title, 'threadType': threadType}
        else:
            threadInfo['nbMessages'] = threadInfo['nbMessages'] + \
                len(data['messages'])
        for index, row in enumerate(data['messages']):

            messageType = row['type']

            if messageType == 'Generic':
                messageCharacterLength = 0
                messageWordLength = 0
                content = None
                timestamp = row['timestamp_ms']
                mType = "text"

                if 'content' in row.keys():
                    content = row["content"]
                    messageCharacterLength = len(content)
                    messageWordLength = len(content.split(" "))
                elif 'photos' in row.keys():
                    mType = "photo"
                elif 'videos' in row.keys():
                    mType = 'video'
                elif 'audio_files' in row.keys():
                    mType = 'audio'
                elif 'gifs' in row.keys():
                    mType = 'gif'
                elif 'sticker' in row.keys():
                    mType = 'sticker'
                else:
                    mType = 'deleted'

                df = df.append({'sender': row['sender_name'], "messageWordLength": messageWordLength,
                               "messageCharacterLength": messageCharacterLength, "time": timestamp, "message": content, "type": mType}, ignore_index=True)
            if 'reactions' in row.keys():
                for reaction in row['reactions']:
                    dfReactions = dfReactions.append(
                        {"reaction": reaction['reaction'], "sender": reaction['actor'], "messageIndex": index, "messageSender": row['sender_name']}, ignore_index=True)
        cpt += 1

    df["time"] = pd.to_datetime(df['time'], unit='ms')
    df["hour"] = df["time"].map(get_hour)
    df["dow"] = df["time"].map(get_dow)
    df["dom"] = df["time"].map(get_dom)
    df['date'] = df["time"].dt.date
    df['messageWordLength'] = df['messageWordLength'].astype(int)
    df['messageCharacterLength'] = df['messageCharacterLength'].astype(int)
    df['emotion'] = df[['message', 'type']].apply(
        lambda x: getDfEmotion(x, posWord, negWord), axis=1)

    threadInfo['threadLifetime'] = (
        df.loc[0, "time"]-df.loc[df.shape[0]-1, "time"]) / pd.Timedelta(weeks=1)
    threadInfo['nbReactions'] = dfReactions.shape[0]
    df = df.reindex(index=df.index[::-1])
    df = df.reset_index()
    return df, threadInfo, dfReactions


def plotUsageOverTime(df):
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs[0, 0].hist(df[['hour']], bins=24, rwidth=0.8, range=(-0.5, 23.5))
    axs[0, 1].hist(df['dow'], bins=7, range=(-0.5, 6+0.5))
    axs[0, 1].set_xticklabels(
        ["", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    axs[1, 0].hist(df['dom'], bins=31, range=(1, 31))
    plotTimeArea(df, TIME_AREA, axs[1, 1])
    axs[0, 0].set(title='Messages frequency by Hour of the day',
                  xlabel="Hours", ylabel="Frequency")
    axs[0, 1].set(title='Messages frequency by week day',
                  xlabel="Weekday", ylabel="Frequency")
    axs[1, 0].set(title='Messages frequency by day of the month',
                  xlabel="Day of the month", ylabel="Frequency")
    fig.tight_layout()
    return fig


def plotEmojiUsageDistribution(df, emojiSet=emojiSet):
    df1 = pd.DataFrame(columns={'sender', 'time', 'emoji'})
    for index, row in df.iterrows():
        if row.type == "text":
            for emoji in emojiSet:
                for char in row.message.split():
                    if emoji == char:
                        df1 = df1.append(
                            {'sender': row.sender, 'time': row.time, 'emoji': emoji}, ignore_index=True)

    sns.set_style({'font.family': 'Segoe UI Emoji',
                  'font.serif': ['Times New Roman']})
    df2 = df1.groupby(['sender', 'emoji']).apply(
        light_scaled_count_rows).unstack()
    fig, ax = plt.subplots(figsize=(22, 8))
    sns.heatmap(df2, linewidths=.5, cmap=sns.color_palette("crest", as_cmap=True)).set(
        title='Scaled Distribution of sent emojis by senders')
    sns.set(font_scale=2)
    plt.xticks(rotation=0)
    return fig


def filterText(df):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    ownSpecificWords = ['vraiment', 'pense', 'coup','tous','rien','aujourd','quelqu','mettre','seul','cas','bon'
    ,'hui','user','m','doit','dire','crois','normal','dis','semble','output','celui', 'https://', 'disponible sondage', 'disponible', 'https', 'dit', 'déjà', 'truc', 'coup', 'juste', 'comme', 'être', 'donc', 'après',
                        'vrai', 'oui', 'dois', 'enfin', 'effet', 'moins', 'sinon', 'peux', 'parce', 'vu', 'non', 'fais', 'quand', 'aussi', 'trop', 'peu', 'avoir', 'autre', 'sans', 'jusqu', 'm', 'iloo', 'liandja', 'ba', 'fini', 'sais', 'p', 'meignan', 'rw', 'r', 'envoyé', 'cela', 'quelle']
    ownStopWords = ['.', '?', 'c\'est', 'ça', 'est', 'c', "J\'a", 'w', 'j\'ai', 'où', ':', 'cette', 'là', 't', 'tout', 'l', "quoi",
                    'j', 'ai', '’', 'l’', 'j’ai', 'c’est', '1', 'ca', 'fait', 'va', 'si', 'faut', 'peut', 'plus', 'j’', 'a', '!', 'faire', 'bien', ]
    stopWords = stopwords.words('french')
    stopWords = stopWords + ownStopWords + ownSpecificWords
    df1 = df[df.type == "text"]
    text = (" ".join(df1['message'].to_list()))
    text = re.sub(r'[\'`’?!.@]', " ", text)
    text = text.split(" ")
    filtered_text = [word for word in text if not word.lower() in stopWords]
    filtered_text = " ".join(filtered_text)
    return filtered_text


def plotWordCloud(filtered_text):
    word_cloud = WordCloud(normalize_plurals=False, stopwords=['oui'], max_words=150, colormap='cividis', min_font_size=1,
                           background_color="white", width=1500, height=800, font_path='../ressources/font/coolvetica rg.ttf').generate(filtered_text)
    fig = plt.figure(figsize=(19, 19))  # 'RdPu' 'cividis'
    plt.imshow(word_cloud)
    plt.axis("off")
    return fig


def plotMessagesType(df):
    df1 = df.groupby(['sender', 'type']).apply(scaled_count_rows).unstack()
    fig = plt.figure(figsize=(25, 10))
    sns.heatmap(df1, linewidths=.5, cmap=sns.color_palette("crest", as_cmap=True), xticklabels=True, yticklabels=True).set(
        title='Scaled Distribution of messages type by senders')
    return fig


def plotGeneralInfo(df, threadInfo):

    fig, ax = plt.subplots(1, 2, figsize=(19, 10))

    # PLOT 1
    size = 0.3

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(([2*i for i in range(10)]))

    df1 = df.groupby(['sender', 'emotion']).apply(count_rows).unstack()
    df1.sort_values(by=['Neutral'], inplace=True)
    df1 = df1.fillna(0)
    df2 = df1[['Negative', 'Positive']]
    df2['Negative'] = df2['Negative'] + df1['Neutral']/2
    df2['Positive'] = df2['Positive'] + df1['Neutral']/2
    npDf1 = df1.to_numpy()
    npDf2 = df2.to_numpy()
    explode1 = [0.025 for i in range(npDf1.sum(axis=1).shape[0])]
    explode2 = [0.00 if i % 2 != 0 else 0.0 for i in range(
        int(npDf2.sum(axis=1).shape[0]*2))]
    patches1, texts = ax[0].pie(npDf1.sum(axis=1), colors=outer_colors,
                                wedgeprops=dict(width=size, edgecolor='w'), explode=explode1)

    patches2, texts = ax[0].pie(npDf2.flatten(), radius=1-size, colors=['#61C585', '#B45B47'],
                                wedgeprops=dict(width=0.08, edgecolor='w'), explode=explode2)

    ax[0].set(aspect="equal", title='Message Emotion grouped by senders')
    ax[1].set(aspect="equal", title='General Statistics')
    legendName = ax[0].legend(patches1, df1.index.to_list(), loc=1, fontsize=7)
    plt.axis('off')
    legendEmotion = ax[0].legend(
        patches2, ["Positive", "Negative"], loc=4)
    ax[0].add_artist(legendName)
    # ax[0,0].add_artist(legendEmotion)

    # PLOT 2

    orderedEmojiCount = getEmojiCount(df)
    orderedEmojiCount = {k: v for k, v in sorted(
        orderedEmojiCount.items(), key=lambda item: item[1], reverse=True)}

    ax[1].text(0.57, 0.7, 'Most used emojis', fontsize=25,
               transform=plt.gcf().transFigure)

    ax[1].text(0.58, 0.60, list(orderedEmojiCount.keys())[0], fontsize=60,
               transform=plt.gcf().transFigure, fontproperties='Segoe UI Emoji')
    ax[1].text(0.603, 0.55, '1th', fontsize=20, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax[1].text(0.68, 0.60, list(orderedEmojiCount.keys())[1], fontsize=50,
               transform=plt.gcf().transFigure, fontproperties='Segoe UI Emoji')
    ax[1].text(0.696, 0.55, '2nd', fontsize=20, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax[1].text(0.78, 0.60, list(orderedEmojiCount.keys())[2], fontsize=40,
               transform=plt.gcf().transFigure, fontproperties='Segoe UI Emoji')
    ax[1].text(0.794, 0.55, '3rd', fontsize=20, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax[1].text(0.57, 0.47, threadInfo['nbMessages'],
               fontsize=36, transform=plt.gcf().transFigure)
    ax[1].text(0.595 + len(str(threadInfo['nbMessages']))/72, 0.47,
               'messages sent', fontsize=30, transform=plt.gcf().transFigure)

    ax[1].text(0.57, 0.39, threadInfo['nbParticipants'],
               fontsize=36, transform=plt.gcf().transFigure)
    ax[1].text(0.595 + len(str(threadInfo['nbParticipants']))/72, 0.39,
               'participants', fontsize=30, transform=plt.gcf().transFigure)

    fig.suptitle("{} thread summary".format(
        threadInfo['title']), y=0.90, fontsize=30)
    return fig


def getEmotion(message, posWord, negWord):
    scallingFactorNeutral = 0.1
    message = message.split(" ")
    lenMes = len(message)
    cptDict = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for word in message:
        if word.lower() in posWord:
            cptDict['Positive'] = cptDict['Positive'] + 1
        elif word.lower() in negWord:
            cptDict['Negative'] = cptDict['Negative'] + 1
        else:
            cptDict['Neutral'] = cptDict['Neutral'] + 1

    cptDict['Neutral'] = scallingFactorNeutral*cptDict['Neutral']/(lenMes)
    return max(cptDict, key=cptDict.get)


def getDfEmotion(row, posWord=posWord, negWord=negWord):
    if row.type == 'text':
        return getEmotion(row.message, posWord, negWord)
    else:
        return None


def plotParticipantSummary(df, dfReactions, sender):
    dfSender = df[df.sender == sender]
    dfReactionsReceived = dfReactions[dfReactions.messageSender == sender]
    dfReactionsSent = dfReactions[dfReactions.sender == sender]
    figSummary, ax = plt.subplots(1, 2, figsize=(15, 10))

    # PLOT 1
    size = 0.25

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(([2*i for i in range(10)]))

    explode = [0.04, 0]

    ax[0].pie([dfSender.shape[0], df.shape[0]-dfSender.shape[0]], colors=outer_colors,
              wedgeprops=dict(width=size, edgecolor='w'), explode=explode)
    ax[0].legend([sender, "Others"])
    ax[0].set(aspect="equal", title='Participant activity')
    ax[1].set(aspect="equal", title='Participant Statistics')
    plt.axis('off')

    # PLOT 2

    orderedEmojiCount = getEmojiCount(dfSender)
    orderedEmojiCount = {k: v for k, v in sorted(
        orderedEmojiCount.items(), key=lambda item: item[1], reverse=True)}

    ax[1].text(0.57, 0.7, 'Most used emojis', fontsize=25,
               transform=plt.gcf().transFigure)

    if list(orderedEmojiCount.values())[0] != 0:
        ax[1].text(0.58, 0.60, list(orderedEmojiCount.keys())[0], fontsize=60,
                   transform=plt.gcf().transFigure, fontproperties='Segoe UI Emoji')
        ax[1].text(0.603, 0.55, '1th', fontsize=20, transform=plt.gcf(
        ).transFigure, fontproperties='Segoe UI Emoji')

        if list(orderedEmojiCount.values())[1] != 0:
            ax[1].text(0.68, 0.60, list(orderedEmojiCount.keys())[
                       1], fontsize=50, transform=plt.gcf().transFigure, fontproperties='Segoe UI Emoji')
            ax[1].text(0.696, 0.55, '2nd', fontsize=20, transform=plt.gcf(
            ).transFigure, fontproperties='Segoe UI Emoji')
            if list(orderedEmojiCount.values())[2] != 0:
                ax[1].text(0.78, 0.60, list(orderedEmojiCount.keys())[
                           2], fontsize=40, transform=plt.gcf().transFigure, fontproperties='Segoe UI Emoji')
                ax[1].text(0.794, 0.55, '3rd', fontsize=20, transform=plt.gcf(
                ).transFigure, fontproperties='Segoe UI Emoji')

    ax[1].text(0.57, 0.47, dfSender.shape[0], fontsize=36,
               transform=plt.gcf().transFigure)
    ax[1].text(0.595 + len(str(dfSender.shape[0]))/72, 0.47,
               'messages sent', fontsize=30, transform=plt.gcf().transFigure)

    ax[1].text(0.57, 0.39, dfReactionsSent.shape[0],
               fontsize=36, transform=plt.gcf().transFigure)
    ax[1].text(0.595 + len(str(dfReactionsSent.shape[0]))/72, 0.39,
               'reactions sent', fontsize=30, transform=plt.gcf().transFigure)

    ax[1].text(0.57, 0.33, dfReactionsReceived.shape[0],
               fontsize=36, transform=plt.gcf().transFigure)
    ax[1].text(0.57, 0.33,
               "".join([" " for i in range(3*len(str(dfReactionsReceived.shape[0])))]) + 'reactions received', fontsize=30, transform=plt.gcf().transFigure)
    figSummary.suptitle("{} activity summary".format(
        sender), y=0.88, fontsize=30)

    # Plot Hist

    figHist, axHist = plt.subplots(figsize=(15, 7))
    axHist.hist([dfSender['hour'], df.loc[df.sender != sender, 'hour']],
                bins=24, density=True, histtype='bar', stacked=True, range=( 0 -0.5, 23 + 0.5 ))
    axHist.set_title('stacked bar')
    for item in [figHist, axHist]:
        item.patch.set_visible(False)
    axHist.set_title("{} activity by hour of the day".format(sender))
    axHist.set_xlabel("Hour of the day")
    axHist.legend([sender, "Others"])
    axHist.set_ylabel("Frequency")

    return figSummary, figHist


def plotMostReaction(df, dfReactions, threadInfo):

    df1 = dfReactions.groupby(['messageSender', 'reaction']).count()
    df1['reaction'] = df1.index.map(lambda x: x[1])
    df1.index = df1.index.map(lambda x: x[0])
    df2 = df[['sender', 'time']].groupby('sender', as_index=False).count()
    df1['reactionRatio'] = df1.apply(
        lambda x: x.messageIndex/int(df2.loc[df2.sender == x.name, 'time']), axis=1)
    df1[df1.reaction == '👍']
    df1 = df1.drop(
        index=df2[df2.time/threadInfo['threadLifetime'] < 4].sender, errors='ignore')
    df1['messageSender'] = df1.index
    df1 = df1.reset_index()
    fun = df1[df1.reaction == '😆'].sort_values(
        by=['sender'], ascending=False, ignore_index=True)
    love = df1[df1.reaction == '❤'].sort_values(
        by=['sender'], ascending=False, ignore_index=True)
    information = df1[df1.reaction == '👍'].sort_values(
        by=['sender'], ascending=False, ignore_index=True)
    ratioFun = fun.sort_values(
        by=['reactionRatio'], ascending=False, ignore_index=True)
    ratioLove = love.sort_values(
        by=['reactionRatio'], ascending=False, ignore_index=True)
    ratioInformation = information.sort_values(
        by=['reactionRatio'], ascending=False, ignore_index=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.axis('off')

    yEmoji = 0.70
    xFun = 0.25
    xOffset = 0.21
    emojiSize = 80
    nameSize = 20
    yOffset = 0.07
    centeringOffset = 0.01
    centeringOffset2 = 0.018
    yMiniOffset = 0.038
    name2Size = 16
    ratioReducEmoji = 0.4
    yEmoji2Offset = 0.3
    nameGrossSize = 16

    ax.text(xFun, yEmoji, '❤️', fontsize=emojiSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun + xOffset, yEmoji, '👍', fontsize=emojiSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun + 2*xOffset, yEmoji, '😆', fontsize=emojiSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun-centeringOffset, yEmoji-yOffset, ratioLove.loc[0, 'messageSender'], fontsize=nameSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')
    ax.text(xFun-centeringOffset+centeringOffset2, yEmoji-yOffset - yMiniOffset, ratioLove.loc[1, 'messageSender'], fontsize=name2Size, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun-centeringOffset + xOffset, yEmoji-yOffset, ratioInformation.loc[0, 'messageSender'], fontsize=nameSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')
    ax.text(xFun-centeringOffset + xOffset+1.2*centeringOffset2, yEmoji-yOffset - yMiniOffset, ratioInformation.loc[1, 'messageSender'], fontsize=name2Size, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun + centeringOffset + 2 * xOffset, yEmoji-yOffset, ratioFun.loc[0, 'messageSender'], fontsize=nameSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')
    ax.text(xFun + centeringOffset + 2 * xOffset+0.7*centeringOffset2, yEmoji-yOffset - yMiniOffset, ratioFun.loc[1, 'messageSender'], fontsize=name2Size, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun+0.03, yEmoji-yEmoji2Offset, '❤️', fontsize=emojiSize*ratioReducEmoji, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')
    ax.text(xFun+0.016, yEmoji-yEmoji2Offset-yMiniOffset, love.loc[0, 'messageSender'], fontsize=nameGrossSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun + xOffset+0.03, yEmoji-yEmoji2Offset, '👍', fontsize=emojiSize*ratioReducEmoji, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')
    ax.text(xFun + xOffset+0.016, yEmoji-yEmoji2Offset-yMiniOffset, information.loc[0, 'messageSender'], fontsize=nameGrossSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.text(xFun + 2*xOffset+0.03, yEmoji-yEmoji2Offset, '😆', fontsize=emojiSize*ratioReducEmoji, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')
    ax.text(xFun + 2*xOffset+0.016, yEmoji-yEmoji2Offset-yMiniOffset, fun.loc[0, 'messageSender'], fontsize=nameGrossSize, transform=plt.gcf(
    ).transFigure, fontproperties='Segoe UI Emoji')

    ax.set_title(
        "Participants with the most reactions weighted by the number of messages", size=20)
    ax.text(0.23, 0.5, "Participants with the most reactions unweighted", fontSize=20)

    return fig


def getSenderInfo(df, sender):
    ''' Return miscelaneous information about the specified sender'''
    dfSender = df[df.sender == sender]

    nbDeleted = dfSender[df.type == "deleted"].shape[0]

    try:
        nbImages = dfSender[df.type == "photo"].shape[0]
        rankNbImages = df.loc[df.type == "photo", ['messageWordLength', 'sender']].groupby(
            'sender', as_index=False).count().sort_values('messageWordLength', ascending=False, ignore_index=True)
        rankNbImages = rankNbImages[rankNbImages.sender == sender].index[0]

    except IndexError:
        nbImages = dfSender[df.type == "photo"].shape[0]
        rankNbImages = -1

    try:
        meanWordLength = dfSender.loc[dfSender.type ==
                                      'text', "messageWordLength"].mean()
        rankWordLength = df.loc[df.type == "text", ['messageWordLength', 'sender']].groupby(
            'sender', as_index=False).mean().sort_values('messageWordLength', ascending=False, ignore_index=True)
        rankWordLength = rankWordLength[rankWordLength.sender ==
                                        sender].index[0]

    except IndexError:
        meanWordLength = dfSender.loc[dfSender.type ==
                                      'text', "messageWordLength"].mean()
        rankWordLength = -1

    return {'nbDeleted': nbDeleted, 'nbImages': nbImages, 'rankNbImages': rankNbImages, 'meanWordLength': np.round(meanWordLength, 1), 'rankWordLength': rankWordLength}

def getDfGrouped(df):
    '''Return the grouped messages dataframe, it is necessary to perform certain calculations, response time for example'''

    # Configuration
    selfResponseTimeThreshold = 30

    # Initialization
    dfGrouped = pd.DataFrame()
    currentSender = df.loc[0,'sender']
    startTime = df.loc[0,'time']
    endTime = df.loc[0,'time']
    isSelfResponse = False
    currentNbMessages = 0
    responseTime = 0

    # Processing
    for index,row in df.iterrows():
        if row['sender'] == currentSender and getMinutesDuration(endTime,row['time']) <= selfResponseTimeThreshold:
            #Updating Duration
            currentNbMessages += 1
            endTime = row['time']
            
        else:
            #Updating the DataFrame
            dfGrouped = dfGrouped.append({'sender':currentSender,'startTime':startTime,'endTime':endTime,'responseTime': responseTime ,'isSelfResponse':isSelfResponse,'nbMessages':currentNbMessages},ignore_index=True)
            
            #Reseting info
            responseTime = getMinutesDuration(endTime,row["time"])
            isSelfResponse = row['sender'] == currentSender
            currentSender = row['sender']
            startTime = row['time']
            endTime = row['time']
            currentNbMessages = 1
    return dfGrouped

def plotSpiderProfile(df, dfReactions,dfGrouped, threadInfo, sender):
    '''Creating a spider plot summarizing participants characteristics'''

    dfSender = df[df.sender == sender]

    # We always compare too the median because mean can be quite bad in conversation with loads of members
    #
    # IMPACT : How much your message raise reaction and create activity in the conversation ?

    receivedReactionRatio = dfReactions[dfReactions.messageSender==sender].shape[0]/dfReactions.groupby('messageSender').count()['sender'].median()

    impact = grade(receivedReactionRatio, multiScale = 1.2,offset = -1)

    # ENGAGEMENT : How often and importantly do you participate in the discussion ?

    #DEV : To implement summed fonction, we need to be able to compare the result of the sum to the other participant results, this code architecture is not suitable for that rn

    sentReactionsRatio = dfReactions[dfReactions.sender==sender].shape[0]/dfReactions.groupby('sender').count()['messageSender'].median()
    sentMessagesRatio = dfSender.shape[0]/df.shape[0]

    wordVolumeByParticipant = df.groupby('sender').sum().messageWordLength
    wordVolumeRatio = wordVolumeByParticipant[sender]/wordVolumeByParticipant.median()

    engagement = sentMessagesRatio
    engagement =  grade(engagement, multiScale = 12,offset = -1/threadInfo['nbParticipants'])
    engagement = grade(wordVolumeRatio, multiScale = 5,offset = -1)



    # REACTIVITY : How fast are you to respond ?
    responseTimeBySender = dfGrouped[dfGrouped.responseTime <= dfGrouped.responseTime.quantile(0.95)].groupby('sender').median()['responseTime']

    responseTimeRatio = (responseTimeBySender[responseTimeBySender.index==sender].to_numpy()[0])/(responseTimeBySender.median())

    reactivity = grade(responseTimeRatio, multiScale = -4,offset = -1) # answer slowly (high response time) is penalized here, so the multiscale is negative



    # ORIGINALITY : Are you the impostor ?
    # DEV : To implement
    originality = 0

    ### Graph creation ###
    df1 = pd.DataFrame(dict(
    r=[impact,reactivity,engagement],
    theta=['impact','reactivity','engagement']))

    fig = px.line_polar(df1, r='r', theta='theta', line_close=True,range_r =[0,6.1],title='Participant fingerprint',line_shape = 'spline')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_traces(fill='toself')

    return fig