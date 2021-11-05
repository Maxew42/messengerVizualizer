import os
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.messenger_data_processing import *

### Messenger Statistics Viewer Web page Code ###
#
#   File name: st_app.py
#   Author: Maximilien Dufau
#   Date created: XX/09/2021 (DD/MM/YYYY)
#   Python Version: 3.1
#
###
st.markdown("# Messenger Statistics Viewer")
st.markdown('##### *by [Maximilien Dufau](https://www.linkedin.com/in/maximilien-dufau/), source code available [here](https://github.com/Maxew42/messengerVizualizer) !*')

st.markdown(
    "### Discover the beautiful secrets hidden in your daily conversation statistics")
st.markdown("---")


# Initializing state dictionnary
if 'loaded' not in st.session_state.keys():
    st.session_state['loaded'] = False

plt.style.use('default')

#DEV : should be delete and replaced by file upload before sending this in production
# def file_selector(folder_path='../cleanData/'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)
#filename = file_selector()

uploaded_files = st.file_uploader("Select your json messages files",type = "json", accept_multiple_files=True,help='Refer to the provided tutorial if needed')
st.markdown("Don't know what to upload ? Check out this [tutorial](https://github.com/Maxew42/messengerVizualizer/blob/main/README.md)")
#st.write(len(uploaded_files))
dataDict = {}

for uploaded_file in uploaded_files:
    # Massively inspired by the work of Luksan on StackOverflow (https://stackoverflow.com/users/1889274/luksan)
    fix_mojibake_escapes = partial(   
        re.compile(rb'\\u00([\da-f]{2})').sub,
        lambda m: bytes.fromhex(m.group(1).decode()))
    repaired = fix_mojibake_escapes(uploaded_file.read())
    dataDict[uploaded_file.name] = json.loads(repaired.decode('utf8'))

loaded = False


@st.cache
def loadFacebookDfCached(filename):
    return loadFacebookDf(filename)


loadButton = st.button("Load file")
if loadButton or st.session_state['loaded']:

    if not st.session_state['loaded']:
        df, threadInfo, dfReactions = loadFacebookDf(dataDict)
        #df, threadInfo, dfReactions = loadFacebookDf(filename)
        dfGrouped = getDfGrouped(df)
        st.session_state['df'] = df
        st.session_state['threadInfo'] = threadInfo
        st.session_state['dfReactions'] = dfReactions
        st.session_state['dfGrouped'] = dfGrouped
        st.session_state['loaded'] = True
        st.session_state['fig1'] = False
        st.session_state['participantDict'] = dict({})
        st.balloons()
    else:
        df = st.session_state['df']
        threadInfo = st.session_state['threadInfo']
        dfReactions = st.session_state['dfReactions']
        dfGrouped = st.session_state['dfGrouped']


    ## Sidebar initilization and fancy things
    st.sidebar.markdown(
        "[I. General thread information](#general-thread-information)")
    st.sidebar.markdown(
        "> [1 - Activity distribution over time](#activity-distribution-over-time)")
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "[II. Statistics breakdown by participants](#statistics-breakdown-by-participants)")
    st.sidebar.markdown(
        "> [1 - Emojis usage distribution across participants](#emojis-usage-distribution-across-participants)")
    st.sidebar.markdown(
        "> [2 - Participants with the most reactions received](#participants-with-the-most-reactions-received)")
    st.sidebar.markdown(
        "> [3 - Messages Types distribution](#messages-types-distribution)")
    st.sidebar.markdown(
        "> [4 - Thread WordCloud](#thread-wordcloud)")
    st.sidebar.markdown(
        "> [4 - Thread WordCloud](#thread-wordcloud)")
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "[III. Participant Summary](#participant-summary)")
    ###

    st.markdown("---")
    st.pyplot(plotGeneralInfo(df, threadInfo))
    st.markdown("---")

    st.markdown("## General thread information")
    st.write("Thread lifetime:       {} weeks".format(
        np.round(threadInfo['threadLifetime'], 2)))
    st.write("Number of images sent: {}".format(
        df[df.type == 'photo'].shape[0]))
    st.write("Number of reactions : {}".format(threadInfo['nbReactions']))

    st.markdown("### Activity distribution over time")
    fig = plotUsageOverTime(df)
    st.pyplot(fig=fig)
    st.markdown("---")

    st.markdown("## Statistics breakdown by participants")
    st.markdown("### Emojis usage distribution across participants")
    fig = plotEmojiUsageDistribution(df)
    st.pyplot(fig)

    st.markdown("### Participants with the most reactions received")
    fig = plotMostReaction(df, dfReactions, threadInfo)
    st.pyplot(fig)

    st.markdown("### Messages Types distribution")
    fig = plotMessagesType(df)
    st.pyplot(fig)

    st.markdown("### Thread WordCloud")

    #if 'fig1' in st.session_state.keys():
    if st.session_state['fig1'] == False:
        fig = plotWordCloud(filterText(df))
        st.session_state['fig1'] = fig
    else:
        fig = st.session_state['fig1']
    st.pyplot(fig)
    st.markdown("---")

    st.markdown("### Participant Summary")
    option = st.selectbox('Select a participant', np.unique(
        df.sender))
    if option in st.session_state['participantDict'].keys():
        fig1, fig2 = st.session_state['participantDict'][option]

    else:
        fig1, fig2 = plotParticipantSummary(df, dfReactions, option)
        st.session_state['participantDict'][option] = (fig1, fig2)
    st.pyplot(fig1)
    st.pyplot(fig2)

    fig = plotSpiderProfile(df,dfReactions,dfGrouped,threadInfo,option)
    st.plotly_chart(fig)

    senderInfo = getSenderInfo(df,option)
    st.write("Number of unsent messages : {}".format(senderInfo['nbDeleted']))
    st.write("Number of images sent : {}   ( #{} )".format(senderInfo['nbImages'],senderInfo['rankNbImages']+1))
    st.write("Mean messages word length : {}   ( #{} )".format(senderInfo['meanWordLength'],senderInfo['rankWordLength']+1))