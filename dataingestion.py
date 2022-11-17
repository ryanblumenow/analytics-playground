import pandas as pd
import streamlit as st

@st.cache
def readdata():

    # Read data
    df = pd.read_csv('allrecordsohe.csv', low_memory=False)
    df2 = pd.read_csv('allrecords.csv', low_memory=False)
    branddf = pd.read_csv('Brandname encoding.csv', low_memory=False)

    # Check for empty data
    df.isnull().sum()
    df2.isnull().sum()

    # Remove NaN
    nr_samples_before = df.shape[0]
    df = df.fillna(0)
    print('Removed %s samples' % (nr_samples_before - df.shape[0]))
    nr_samples_before = df2.shape[0]
    df2 = df2.fillna(0)
    print('Removed %s samples' % (nr_samples_before - df2.shape[0]))

    # Drop irrelevant variables
    df.drop(['TD_ID', 'KRUX_ID', 'TAP_IT_ID', 'GOOGLE_CLIENT_ID'], axis=1, inplace=True)
    df2.drop(['TD_ID', 'KRUX_ID', 'TAP_IT_ID', 'GOOGLE_CLIENT_ID'], axis=1, inplace=True)

    # df = df.reset_index()
    # df2 = df2.reset_index()

    return df, df2, branddf

    ### End


