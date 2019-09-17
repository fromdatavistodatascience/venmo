# Functions for venmo classification project
import pandas as pd
import numpy as np
import psycopg2
import pymongo
import json
import datetime
import pickle


def collection():
    "Function that returns a collection from a MongoDB"
    # Instantiate a MongoClient and inspect the database names
    mc = pymongo.MongoClient()
    # Create a db from the test database in MongoClient
    mydb = mc['test']
    # Accessing the venmo collection in the test database
    venmo = mydb['venmo']
    return venmo


def initial_5pct(collection):
    """Function that returns a list of dictionaries with the initial 5% of
    transactions
    :input param - collection: MongoDB collection containing all transactions
    :ouput param - initial_10pct: returns initial 5% of transactions as a
    list of dictionaries
    """
    _5pct = round(0.05 * (collection.count()))
    cur = collection.find({})[:_5pct]
    transactions = [transaction for transaction in cur]
    with open('initial_5pct_transactions.pkl', 'wb') as f:
        pickle.dump(transactions, f)
