# Functions for venmo classification project
import pandas as pd
import numpy as np
import psycopg2
import pymongo
import json
import datetime
import pickle


# Functions to extract data from the Mongo DB database

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


# Function to extract and store transaction specific info into the venmo db

def get_transaction_specific_information(json_list_of_transactions):
    """Function that extracts transaction specific information and
       stores each it in a table in the venmo transactions database."""
    transactions = []
    # Not including in _id because that is the object id from Venmo's db
    keys = ['note', 'type', 'date_updated', 'id', 'date_created', 'audience']
    subkeys = ['mentions', 'likes', 'comments', 'payment', 'app']
    payment_keys = ['id', 'date_created']
    app_key = ['id']
    for details in json_list_of_transactions:
        transaction = {}
        for key, val in details.items():
            if key in keys:
                transaction[key] = val
            # Subkeys 1 to 3 have the same subdictionary structure
            elif key in subkeys[:2]:
                for subkey, subval in val.items():
                    unpacked = f'{key}_{subkey}'
                    transaction[unpacked] = subval
            # From the payments subkey we only extract the payment id and the
            # date the payment was created (date_completed has null values)
            elif key in subkeys[3]:
                for subkey, subval in val.items():
                    if subkey in payment_keys:
                        unpacked = f'{key}_{subkey}'
                        transaction[unpacked] = subval
                    else:
                        pass
            # From the app subkey we only extract the id as this will be enough
            # to link to the apps_details table in the db.
            elif key in subkeys[4]:
                app_id = f'{key}_id'
                app_id_val = details[f'{key}']['id']
                transaction[app_id] = app_id_val
            else:
                continue
        transactions.append(transaction.copy())
    # Transform the list of dictionaries into independent dataframes
    transactions_df = pd.DataFrame(transactions)
    # Rename col id to transaction_id for easier recognition in the db
    transactions_df = transactions_df.rename(columns={"id": "transaction_id"})
    # Converting the date_created and date_completed objects into a
    # datetime.datetime field
    transactions_df['payment_date_created'] = pd.to_datetime(
        transactions_df['payment_date_created'], format='%Y-%m-%dT%H:%M:%S')
    # For now, drop like and mentions information
    drop = ['likes_count', 'likes_data', 'mentions_count', 'mentions_data']
    transactions_df.drop(drop, axis=1, inplace=True)
    return transactions_df


# Function to extract unique user data and store it into the venmo database

def get_payer_information(json_list_of_transactions):
    """Function that returns payer specific information from each transaction
       and adds columns relating to user settings."""
    # Identifying columns that contain null values
    null_columns = (['email', 'friend_status', 'friends_count', 'identity',
                     'phone', 'trust_request'])
    payers = []
    payer_ids = set()  # Set because I only want to retrieve unique ids
    for transaction in json_list_of_transactions:
        actor = transaction['payment']['actor']
        actor_id = actor['id']
        if actor_id in payer_ids:
            continue
        else:
            payer_ids.add(actor_id)
            payer = {}
            for key, val in transaction['payment']['actor'].items():
                if key in null_columns:
                    continue
                else:
                    payer[key] = val
            payers.append(payer.copy())

    payers_df = pd.DataFrame(payers)
    # Convert the date_joined objects into a datetime field
    payers_df['date_joined'] = pd.to_datetime(payers_df['date_joined'],
                                              format='%Y-%m-%dT%H:%M:%S')
    # Drop the first and last name columns given that the same information is
    # in the display_name column
    payers_df.drop(['first_name', 'last_name'], axis=1, inplace=True)
    # Create a column to determine if they have personalised the about column
    # Values for default come after having explored the data in eda_venmo.ipynb
    about_default = ([' ', 'No Short Bio', 'No short bio', '\n', ' \n', '  ',
                      'No Short Bio\n'])
    about_personalised = ([0 if about in about_default else 1
                           for about in payers_df['about']])
    payers_df['about_personalised'] = about_personalised
    # Create a column to determine if they have included a photo other than
    # the default photo
    pic_default = (['https://s3.amazonaws.com/venmo/no-image.gif',
                'https://s3.amazonaws.com/venmo/placeholder-image/groups-placeholder.svg'])
    pic_personalised = ([0 if about in pic_default else 1
                         for about in payers_df['about']])
    payers_df['pic_personalised'] = pic_personalised
    # Note, there is a case where a user has no about, date_joined or username.
    # They have, however, previously made a transaction so we will not drop.
    return payers_df, payer_ids


def get_payee_information(json_list_of_transactions):
    """Function that returns payee specific information from each transaction
       and adds columns relating to user settings."""
    # Identifying columns that contain null values
    null_columns = (['email', 'friend_status', 'friends_count', 'identity',
                     'phone', 'trust_request'])

    payees = []
    payee_ids = set()  # Set because I only want to retrieve unique ids
    counter = 0
    # Some transactions are deemed as unsettled because they never reach the
    # targeted payee. Hence, a try function has to be placed for now.
    for transaction in json_list_of_transactions:
        user = transaction['payment']['target']['user']
        try:
            user_id = user['id']
        except TypeError:
            counter += 1
            continue
        if user_id in payee_ids:
            continue
        else:
            payee_ids.add(user_id)
            payee = {}
            for key, val in transaction['payment']['target']['user'].items():
                if key in null_columns:
                    continue
                else:
                    payee[key] = val
            payees.append(payee.copy())
    payees_df = pd.DataFrame(payees)
    # Convert the date_joined objects into a datetime field
    payees_df['date_joined'] = pd.to_datetime(payees_df['date_joined'],
                                              format='%Y-%m-%dT%H:%M:%S')
    # Drop the first and last name columns given that the same information is
    # in display_name
    payees_df.drop(['first_name', 'last_name'], axis=1, inplace=True)
    # Create a column to determine if they have personalised the about column
    # Values for default come after having explored the data in eda_venmo.ipynb
    about_default = ([' ', 'No Short Bio', 'No short bio', '\n', ' \n', '  ',
                      'No Short Bio\n'])
    about_personalised = ([0 if about in about_default else 1
                           for about in payees_df['about']])
    payees_df['about_personalised'] = about_personalised
    # Create a column to determine if they have included a photo other than
    # the default photo
    pic_default = (['https://s3.amazonaws.com/venmo/no-image.gif',
                    'https://s3.amazonaws.com/venmo/placeholder-image/groups-placeholder.svg'])
    pic_personalised = ([0 if about in pic_default else 1
                         for about in payees_df['about']])
    payees_df['pic_personalised'] = pic_personalised
    return payees_df, payee_ids


def get_unique_user_table(json_list_of_transactions):
    """Function that returns unique user information from the combination
       of payer details and payee details."""
    # Retrieve payer and payee details
    payers_df, payer_ids = get_payer_information(json_list_of_transactions)
    payees_df, payee_ids = get_payee_information(json_list_of_transactions)
    # Identifying the payees that have not been payers for a complete user list
    payees_not_payers = set()
    for payee_id in payee_ids:
        if payee_id not in payer_ids:
            payees_not_payers.add(payee_id)
    payees_not_payers_df = payees_df.loc[payees_df['id'].apply(
        lambda x: x in payees_not_payers)]
    # Concatenate the payees that have not been payers df to the payers df to
    # generate the unique user table
    unique_users = pd.concat([payers_df, payees_not_payers_df], axis=0)
    unique_users = unique_users.rename(columns={"id": "user_id"})
    return unique_users


# Function to extract and store different apps into the venmo database

def get_app_specific_information(json_list_of_transactions):
    """Function that extracts the application through which the venmo
       transaction was made (ie iPhone app, desktop, etc) and stores
       each type in a table in the venmo transactions database."""
    apps = []
    # Only extracting app information
    app_subkeys = ['id', 'image_url', 'description', 'site_url', 'name']
    for app_details in json_list_of_transactions:
        app = {}
        for key, val in app_details['app'].items():
            app[key] = val
        apps.append(app.copy())
    apps_df = pd.DataFrame(apps)
    # Dropping duplicates because there are only 8 different ways to
    # make venmo payments
    apps_df.drop_duplicates(inplace=True)
    return apps_df


# Functions to generate the relevant user statistics

def get_keys(path):
    with open(path) as f:
        return json.load(f)


def extracting_cursor(username, password):
    "Function that connects to a database and returns the cursor"
    connection = psycopg2.connect(user=f'{username}',
                                  password=f'{password}',
                                  database='venmo_transactions')
    cursor = connection.cursor()
    return cursor


def user_info(username, password, train_window_end):
    """ Function that returns the time period since the user opened the
        account and whether or not they have a personalised bio."""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT user_id, about_personalised as personalised_bio,
            SUM(CAST('{train_window_end}' AS timestamp) - date_joined) as time_since_account_inception
            FROM users
            GROUP BY (user_id, about_personalised);"""
    cursor.execute(q)
    user_info_df = pd.DataFrame(cursor.fetchall())
    user_info_df.columns = [x[0] for x in cursor.description]
    return user_info_df


def payed_transactions(username, password, train_window_end):
    """ Function that returns the total number of transactions made during a
        given period and the mean, max of the previous transactions made."""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT DISTINCT u.user_id, MAX(p1.diff_time) as max_time_diff_made_trans,
                   AVG(p1.diff_time) as mean_time_diff_made_trans,
                   COUNT (DISTINCT p1.payment_id) as n_transactions_made,
                   COUNT (DISTINCT p1.target_user_id) as n_trans_made_to_diff_users
            FROM (SELECT p.actor_id, p.payment_id, p.target_user_id,
                         (LEAD(p.date_created, 1) OVER (PARTITION BY p.actor_id ORDER BY p.date_created)
                         - p.date_created) as diff_time
                  FROM payments p
                  WHERE p.date_created <= CAST('{train_window_end}' AS timestamp)) as p1
            INNER JOIN users u ON u.user_id = p1.actor_id
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    payed_transactions_df = pd.DataFrame(cursor.fetchall())
    payed_transactions_df.columns = [x[0] for x in cursor.description]
    return payed_transactions_df


def received_transactions(username, password, train_window_end):
    """ Function that returns the total number of transactions received during a given period and
        the mean, max of the previous transactions received."""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT DISTINCT u.user_id, MAX(p1.diff_time) as max_time_diff_received_trans,
                   AVG(p1.diff_time) as mean_time_diff_received_trans,
                   COUNT (DISTINCT p1.payment_id) as n_transactions_received,
                   COUNT (DISTINCT p1.actor_id) as trans_rec_from_diff_users
            FROM (SELECT p.target_user_id, p.payment_id, p.actor_id,
                         (LEAD(p.date_created, 1) OVER (PARTITION BY p.target_user_id ORDER BY p.date_created)
                         - p.date_created) as diff_time
                  FROM payments p
                  WHERE p.date_created <= CAST('{train_window_end}' AS timestamp)) as p1
            INNER JOIN users u ON u.user_id = p1.target_user_id
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    received_transactions_df = pd.DataFrame(cursor.fetchall())
    received_transactions_df.columns = [x[0] for x in cursor.description]
    return received_transactions_df


def transactions_made_weekdays(username, password, train_window_end):
    """Function that calculates the number of transactions made during the week
       for each user"""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_made_week
            FROM payments p
            INNER JOIN users u ON u.user_id = p.actor_id
            WHERE EXTRACT (DOW FROM p.date_created) NOT IN (0, 6)
            AND p.date_created <= CAST('{train_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    trans_made_week = pd.DataFrame(cursor.fetchall())
    trans_made_week.columns = [x[0] for x in cursor.description]
    return trans_made_week


def transactions_made_weekends(username, password, train_window_end):
    """Function that calculates the number of transactions made during the
       weekend for each user"""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_made_weeknd
            FROM payments p
            INNER JOIN users u ON u.user_id = p.actor_id
            WHERE EXTRACT (DOW FROM p.date_created) IN (0, 6)
            AND p.date_created <= CAST('{train_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    trans_made_weeknd = pd.DataFrame(cursor.fetchall())
    trans_made_weeknd.columns = [x[0] for x in cursor.description]
    return trans_made_weeknd


def transactions_made_previous_day(username, password, previous_day_start,
                                   train_window_end):
    """ Function that returns the total number of transactions made the
        previous day to our testing time frame."""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, COUNT (DISTINCT p.payment_id) as n_trans_made_yest
            FROM payments p
            INNER JOIN users u ON u.user_id = p.actor_id
            WHERE p.date_created >= CAST('{previous_day_start}' AS timestamp)
            AND p.date_created <= CAST('{train_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    trans_made_yest_df = pd.DataFrame(cursor.fetchall())
    trans_made_yest_df.columns = [x[0] for x in cursor.description]
    return trans_made_yest_df


def transactions_rec_weekdays(username, password, train_window_end):
    """Function that calculates the number of transactions received during the
       week for each user"""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_rec_week
            FROM payments p
            INNER JOIN users u ON u.user_id = p.target_user_id
            WHERE EXTRACT (DOW FROM p.date_created) NOT IN (0, 6)
            AND p.date_created <= CAST('{train_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    trans_rec_week = pd.DataFrame(cursor.fetchall())
    trans_rec_week.columns = [x[0] for x in cursor.description]
    return trans_rec_week


def transactions_rec_weekends(username, password, train_window_end):
    """Function that calculates the number of transactions received during
       the weekend for each user"""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, COUNT (DISTINCT p.payment_id) as trans_rec_weeknd
            FROM payments p
            INNER JOIN users u ON u.user_id = p.target_user_id
            WHERE EXTRACT (DOW FROM p.date_created) IN (0, 6)
            AND p.date_created <= CAST('{train_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    trans_rec_weeknd = pd.DataFrame(cursor.fetchall())
    trans_rec_weeknd.columns = [x[0] for x in cursor.description]
    return trans_rec_weeknd


def transactions_rec_previous_day(username, password, previous_day_start,
                                  train_window_end):
    """ Function that returns the total number of transactions received the
        previous day to our testing time frame."""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, COUNT (DISTINCT p.payment_id) as n_trans_rec_yest
            FROM payments p
            INNER JOIN users u ON u.user_id = p.target_user_id
            WHERE p.date_created >= CAST('{previous_day_start}' AS timestamp)
            AND p.date_created <= CAST('{train_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    trans_rec_yest_df = pd.DataFrame(cursor.fetchall())
    trans_rec_yest_df.columns = [x[0] for x in cursor.description]
    return trans_rec_yest_df


def made(username, password, previous_day_start, train_window_end):
    "Function that returns a dataframe with combined statistics for payers"
    payed_transactions_df = payed_transactions(username, password,
                                               train_window_end)
    transactions_made_previous_day_df = (
        transactions_made_previous_day(username, password, previous_day_start,
                                       train_window_end)
    )
    transactions_made_weekdays_df = (
        transactions_made_weekdays(username, password, train_window_end)
    )
    transactions_made_weekends_df = (
        transactions_made_weekends(username, password, train_window_end)
    )
    # Outer join because not everyone who has previously made a transaction
    # necessarily made one yesterday
    payed_and_previous_day = pd.merge(payed_transactions_df,
                                      transactions_made_previous_day_df,
                                      'outer', on='user_id')
    dow = pd.merge(transactions_made_weekdays_df,
                   transactions_made_weekends_df, 'outer', on='user_id')
    # Inner join because every user in either df would have made a transaction
    trans_made = pd.merge(payed_and_previous_day, dow, 'inner', on='user_id')
    # Filling with 0s the null values that arise when users have made a
    # transaction but not yesterday
    trans_made.fillna(0, inplace=True)
    return trans_made


def received(username, password, previous_day_start, train_window_end):
    "Function that returns a dataframe with combined statistics for payees"
    received_transactions_df = received_transactions(username, password,
                                                     train_window_end)
    transactions_rec_previous_day_df = (
        transactions_rec_previous_day(username, password, previous_day_start,
                                      train_window_end)
    )
    transactions_rec_weekdays_df = (
        transactions_rec_weekdays(username, password, train_window_end)
    )
    transactions_rec_weekends_df = (
        transactions_rec_weekends(username, password, train_window_end)
    )
    # Outer join because not everyone who has previously received a transaction
    # necessarily received one yesterday
    rec_and_previous_day = pd.merge(received_transactions_df,
                                    transactions_rec_previous_day_df,
                                      'outer', on='user_id')
    dow = pd.merge(transactions_rec_weekdays_df, transactions_rec_weekends_df,
                   'outer', on='user_id')
    # Inner join because every user in either df would have received a transaction
    trans_rec = pd.merge(rec_and_previous_day, dow, 'inner', on='user_id')
    # Filling with 0s the null values that arise when users have received a
    # transaction but not yesterday
    trans_rec.fillna(0, inplace=True)
    return trans_rec


def transactions(username, password, previous_day_start, train_window_end):
    "Function that returns a dataframe with combined statistics for payees"
    made_df = made(username, password, previous_day_start, train_window_end)
    received_df = received(username, password, previous_day_start,
                           train_window_end)
    # Outer join because not everyone who has made a transaction necessarily
    # received one and viceversa
    trans = pd.merge(made_df, received_df, 'outer', on='user_id')
    # Filling with 0s the null values that arise when users have made a
    # transaction but not received one
    trans.fillna(0, inplace=True)
    return trans


def get_aggregated_user_statistics(username, password, previous_day_start,
                                   train_window_end):
    """Function that returns a dataframe with aggregated user statistics and
    personal user information statistics"""
    user_df = user_info(username, password, train_window_end)
    trans_df = transactions(username, password, previous_day_start,
                            train_window_end)
    # Inner join because all users should have either made or received a
    # transaction, so they will have a user_id
    agg_table = pd.merge(user_df, trans_df, 'inner', on='user_id')
    time_delta_cols = (['time_since_account_inception',
                        'max_time_diff_made_trans',
                        'max_time_diff_received_trans',
                        'mean_time_diff_made_trans',
                        'mean_time_diff_received_trans'])
    for col in time_delta_cols:
        agg_table[f'{col}'] = [diff.total_seconds() for diff in agg_table[f'{col}']]
    return agg_table


def extract_target(username, password, test_window_start, test_window_end):
    """Function that returns the target variable (whether someone made a
       transaction during a given time period) or not."""
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, COUNT (DISTINCT p.payment_id) as n_transactions_made_29th
            FROM payments p
            INNER JOIN users u ON u.user_id = p.actor_id
            WHERE p.date_created >= CAST('{test_window_start}' AS timestamp)
            AND p.date_created <= CAST('{test_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    tran_or_not_df = pd.DataFrame(cursor.fetchall())
    tran_or_not_df.columns = [x[0] for x in cursor.description]
    tran_or_not_df['n_transactions_made_29th'] = (
        [1 for trans in tran_or_not_df['n_transactions_made_29th']]
    )
    return tran_or_not_df
