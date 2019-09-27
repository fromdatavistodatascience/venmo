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


def initial_25pct(collection):
    """Function that returns a list of dictionaries with the initial 5% of
    transactions
    :input param - collection: MongoDB collection containing all transactions
    :ouput param - initial_10pct: returns initial 5% of transactions as a
    list of dictionaries
    """
    _25pct = round(0.25 * (collection.count()))
    cur = collection.find({})[:_25pct]
    transactions = [transaction for transaction in cur]
    #  with open('initial_5pct_transactions.pkl', 'wb') as f:
    #    pickle.dump(transactions, f)
    return transactions


# Function to extract and store transaction specific info into the venmo db

def get_transaction_specific_information(json_list_of_transactions):
    """Function that extracts transaction specific information and
       stores each it in a table in the venmo transactions database."""
    transactions = []
    # Not including in _id because that is the object id from Venmo's db
    keys = ['note', 'type', 'date_updated', 'id', 'date_created', 'audience']
    subkeys = ['mentions', 'likes', 'comments']
    for details in json_list_of_transactions:
        transaction = {}
        for key in keys:
            transaction[key] = details.get(key)
        for key in subkeys:
            transaction[f'{key}_count'] = details[key].get('count')
            # Count determines if users interacted with transactions
            if transaction[f'{key}_count'] > 0:
                details[key].get('data')
                transaction[f'{key}_data'] = []
                # Getting the ids of users than interacted with transactions
                for inter in details[f'{key}']['data']:
                    try:
                        transaction[f'{key}_data'].append(inter['user']['id'])
                    except:
                        transaction[f'{key}_data'].append(inter['id'])
            else:
                transaction[f'{key}_data'] = None
        transaction['payment_id'] = details['payment'].get('id')
        transaction['payment_actor_id'] = details['payment']['actor'].get('id')
        # Rename col id to transaction_id for easier recognition in the db
        transaction['transaction_id'] = transaction.pop('id')
        transactions.append(transaction.copy())
    return transactions


# Function to extract payment data and store it into the venmo database

def get_payment_info(json_list_of_transactions):
    """Function that extracts payment specific information and identifies
       whether payers have made settled, unsettled or both types of payments."""
    payments = []
    # Keys in the payment dictionary that have the same structure
    keys = (['note', 'action', 'status', 'date_created', 'id',
             'audience', 'date_completed'])
    settled_payer_id = set()  # Set of actor_ids that have settled payments
    unsettled_payer_id = set()  # Set of actor_ids that have unsettled payments
    for transaction in json_list_of_transactions:
        payment = {}
        payment_details = transaction['payment']
        for key in keys:
            payment[key] = payment_details.get(key)
        payment['target_type'] = payment_details['target'].get('type')
        try:
            payment['target_user_id'] = payment_details['target']['user']['id']
            settled_payer_id.add(transaction['payment']['actor']['id'])
        except TypeError:
            # Identify payers who have pending or cancelled transactions
            unsettled_payer_id.add(transaction['payment']['actor']['id'])
        payment['actor_id'] = payment_details['actor'].get('id')
        # Rename col id to payment_id for easier recognition in the db
        payment['payment_id'] = payment.pop('id')
        # Transforming the date created col into datetime object
        payment['date_created'] = datetime.datetime.strptime(
            payment['date_created'], '%Y-%m-%dT%H:%M:%S')
        payments.append(payment.copy())
    settled_and_unsettled_payer_ids = settled_payer_id.intersection(
        unsettled_payer_id)
    unsettled_payer_ids = unsettled_payer_id - settled_payer_id
    return payments


def get_true_and_false_transactions_from_settled_transactions(json_list_of_transactions):
    """Function that returns a set of successful and duplicated payment ids.
       Payments are deemed as duplicates if a successful payments has happened
       within 10 minutes before or after the unsuccessful transaction occured."""
    payments_df, settled_and_unsettled_payer_ids, unsettled_payer_ids = get_payment_info(json_list_of_transactions)
    duplicated_transaction_ids = set()
    non_duplicated_transaction_ids = set()
    for actor in settled_and_unsettled_payer_ids:
        #Creating actor specific dataframes
        settled_and_unsettled_trans_df = payments_df.loc[payments_df['actor_id'] == f'{actor}']
        transaction_dates = [date for date in settled_and_unsettled_trans_df['date_created']]
        #Separating the dates of created payments for each user
        for i in range(len(transaction_dates)-1):
            time_diff = transaction_dates[i+1] - transaction_dates[i]
            time_diff = time_diff.total_seconds()
            #If the payments are made within 10 minutes then identify those transactions
            if time_diff < 600: #WHY 10 MINUTES THOUGH?
                date_tuple = (transaction_dates[i], transaction_dates[i+1])
                #Create a new dataframe for each user that contains transactions made within 10 minute of each other
                transaction_within_10 = (
                    settled_and_unsettled_trans_df.loc[settled_and_unsettled_trans_df['date_created'].isin(date_tuple)])
                #Extract the status' of both transactions
                for status in transaction_within_10['status']:
                #If one of the status' is settled it means that the rest are duplicates
                    if status != 'settled':
                        duplicated_id = transaction_within_10.loc[transaction_within_10['status'] == status]['payment_id']
                        for _id in duplicated_id:
                            duplicated_transaction_ids.add(_id)
            else:
                date_tuple = (transaction_dates[i], transaction_dates[i+1])
                #Create a new dataframe for each user that contains transactions made within 10 minute of each other
                transaction_within_10 = (
                    settled_and_unsettled_trans_df.loc[settled_and_unsettled_trans_df['date_created'].isin(date_tuple)])
                #Extract the status' of both transactions
                for status in transaction_within_10['status']:
                #If one of the status' is settled it means that the rest are duplicates
                    if status != 'settled':
                        non_duplicated_id = transaction_within_10.loc[transaction_within_10['status'] == status]['payment_id']
                        for _id in non_duplicated_id:
                            non_duplicated_transaction_ids.add(_id)
    return duplicated_transaction_ids, non_duplicated_transaction_ids


def get_true_and_false_transactions_from_unsettled_transactions(json_list_of_transactions):
    """Function that returns a set of duplicated payment ids from unsettled transactions. 
       Payments are deemed as duplicates if another unsuccessfull payment has happened 
       10 minutes before the unsuccessful transaction occured."""
    payments_df, settled_and_unsettled_payer_ids, unsettled_payer_ids = get_payment_info(json_list_of_transactions) 
    # Select the transactions which users with unsettled payments have made within 10 minutes of each other.
    duplicated_unsettled_transaction_ids = set()
    non_duplicated_unsettled_transaction_ids = set()
    for actor in unsettled_payer_ids:
        #Creating actor specific dataframes
        unsettled_trans_df = payments_df.loc[payments_df['actor_id'] == f'{actor}']
        #Separating the dates of created payments for each user
        transaction_dates = [date for date in unsettled_trans_df['date_created']]
        if len(transaction_dates) == 1:
            tran_id = (
                unsettled_trans_df.loc[unsettled_trans_df['date_created'] == transaction_dates[0]]['payment_id'])
            non_duplicated_unsettled_transaction_ids.add(tran_id.any())
        else:
            first_trans_date = None
            for i in range(len(transaction_dates)-1):
                time_diff = transaction_dates[i+1] - transaction_dates[i]
                time_diff = time_diff.total_seconds()
                #If the payments are made within 10 minutes then identify those transactions
                if time_diff < 600: #WHY 10 MINUTES THOUGH?
                    date_tuple = (transaction_dates[i], transaction_dates[i+1])
                    trans_ids_for_date_tuple = (
                        unsettled_trans_df.loc[unsettled_trans_df['date_created'] == transaction_dates[i]]['payment_id'])
                    if trans_ids_for_date_tuple.all() in duplicated_unsettled_transaction_ids:
                        duplicated_trans_id = (
                            unsettled_trans_df.loc[unsettled_trans_df['date_created'] == transaction_dates[i+1]]['payment_id'])
                        duplicated_unsettled_transaction_ids.add(duplicated_trans_id.any())
                    else:
                        first_trans_id = (
                            unsettled_trans_df.loc[unsettled_trans_df['date_created'] == transaction_dates[i]]['payment_id'])
                        non_duplicated_unsettled_transaction_ids.add(first_trans_id.any())
                        duplicated_trans_id = (
                            unsettled_trans_df.loc[unsettled_trans_df['date_created'] == transaction_dates[i+1]]['payment_id'])
                        duplicated_unsettled_transaction_ids.add(duplicated_trans_id.any())
                else:
                    if transaction_dates[i+1] == transaction_dates[-1]:
                        date_tuple = (transaction_dates[i], transaction_dates[i+1])
                        non_duplicated_transaction_id = (
                            unsettled_trans_df.loc[unsettled_trans_df['date_created'].isin(date_tuple)]['payment_id'])
                        for _id in non_duplicated_transaction_id:
                            non_duplicated_unsettled_transaction_ids.add(_id)
                    else:
                        non_duplicated_transaction_id = (
                                unsettled_trans_df.loc[unsettled_trans_df['date_created'] == transaction_dates[i]]['payment_id'])
                        non_duplicated_unsettled_transaction_ids.add(non_duplicated_transaction_id.any())
    return duplicated_unsettled_transaction_ids, non_duplicated_unsettled_transaction_ids


def diff_between_true_and_false_payments(json_list_of_transactions):
    "Function that adds columns to differentiate between true and false payments"
    duplicated_transaction_ids, non_duplicated_transaction_ids = (
        get_true_and_false_transactions_from_settled_transactions(json_list_of_transactions)
    )
    duplicated_unsettled_transaction_ids, non_duplicated_unsettled_transaction_ids = (
        get_true_and_false_transactions_from_unsettled_transactions(json_list_of_transactions))
    payments_df, settled_and_unsettled_payer_ids, unsettled_payer_ids = (
        get_payment_info(json_list_of_transactions))
    settled_payment_ids = set(payments_df.loc[payments_df['status'] == 'settled']['payment_id'])
    #Create new columns to identify between the two types of transactions
    payments_df['true_transactions'] = ([1 if _id in settled_payment_ids else 1
                                         if _id in non_duplicated_transaction_ids else 1
                                         if _id in non_duplicated_unsettled_transaction_ids 
                                         else 0 for _id in payments_df['payment_id']])
    payments_df['false_transactions'] = ([1 if _id in duplicated_transaction_ids else 1
                                          if _id in duplicated_unsettled_transaction_ids
                                          else 0 for _id in payments_df['payment_id']])
    return payments_df


def get_payments_df_with_differentiated_payments(json_list_of_transactions):
    """Function that perform final manipulation on the payments df prior to dumping
       the data in the venmo database"""
    payments_df = diff_between_true_and_false_payments(json_list_of_transactions)
    # Unpack the merchant split type into two diff cols
    payments_df = payments_df.drop('merchant_split_purchase', 1).assign(**payments_df['merchant_split_purchase']
                                                                    .dropna().apply(pd.Series))
    # Rename to miror the json structure
    payments_df = payments_df.rename(columns={"authorization_id": "merchant_authorization_id"})
    # Same process with the target_redeemable_target_col
    payments_df = payments_df.drop('target_redeemable_target', 1).assign(**payments_df['target_redeemable_target']
                                                                     .dropna().apply(pd.Series))
    # Rename to miror the json structure
    payments_df = payments_df.rename(columns = {"display_name": "target_redeemable_target_display_name",
                                                "type": "target_redeemable_target_type"})
    return payments_df


# Function to extract unique user data and store it into the venmo database

def get_payer_information(json_list_of_transactions):
    """Function that returns payer specific information from each transaction
       and adds columns relating to user settings."""
    # Identifying columns that don't contain values.
    # Not adding first or last name since they are present in display name
    keys = (["username", "is_active", "display_name", "is_blocked", "about",
             "profile_picture_url", "id", "date_joined", "is_group" ])
    # Values for default come after having explored the data in eda_venmo.ipynb
    about_default = ([' ', 'No Short Bio', 'No short bio', '\n', ' \n', '  ',
                      'No Short Bio\n'])
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
            for key in keys:
                # Determine if their about col is personalised
                if key == 'about':
                    about = actor.get(key)
                    payer[key] = actor.get(key)
                    if about in about_default:
                        # Col to show if personalised about or not
                        payer['about_personalised'] = 0
                    else:
                        payer['about_personalised'] = 1
                else:
                    payer[key] = actor.get(key)
            payer['user_id'] = payer.pop('id')
            payers.append(payer.copy())
    # Note, there is a case where a user has no about, date_joined or username.
    # They have, however, previously made a transaction so we will not drop.
    return payers, payer_ids


def get_payee_information(json_list_of_transactions):
    """Function that returns payee specific information from each transaction
       and adds columns relating to user settings."""
    # Identifying columns that contain null values
    keys = (["username", "is_active", "display_name", "is_blocked", "about",
             "profile_picture_url", "id", "date_joined", "is_group" ])
    # Values for default come after having explored the data in eda_venmo.ipynb
    about_default = ([' ', 'No Short Bio', 'No short bio', '\n', ' \n', '  ',
                      'No Short Bio\n'])
    payees = []
    payee_ids = set()  # Set because I only want to retrieve unique ids
    # Some transactions are deemed as unsettled because they never reach the
    # targeted payee. Hence, a try function has to be placed for now.
    for transaction in json_list_of_transactions:
        user = transaction['payment']['target']['user']
        try:
            user_id = user['id']
        except TypeError:
            continue
        if user_id in payee_ids:
            continue
        else:
            payee_ids.add(user_id)
            payee = {}
            for key in keys:
                # Determine if their about col is personalised
                if key == 'about':
                    about = user.get(key)
                    payee[key] = user.get(key)
                    if about in about_default:
                        # Col to show if personalised about or not
                        payee['about_personalised'] = 0
                    else:
                        payee['about_personalised'] = 1
                else:
                    payee[key] = user.get(key)
            payee['user_id'] = payee.pop('id')
            payees.append(payee.copy())
    return payees, payee_ids


def get_unique_user_table(json_list_of_transactions):
    """Function that returns unique user information from the combination
       of payer details and payee details."""
    # Retrieve payer and payee details
    payers, payer_ids = get_payer_information(json_list_of_transactions)
    payees, payee_ids = get_payee_information(json_list_of_transactions)
    # Identifying the payees that have not been payers for a complete user list
    payee_ids.difference_update(payer_ids)
    clean_payees = [payee for payee in payees if payee['user_id'] in payee_ids]
    # Concatenate the payees that have not been payers to the payers to
    # generate the unique user table
    payers.extend(clean_payees)
    return payers


# Function to extract and store different apps into the venmo database

def get_app_specific_information(json_list_of_transactions):
    """Function that extracts the application through which the venmo
       transaction was made (ie iPhone app, desktop, etc) and stores
       each type in a table in the venmo transactions database."""
    apps = []
    # Only extracting app information
    app_subkeys = ['id', 'image_url', 'description', 'site_url', 'name']
    app_ids = set()
    for app_detail in json_list_of_transactions:
        app_details = app_detail['app']
        app_id = app_details['id']
        # There are only 8 diff types of apps, so by checking the id
        # the process becomes much less computationally expensive
        if app_id in app_ids:
            continue
        else:
            app_ids.add(app_id)
            app = {}
            for key in app_details:
                app[key] = app_details.get(key)
            apps.append(app.copy())
    return apps


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
            SUM(CAST('{train_window_end}' AS timestamp) -
            CAST(date_joined AS timestamp)) as time_since_account_inception
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
