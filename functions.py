# Functions for venmo classification project
import pandas as pd
import numpy as np
import psycopg2
import pymongo
import json
import datetime
import pickle
import requests
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import emoji
import regex
import nltk
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
import string
from emoji.unicode_codes import UNICODE_EMOJI as ue
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.cluster import cosine_distance
from nltk.cluster.kmeans import KMeansClusterer
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora

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
    weird_transactions= []
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
        try:
            transaction['payment_id'] = details['payment'].get('id')
            transaction['payment_actor_id'] = details['payment']['actor'].get('id')
        except:
            weird_transactions.append(transaction.copy())
        # Rename col id to transaction_id for easier recognition in the db
        transaction['transaction_id'] = transaction.pop('id')
        transactions.append(transaction.copy())
    return transactions, weird_transactions


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
        if transaction['id'] == '2541220786958500195':
            continue
        else:
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
        #settled_and_unsettled_payer_ids = settled_payer_id.intersection(
        #    unsettled_payer_id)
        #unsettled_payer_ids = unsettled_payer_id - settled_payer_id
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
        if transaction['id'] == '2541220786958500195':
            continue
        else:
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
        if transaction['id'] == '2541220786958500195':
            continue
        else:
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


# Functions to vectorize text for each user

def extract_user_notes(username, password, train_window_end):
    cursor = extracting_cursor(username, password)
    q = f"""SELECT u.user_id, t.note
            FROM transactions t
            JOIN payments p USING (payment_id)
            JOIN users u on u.user_id = p.actor_id
            WHERE p.date_created <= CAST('{train_window_end}' AS timestamp)
            ORDER BY u.user_id ASC;"""
    cursor.execute(q)
    user_notes = pd.DataFrame(cursor.fetchall())
    user_notes.columns = [x[0] for x in cursor.description]
    return user_notes


def get_notes_into_unicode(notes):
    """Function that takes in all the notes and returns the text as well as
    the ejomis used in unicode."""
    emoji_dict = {}
    recomposed_note = []
    for note in notes:
        note_text = []
        data = regex.findall(r'\X', note)
        for word in data:
            if any(char in emoji.UNICODE_EMOJI for char in word):
                unicode_emoji = word.encode('unicode-escape').decode('ASCII')
                emoji_dict[word] = unicode_emoji.lower()
                note_text.append(unicode_emoji+' ')
            else:
                note_text.append(word)
        recomposed_note.append(''.join(note_text))
    return recomposed_note, emoji_dict


def get_clean_text_pattern(recomposed_note):
    """Function that filters through the notes, retrieves those that match
     the specified pattern and removes stopwords."""
    pattern = "([a-zA-Z0-9\\\]+(?:'[a-z]+)?)"
    recomposed_note_raw = []
    recomposed_note_raw = (
        [nltk.regexp_tokenize(note, pattern) for note in recomposed_note])
    # Create a list of stopwords and remove them from our corpus
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    # additional slang and informal versions of the original words had to be added to the corpus.
    stopwords_list += (["im", "ur", "u", "'s", "n", "z", "n't", "brewskies", "mcd’s", "Ty$",
                        "Diploooooo", "thx", "Clothessss", "K2", "B", "Comida", "yo", "jobby",
                        "F", "jus", "bc", "queso", "fil", "Lol", "EZ", "RF", "기프트카드", "감사합니다",
                        "Bts", "youuuu", "X’s", "bday", "WF", "Fooooood", "Yeeeeehaw", "temp",
                        "af", "Chipoodle", "Hhuhhyhy", "Yummmmers", "MGE", "O", "Coook", "wahoooo",
                        "Cuz", "y", "Cutz", "Lax", "LisBnB", "vamanos", "vroom", "Para", "el", "8==",
                        "bitchhh", "¯\\_(ツ)_/¯", "Ily", "CURRYYYYYYY", "Depósito", "Yup", "Shhhhh"])

    recomposed_note_stopped = (
        [[w.lower() for w in note if w not in stopwords_list] for note in recomposed_note_raw])
    return recomposed_note_stopped


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_notes(recomposed_note_stopped):
    "Function that lemmatizes the different notes."
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_notes = []
    for sentence in recomposed_note_stopped:
        # Notes have unicode to represent emojis and those can't be lemmatized
        try:
            for word in nltk.word_tokenize(sentence):
                # Notes that combine emojis and text
                try:
                    lem = lemmatizer.lemmatize(word, get_wordnet_pos(word))
                    lemmatized_notes.append(lem)
                except:
                    lemmatized_notes.append(word)
        except:
            lemmatized_notes.append(sentence)
    return lemmatized_notes


def turn_emoji_unicode_to_text(lemmatized_notes, emoji_dict):
    "Function that converts unicode into emojis."
    recomposed_note_stopped_em = []
    for note in lemmatized_notes:
        note_list = []
        for word in note:
            if word.startswith('\\'):
                # Emoji dict represents a dict matching emojis and unicode.
                for key, val in emoji_dict.items():
                    if word == val:
                        note_list.append(key)
            else:
                 note_list.append(word)
        recomposed_note_stopped_em.append(note_list)
    return recomposed_note_stopped_em


def emojis_to_text(notes_list):
    """Function that takes in all the notes and returns the emojis used
    in the form of text captured by :colons:"""
    recomposed_note = []
    for notes in notes_list:
        note_list = []
        for note in notes:
            note_text = []
            data = regex.findall(r'\X', note)
            for word in data:
                if any(char in emoji.UNICODE_EMOJI for char in word):
                    note_text.append(emoji.demojize(f'{word}'))
                else:
                    note_text.append(word)
            note_list.append(''.join(note_text))
        recomposed_note.append(note_list)
    return recomposed_note


def train_doc2vec_vectorizer(fully_recomposed_notes, whole_corpus_notes):
    "Function that returns a trained doc2vec model for the whole note corpus."
    # In order to train the Doc2Vec model all the words need to be in the same list
    tagged_data = [TaggedDocument(words=w.split(' '), tags=[str(i)]) for i, w in enumerate(whole_corpus_notes)]
    # Select model hyperparameters
    max_epochs = 10
    vec_size = 20
    alpha = 0.025
    min_alpha=0.00025
    min_count=1
    dm =1
    # Input hyparameters into the model
    vectorizer = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=min_alpha,
                     min_count=min_count, dm =dm)
    # Build vocab of the notes with tagged data
    vectorizer.build_vocab(tagged_data)
    # Train the model for the range of epochs specified
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        vectorizer.train(tagged_data,
                    total_examples=vectorizer.corpus_count,
                    epochs=vectorizer.iter)
        # decrease the learning rate
        vectorizer.alpha -= 0.0002
        # fix the learning rate, no decay
        vectorizer.min_alpha = vectorizer.alpha
    # Save the model
    vectorizer.save("d2v.model")


def get_aggregated_user_note_vector(username, password, train_window_end):
    "Function that turns the notes for each transaction into an n dimmensional vector."
    
    # Load functions to generate a list of fully composed notes
    user_notes = extract_user_notes(username, password, train_window_end)
    notes = user_notes['note']
    recomposed_note, emoji_dict = get_notes_into_unicode(notes)
    recomposed_note_stopped = get_clean_text_pattern(recomposed_note)
    lemmatized_notes = lemmatize_notes(recomposed_note_stopped)
    recomposed_note_stopped_em = turn_emoji_unicode_to_text(lemmatized_notes, emoji_dict)
    fully_recomposed_notes = emojis_to_text(recomposed_note_stopped_em)
    
    # Combine the notes into a single corpus
    whole_corpus_notes = [' '.join(note) for note in fully_recomposed_notes]
    
    # Load the model
    vectorizer= Doc2Vec.load("d2v.model")
    
    # Find the vectors for each note in the whole note corpus
    _vectrs = []
    for note in whole_corpus_notes:
        v = np.array(vectorizer.infer_vector(note))
        _vectrs.append(v)
    _vectrs_df = pd.DataFrame(_vectrs)
    
    # Each payment note vectorized for each user
    user_notes.drop('note', axis=1, inplace=True)
    user_vectrs_df = pd.concat([user_notes, _vectrs_df], axis=1)
    
    # Calculate the mean for users with multiple transactions (multiple notes)
    user_vectrs_df = user_vectrs_df.groupby('user_id').mean()
    
    return user_vectrs_df


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
    q = f"""SELECT u.user_id, u.about_personalised as personalised_bio,
            SUM(CAST('{train_window_end}' AS timestamp) -
            CAST(u.date_joined AS timestamp)) as time_since_account_inception,
            COUNT(CASE WHEN p.status = 'settled'  THEN 1 END) as settled,
            COUNT(CASE WHEN p.status = 'pending'  THEN 1 END) as pending,
            COUNT(CASE WHEN p.status = 'cancelled'  THEN 1 END) as cancelled
            FROM users u
            INNER JOIN payments p ON p.actor_id=u.user_id
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
    user_vectrs_df = get_aggregated_user_note_vector(username, password, 
                                                     train_window_end)
    combined_user_info = pd.merge(user_df, user_vectrs_df, 'inner', on='user_id')
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
    q = f"""SELECT u.user_id,
            COUNT (DISTINCT p.payment_id) as n_trans_made_in_measured_period
            FROM payments p
            INNER JOIN users u ON u.user_id = p.actor_id
            WHERE p.date_created >= CAST('{test_window_start}' AS timestamp)
            AND p.date_created <= CAST('{test_window_end}' AS timestamp)
            GROUP BY (u.user_id);"""
    cursor.execute(q)
    tran_or_not_df = pd.DataFrame(cursor.fetchall())
    tran_or_not_df.columns = [x[0] for x in cursor.description]
    tran_or_not_df['n_trans_made_in_measured_period'] = (
        [1 for trans in tran_or_not_df['n_trans_made_in_measured_period']]
    )
    return tran_or_not_df


# Formulas for currency analysis

def get_fx_rates(api_key, exchange_currency, desired_currency):
    """Function that returns the 100 day FX rate history for the currency wished to
    be exchanged in json format."""
    url = 'https://www.alphavantage.co/query?'
    function_input = 'FX_DAILY'
    from_symbol_input = f'{exchange_currency}'
    to_symbol_input = f'{desired_currency}'
    url_params = (f"""function={function_input}&from_symbol={from_symbol_input}&to_symbol={to_symbol_input}&apikey={api_key}""")
    request_url = url + url_params
    response = requests.get(request_url)
    return response


def get_adjusted_rate(response_json):
    "Function that converts json into pd dataframe with historic adj closed prices."
    response_dict = {}
    for key, val in response.json()['Time Series FX (Daily)'].items():
        response_dict[key] = float(val['4. close'])
    response_df = pd.DataFrame.from_dict(response_dict, 'index')
    response_df.columns = ['Adj Close Price']
    response_df = response_df.reindex(index=response_df.index[::-1])
    return response_df


def get_bollinger_bands(response_df):
    """Function that returns the bollinger bands for the exchange rate in question."""
    response_df['30 Day MA'] = response_df['Adj Close Price'].rolling(window=20).mean()
    response_df['30 Day STD'] = response_df['Adj Close Price'].rolling(window=20).std()
    response_df['Upper Band'] = response_df['30 Day MA'] + (response_df['30 Day STD'] * 2)
    response_df['Lower Band'] = response_df['30 Day MA'] - (response_df['30 Day STD'] * 2)
    return response_df


def get_graphical_view(response_df, exchange_currency, desired_currency, today):
    """Function that returns a graphic view of the exchange rate in question
    and the corresponding bollinger bands."""
    # We only want to show the previous month, therefore subset the dataframe
    one_month_ago = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=today.day).strftime("%Y-%m-%d")
    date_15_days_ago = (today - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    response_df = response_df.loc[(response_df.index >= one_month_ago) & (response_df.index <= today.strftime("%Y-%m-%d"))]
    
    # set style, empty figure and axes
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12,6), facecolor='w')
    ax = fig.add_subplot(111)
    
    # Get index values for the X axis for exchange rate DataFrame
    x_axis = response_df.index
    
    # Plot shaded 21 Day Bollinger Band for exchange rate
    ax.fill_between(x_axis, response_df['Upper Band'], response_df['Lower Band'], color='white')
    
    # Plot Adjust Closing Price and Moving Averages
    ax.plot(x_axis, response_df['Adj Close Price'], color='blue', lw=2)
    #ax.plot(x_axis, response_df['30 Day MA'], color='black', lw=2)
    ax.plot(x_axis, response_df['Upper Band'], color='green', lw=2)
    ax.plot(x_axis, response_df['Lower Band'], color='red', lw=2)
    ax.set_xticks([one_month_ago, date_15_days_ago, today.strftime("%Y-%m-%d")])
    
    # Set Title & Show the Image
    ax.set_title(f'30 Day Bollinger Band For {exchange_currency}/{desired_currency} rate')
    ax.set_ylabel(f'Value of 1 {exchange_currency} in {desired_currency}')
    ax.legend(['Adj Close Price', f'Strong {exchange_currency}', f'Weak {exchange_currency}'])
    
    # Compare the value of the exchange rate currencies
    compare = response_bb_df.loc[response_bb_df.index == today.strftime("%Y-%m-%d")]
    if compare['Adj Close Price'].values > compare['Upper Band'].values:
        print(f'The {exchange_currency} is strong, consider making your international transaction today.')
    elif compare['Adj Close Price'].values > compare['Lower Band'].values:
        print(f"The {exchange_currency} is currently trading according to its boundaries.")
    else:
        print(f"The {exchange_currency} is weak, consider making your international transaction another day.")
    return plt.show()


# Functions to calculate clusters

def get_distortion_plot(whole_corpus_notes):
    "Function that retrieves the distortion measures for a range of k clusters."
    # Calculate the vectors and store them in a list of arrays
    _vectrs = []
    for note in whole_corpus_notes:
        v = np.array(vectorizer.infer_vector(note))
        _vectrs.append(v)

    # Calculate the distortion with the vector arrays
    distortions = []
    for k in range(1,10):
        kclusterer = KMeansClusterer(k, distance=cosine_distance)
        assigned_clusters = kclusterer.cluster(_vectrs, assign_clusters=True)

        sum_of_squares = 0
        current_cluster = 0
        for centroid in kclusterer.means():
            current_page = 0
            for index_of_cluster_of_page in assigned_clusters:
                if index_of_cluster_of_page == current_cluster:
                    y = _vectrs[current_page]
                    # Calculate SSE for different K
                    sum_of_squares += np.sum((centroid - y) ** 2)
                current_page += 1
            current_cluster += 1
        distortions.append(round(sum_of_squares))
    
    # Plot values of SSE
    plt.figure(figsize=(15,8))
    plt.subplot(121, title='Elbow curve')
    plt.xlabel('k')
    plt.plot(range(1, 10), distortions)
    plt.grid(True)
    return plt.show()


def get_cluster_topics_with_LDA(recomposed_note_stopped_em):
    "Function that calculates cluster topics for documents in a corpus."
    # Retrieve the different documents
    fully_recomposed_notes = emojis_to_text(recomposed_note_stopped_em)
    # Create the Inputs of LDA model: Dictionary and Corpus
    dct = corpora.Dictionary(fully_recomposed_notes)
    corpus = [dct.doc2bow(note) for note in fully_recomposed_notes]
    # Train the LDA model
    lda_model = LdaMulticore(corpus=corpus, id2word=dct, random_state=100,
                             num_topics=6, passes=10, chunksize=500,
                             batch=False, offset=64, eval_every=0, iterations=100,
                             gamma_threshold=0.001)
    for idx, topic in lda_model.print_topics():
        print('Topic: {} Word: {}'.format(idx, topic))
    return lda_model

    