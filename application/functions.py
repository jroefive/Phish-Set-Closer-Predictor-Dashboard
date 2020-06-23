import pandas as pd
import lightgbm as lgb
import category_encoders as ce
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
import dash_table


def impute_sets(df):
    for col in df.columns:
        df[col] = df[col].fillna((df[col].mean()))

    return df

# Function to split the data for training, validating, and testing.  Taken directly from the Kaggle mini-course
def get_data_splits(dataframe, valid_fraction=0.1):
    valid_fraction = 0.1
    valid_size = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_size * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_size * 2:-valid_size]
    test = dataframe[-valid_size:]

    train = impute_sets(train)
    valid = impute_sets(valid)
    test = impute_sets(test)

    return train, valid, test

# Function to train the model and produce vallidation and test scores.  Lifted from Kaggle with some alterations to create a text file of the model.
def train_model(train, valid, test=None, feature_cols=None):

    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

    param = {'num_leaves': 64, 'objective': 'binary',
             'metric': 'auc', 'seed': 7}
    num_boost_round = 1000
    model = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                    early_stopping_rounds=10, verbose_eval=False    )

    valid_pred = model.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)

    if test is not None:
        test_pred = model.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['outcome'], test_pred)
        return model, valid_score, test_score
    else:
        return model, valid_score

def get_show_date_id(date):
    track_length_combined = pd.read_csv('https://jroefive.github.io/track_length_combined')
    sd = str(date)
    # Get ID number for show date that was chosen
    show_date_id = track_length_combined[track_length_combined['date']==sd]['order_id'].values
    # Previous line creates a series of the same ID values, so need to reset to equal just the first value.
    show_date_id = show_date_id[0]
    return show_date_id

def call_model(date, cutoff, num_features):
    #Call all data for modeling
    set_close_model_data = pd.read_csv('app/data/set_close_all')

    #A list of all possible model input descriptions
    feature_descriptions = ['key','ID number of track', 'ID number of Show', 'ID number of Show based on order', 'Date of Show', 'Overall Position of Song in Show',
        'Set Number of Song', 'Position of Song in Set', 'Did the song close the set?', 'Average songs in a set over the last 10 shows',
         'Average songs in a set over the last 50 shows','Average songs in a set over the last 100 shows',
         'A measure of the percentage into a set based on the number of songs played with the numerator as the set position and the average number of songs in a set over the past 10 shows as the denominator.',
         'A measure of the percentage into a set based on the number of songs played with the numerator as the set position and the average number of songs in a set over the past 50 shows as the denominator.',
         'A measure of the percentage into a set based on the number of songs played with the numerator as the set position and the average number of songs in a set over the past 100 shows as the denominator.',
         'The average placement of the song over the last 10 shows based on how many minutes into the show over the full length of the show',
         'The average placement of the song over the last 50 shows based on how many minutes into the show over the full length of the show',
         'The average placement of the song over the last 100 shows based on how many minutes into the show over the full length of the show',
         'Percentage of time the song opens a set', 'Percentage of time a song is the second song in a set', 'Percentage of time a song is the second to last song in a set',
         'Percentage of all times played that the song closes a set', 'The amount of minutes into a set when the song started',
         'A measure of the percentage into a set based on the amount of time into the set the songs starts as the numerator and the average sdt duration over the past 10 shows as the denominator.',
         'A measure of the percentage into a set based on the amount of time into the set the songs starts as the numerator and the average sdt duration over the past 50 shows as the denominator.',
         'A measure of the percentage into a set based on the amount of time into the set the songs starts as the numerator and the average sdt duration over the past 100 shows as the denominator.',
         'The average placement in a set for the song over the last 50 shows',
         'The standard deviation of the average placement in a set for the song over the last 50 shows',
         'The average duration of a set over the last 10 shows',
         'The average duration of a set over the last 50 shows',
         'The average duration of a set over the last 100 shows', 'The number of times the song has been played', 'The set the song was played in']

    #zip together feature names and descrptions into a dictionary for display
    all_features = list(set_close_model_data.columns)
    all_features = all_features + ['title_count']
    all_features = all_features + ['set_count']
    feature_desc_dict = dict(zip(all_features,feature_descriptions))

    # change set close t/f to 1 and 0.
    set_close_model_data = set_close_model_data.assign(outcome=(set_close_model_data['setclose'] == 't').astype(int))

    # Count encoding - Change title and set number to integers based on how often they are in the dataframe
    cat_features = ['title', 'set']
    count_enc = ce.CountEncoder()
    count_encoded = count_enc.fit_transform(set_close_model_data[cat_features])

    # Add data cols for ecoded categories
    baseline_data = set_close_model_data.join(count_encoded.add_suffix("_count"))
    show_date_id = get_show_date_id(date)

    # Sort by show date and then drop all shows after the input date
    baseline_data = baseline_data.sort_values('order_id')
    baseline_data = baseline_data[baseline_data.order_id <= show_date_id]

    #Pull just the songs from the chosen show to pass into model for predictions
    songs_from_show_to_test_df = baseline_data[baseline_data.order_id == show_date_id]

    #Update the cutoff if there aren't enough songs in the set to be able to make a prediction
    og_cutoff = cutoff
    while songs_from_show_to_test_df[songs_from_show_to_test_df.timeintoset > cutoff].shape[0] < 4:
        cutoff -= 1

    cutoff_change_msg = ''
    if og_cutoff != cutoff:
        cutoff_change_msg = 'Cutoff changed to ' + str(cutoff) + ' minutes to ensure that four or more songs included for testing.'


    # Drop all songs that started before the cutoff time
    if cutoff > 0:
        baseline_data = baseline_data[baseline_data.timeintoset >= cutoff]

    # Save a copy of the data without dropping the title for use in output
    baseline_w_title = baseline_data

    # Drop all columns with categorical values before training the model
    baseline_data = baseline_data.drop(['title', 'set', 'setclose', 'date'], axis=1)

    # Train the model with all features included first
    feature_cols = baseline_data.columns.drop(['outcome'])
    train, valid, test = get_data_splits(baseline_data)

    # Reduce the model to the most important inputs based on num_feature choice
    selector = SelectKBest(f_classif, k=num_features)
    X_new = selector.fit_transform(train[feature_cols], train['outcome'])
    selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                     index=train.index,
                                     columns=feature_cols)
    selected_columns = selected_features.columns[selected_features.var() != 0]
    dropped_columns = selected_features.columns[selected_features.var() == 0]

    # Train model with only the top x features
    model, valid_score, test_score = train_model(train.drop(dropped_columns, axis=1), valid.drop(dropped_columns, axis=1),
                      test.drop(dropped_columns, axis=1), selected_columns)

    #Get a df of feature names used and their importances for bar chart
    feature_names = model.feature_name()
    feature_imps = model.feature_importance()
    feature_labels = [feature_desc_dict[x] for x in feature_names]
    feature_tuples = list(zip(feature_names,feature_labels))
    feature_desc_df = pd.DataFrame(feature_tuples, columns=['Feature Name','Feature Description'])

    # Make predictions just for the songs in the target show
    # Sort to make sure songs are in order
    baseline_w_title = baseline_w_title.sort_values('position')

    # Set up a df for passing to the model for predicting and getting accuracy score
    closer_predict_songs_df = baseline_w_title[baseline_w_title['order_id'] == show_date_id]
    closer_prediction_values = model.predict(closer_predict_songs_df[selected_columns])
    close_score = metrics.roc_auc_score(closer_predict_songs_df['outcome'], closer_prediction_values)

    #Choose columns to display
    closer_predict_songs_df['title'] = closer_predict_songs_df.id.map(baseline_w_title.set_index('id')['title'])
    display_cols = ['title', 'set', 'setposition', 'timeintoset']
    display_cols = display_cols + feature_names
    display_cols = display_cols + ['setclose']
    closer_predict_display = closer_predict_songs_df[display_cols]

    #Convert predictions to prectages and add percent sign for display
    closer_prediction_values = [round(x*100,2) for x in closer_prediction_values]
    for ind, i in enumerate(closer_prediction_values):
        closer_prediction_values[ind] = "{}%".format(i)

    # Attach prediction values to the input df for display
    closer_predict_display['Chance of Closing'] = closer_prediction_values

    # Create a df of accuracy scores for display
    score_labels = ['Validation Score: ', "Test Score: ", "Predict Score: "]
    scores = [valid_score, test_score, close_score]
    scores = [round(x * 100, 2) for x in scores]
    for ind, i in enumerate(scores):
        scores[ind] = "{}%".format(i)
    score_table_tuples = list(zip(score_labels,scores))
    score_table = pd.DataFrame(score_table_tuples, columns=['Score_Type','Score'])

    return closer_predict_display, score_table, cutoff_change_msg, feature_names, feature_imps, feature_desc_df

def generate_table(df):
    if df.empty:
        return ''
    else:
        return dash_table.DataTable(
            data=df.to_dict('records'),
            style_data={
            'whiteSpace': 'normal',
            'height': 'auto'},
            columns=[{'id': c, 'name': c} for c in df.columns],
            style_cell={'textAlign': 'left', 'width': '30px'},
            style_data_conditional=[
                {
                    'if': {
                        'column_id': 'Chance of Closing',
                    },
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'column_id': 'title',
                    },
                    'width': '90px',
                },
            ],
            style_as_list_view=True,)