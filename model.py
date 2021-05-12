import shap
import joblib
import pickle
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KDTree
#@st.cache
# IMPORT COLUMNS NAMES
with open('dashboard_col_name.pkl', 'rb') as f:
    dashboard_col_name = pickle.load(f)
with open('col_name_list.pkl', 'rb') as f:
    col_name_list = pickle.load(f)

def load_data():
    n_rows = 100
    path = "C:/Users/EJ/Desktop/Formation/DataScience/P7/flask_project/"
    data = pd.read_csv(path + "application_train.csv",
                       nrows=n_rows, index_col='SK_ID_CURR')
    return data

def process_data_dashboard(dataset):
    process_pretraitement_dashboard = open(
        "pretraitement_dashboard.joblib", "rb")
    pretraitement_dashboard = joblib.load(process_pretraitement_dashboard)
    data_for_dashboard = pretraitement_dashboard.transform(dataset)
    return data_for_dashboard

def process_prediction(dataset):
    process_pretraitement_prediction = open(
        "pretraitement_prediction.joblib", "rb")
    pretraitement_prediction = joblib.load(process_pretraitement_prediction)

    process_prediction = open("model.joblib", "rb")
    model = joblib.load(process_prediction)

    data_for_prediction = pretraitement_prediction.transform(
        dataset)#.drop(['TARGET'], axis=1))
    data_for_prediction = pd.DataFrame(
        data_for_prediction, index=dataset.index, columns=col_name_list)

    prediction = model.predict(data_for_prediction)
    df_prediction = pd.DataFrame(
        prediction, index=dataset.index, columns=["label_predicted"])

    proba_prediction = model.predict_proba(data_for_prediction)
    df_proba_prediction = pd.DataFrame(proba_prediction, index=dataset.index, columns=[
                                       "proba_label_0", "proba_label_1"])

    return data_for_prediction, df_prediction, df_proba_prediction

def prediction_score(dataset, dataset_target):
    process_prediction = open("model.joblib", "rb")
    model = joblib.load(process_prediction)
    prediction = model.predict(dataset)
    score = model.score(dataset, dataset_target)
    return score

def get_max_value(dataset, list_col):
    max_value = []
    for i in range(0, len(list_col)):
        var = list_col[i]
        inter_list = dataset[var]
        maximum_value = max(inter_list)
        max_value.append(maximum_value)
    return max_value

def calcul_mean(dataset, list_col):
    result = []
    for i in range(0, len(list_col)):
        var = list_col[i]
        value = dataset[var].mean()
        result.append(value)
    return result

def radar_value(value_list, max_value_list):
    result = []
    for i in range(0, len(value_list)):
        a = value_list[i]
        b = max_value_list[i]
        value = float(a/b)
        result.append(value)
    return result

def convert_df_to_list(dataset, list_col):
    result = []
    for i in range(0, len(list_col)):
        var = list_col[i]
        value = int(dataset[var].values[0])
        result.append(value)
    return result

def radar_chart(df_good_radar, df_bad_radar, identifiant_client, df_client_radar, list_col):
    placeholder = st.empty()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df_good_radar,
        theta=list_col,
        fill='toself',
        name='Good_Client'
    ))
    fig.add_trace(go.Scatterpolar(
        r=df_bad_radar,
        theta=list_col,
        fill='toself',
        name='Bad_Client'
    ))
    fig.add_trace(go.Scatterpolar(
        r=df_client_radar,
        theta=list_col,
        fill='toself',
        name='Client_{}'.format(identifiant_client)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )
    placeholder.write(fig)

def get_index(index_df, list_ind):
    index_list = []
    for i in range(0, len(list_ind)):
        value_to_find = list_ind[i]
        value = index_df[index_df.position == value_to_find].index.values.tolist()[
            0]
        index_list.append(value)
    return index_list

def main():
  
    df = load_data()
    X = df.drop(['TARGET'], axis=1)
    y = pd.DataFrame(df['TARGET'], index=df.index, columns=['TARGET'])

    X['AGE'] = (X['DAYS_BIRTH']/-365).astype(int)
    X['DAYS_EMPLOYED'].replace({365243: 0}, inplace=True)
    X['YEARS_EMPLOYED'] = round((X['DAYS_EMPLOYED']/-365).astype(int), 0)
    X.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    
    X = X[dashboard_col_name] 
    df_dashboard_inter = pd.DataFrame(process_data_dashboard(
        X), index = X.index, columns = dashboard_col_name)
    df_dashboard = y.merge(
        df_dashboard_inter, left_index=True, right_index=True)
    
    # CREATE DATA SET DASHBOARD LIGHT
    cols_keep = ['AGE', 'AMT_ANNUITY','AMT_CREDIT', 'AMT_GOODS_PRICE', 
    'AMT_INCOME_TOTAL' , 'FLAG_OWN_REALTY', 'CODE_GENDER', 'YEARS_EMPLOYED', 'TARGET']
    df_light = df_dashboard[cols_keep]
  
    # REORGANISATION & RENAME COLUMNS
    df_light.columns = ['AGE','AMOUNT ANNUITY', 'AMOUNT CREDIT', 'GOODS VALUE',
    'ANNUAL INCOME', 'FLAT OWNER', 'YEARS EMPLOYED', 'TARGET']

    category = ['AGE', 'ANNUAL INCOME',
                'AMOUNT CREDIT', 'AMOUNT ANNUITY', 'GOODS VALUE']

    ####    DASHBOARD   ####

    #   1. CREATE SIDE BAR
    st.title('Analyse Client')
    id_client = st.sidebar.selectbox('Choisir ID Client :', df_light.index)
    st.sidebar.table(df_light.loc[id_client][category])
    data_client = df_light.loc[id_client:id_client][category]

    #   2. RADAR CHART
    
    max_value_list = get_max_value(df_light, category)

    # SPLIT TWO GROUPS OF CLIENTS
    df_good_clients = df_light[df_light['TARGET'] == 0][category]
    df_bad_clients = df_light[df_light['TARGET'] == 1][category]

    # CREATE THREE DATA SET FOR RADAR CHART
    df_good = calcul_mean(df_good_clients, category)
    df_good = radar_value(df_good, max_value_list)
    df_bad = calcul_mean(df_bad_clients, category)
    df_bad = radar_value(df_bad, max_value_list)
    client_data_radar = convert_df_to_list(data_client, category)
    client_data_radar = radar_value(client_data_radar, max_value_list)

    # RADAR CHART
    st.header("1. Comparaison données client versus autres clients")
    st.write(radar_chart(df_good, df_bad, id_client , client_data_radar, category))

    #   3. PREDICTION PART
    data_for_prediction, df_prediction, df_proba_prediction = process_prediction(
        X)
    
    model_prediction = open("model.joblib", "rb")
    model = joblib.load(model_prediction)
    model.fit(data_for_prediction, df["TARGET"])

    # PREDICTION OF CLIENT
    client_pred_proba = df_proba_prediction.loc[id_client:
                                                id_client]["proba_label_1"].values[0]*100

    # NEIGHBORHOOD CLIENTS & THEIR PROBABILITY PREDICTION
    df_voisins = pd.get_dummies(df_light.iloc[:, ])
    tree = KDTree(df_voisins)
    nb_neighborhood = 100
    idx_voisins = tree.query(df_voisins.loc[id_client:id_client].fillna(
        0), k=nb_neighborhood)[1][0].tolist()
    df_index = pd.DataFrame(data=range(0, len(df_light), 1),
                            index=df_light.index, columns=['position'])
    new_idx_voisins = get_index(df_index, idx_voisins)
    data_voisins = df_proba_prediction.loc[new_idx_voisins]
    predict_voisins = 100 * (data_voisins["proba_label_1"].mean())

    if st.sidebar.checkbox("Afficher Proba Défaut", False, key=2):
        st.header('2. Probabilité de défaut pour le client {}'.format(id_client))
        placeholder = st.empty()
        fig0 = go.Figure(go.Indicator(mode="number+gauge+delta",
                                    value=client_pred_proba,
                                    domain={
                                        'x': [0, 1],
                                        'y': [0, 1]
                                    },
                                    delta={'reference': predict_voisins,
                                            'increasing': {'color': 'red'},
                                            'decreasing': {'color': 'green'},
                                            'position': "top"
                                            },
                                    title={'text': "<b>En %</b><br><span style='color: gray; font-size:0.8em'></span>",
                                            'font': {"size": 14}
                                            },
                                    gauge={
                                        # 'shape': "bullet",
                                        'axis': {
                                            'range': [None, 100]
                                        },
                                        'threshold': {
                                            'line': {'color': "white", 'width': 3},
                                            'thickness': 0.75,
                                            'value': predict_voisins
                                        },
                                        'bgcolor': "white",
                                        'steps': [
                                            {'range': [0, 50],
                                            'color': "lightgreen"},
                                            {'range': [50, 60],
                                            'color': "orange"},
                                            {'range': [60, 100],
                                            'color': "red"}
                                        ],
                                        'bar': {'color': "darkblue"}
                                    }
                                    )
                        )
        fig0.update_layout(height=250)
        # st.plotly_chart(fig0)​
        st.write(fig0)

        st.markdown(
            'Proba défaut client sélectionné : **{0:.1f}%**'.format(client_pred_proba))
        st.markdown('Proba défaut clients similaires : **{0:.1f}%** \
        (critères de similarité : âge, genre,situation familiale, éducation, profession)'.format(predict_voisins))

    #   4. INTERPRETATION
    features = ['EXT_SOURCE_1', 'EXT_SOURCE_2',
                'EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_ANNUITY']

    data_client_interpretation = df.loc[id_client:id_client][features]

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.sidebar.checkbox("Afficher Explication Proba Def", False, key=0):
        st.header("3. Variables principales influançant l'accord de crédit")
        placeholder = st.empty()

        ind = df_index.loc[id_client]["position"]

        shap.initjs()
        explainer = shap.LinearExplainer(
            model.named_steps['classifier'], data_for_prediction)
        shap_values = explainer.shap_values(data_for_prediction)
        df_prediction_array = data_for_prediction.values
        #shap.summary_plot(
        #    shap_values,
        #    df_prediction_array,
        #    feature_names=data_for_prediction.columns,
        #    max_display=10
        #)
        #st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        #pl.clf()

        shap.force_plot(
            explainer.expected_value,
            shap_values[ind, :],
            df_prediction_array[ind, :],
            feature_names=data_for_prediction.columns,
            matplotlib=True,
        )
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        pl.clf()

if __name__ == "__main__":
    main()
