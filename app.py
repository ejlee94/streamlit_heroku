import shap, joblib, pickle
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
    data = pd.read_csv("sample_data.csv",
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

def get_max_value(dataset, list_col, ext_source):
    max_value = []
    for i in range(0, len(list_col)):
        var = list_col[i]
        if var in ext_source:
            maximum_value = 1
        else :
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
        value = dataset[var].values[0]
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
                visible=False,
                range=[0, 1]
            )),
        showlegend=False
    )
    #placeholder.write(fig)
    fig.update_layout()
    return fig

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
    X['YEARS_EMPLOYED_PERCENT'] = X['YEARS_EMPLOYED'] / X['AGE']    
    X.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    #CREATE SOME VARIABLES
    X["CREDIT_ANNUITY_RATIO"] = X["AMT_CREDIT"] / X["AMT_ANNUITY"]
    X["INCOME_ANNUITY_RATIO"] = X["AMT_INCOME_TOTAL"] / X["AMT_ANNUITY"]
    X["INCOME_CREDIT_RATIO"] = X["AMT_INCOME_TOTAL"] / X["AMT_CREDIT"]
    X["CREDIT_GOODS_PRICE_RATIO"] = X["AMT_CREDIT"] / X["AMT_GOODS_PRICE"]
    X["CREDIT_DOWNPAYMENT"] = X["AMT_GOODS_PRICE"] / X["AMT_CREDIT"]
    X["CREDIT_INCOME_PERCENT"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
    X["ANNUITY_INCOME_PERCENT"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
    X["RATIO_CREDIT_GOODS_PRICE"] = X["AMT_CREDIT"] / X["AMT_GOODS_PRICE"]
    X["DIFF_GOODS_PRICE_CREDIT"] = X["AMT_CREDIT"] - X["AMT_GOODS_PRICE"]
    X['CREDIT_TERM'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
   
    X = X[dashboard_col_name] 
    df_dashboard_inter = pd.DataFrame(process_data_dashboard(
        X), index = X.index, columns = dashboard_col_name)
    df_dashboard = y.merge(
        df_dashboard_inter, left_index=True, right_index=True)
    
    # CREATE DATA SET DASHBOARD LIGHT
    cols_keep = ['AGE', 'AMT_ANNUITY','AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
    'CNT_CHILDREN', 'CREDIT_ANNUITY_RATIO', 'CREDIT_TERM', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
    'EXT_SOURCE_3', 'INCOME_ANNUITY_RATIO', 'CODE_GENDER', 'FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'TARGET']
    df_light = df_dashboard[cols_keep]
  
    # REORGANISATION & RENAME COLUMNS
    df_light.columns = ['AGE','AMOUNT ANNUITY', 'AMOUNT CREDIT', 'GOODS VALUE', 'ANNUAL INCOME',
    'NB CHILDREN', 'RATIO CREDIT ANNUITY', 'CREDIT TERM', 'NORM. EXT_SCORE 1', 'NORM. EXT_SCORE 2',
    'NORM. EXT_SCORE 3', 'RATIO INCOME ANNUITY', 'CODE GENDER', 'FLAT OWNER', 'EDUCATION',
    'FAMILY STATUS', 'HOUSING TYPE', 'INCOME TYPE', 'JOB', 'TARGET']

    category = ['AGE', 'ANNUAL INCOME', 'AMOUNT ANNUITY', 'GOODS VALUE','AMOUNT CREDIT',
    'NORM. EXT_SCORE 1', 'NORM. EXT_SCORE 2', 'NORM. EXT_SCORE 3']

    ext_source = ['NORM. EXT_SCORE 1', 'NORM. EXT_SCORE 2', 'NORM. EXT_SCORE 3']
    int_source = ['AGE', 'ANNUAL INCOME', 'AMOUNT ANNUITY', 'GOODS VALUE','AMOUNT CREDIT']
    
    ########################
    ####    DASHBOARD   ####
    ########################

    #________________________
    #   1. CREATE SIDE BAR
    #________________________
    st.title('Analyse Client')

    id_client = st.sidebar.selectbox('Choisir ID Client :', df_light.index)
    data_client = df_light.loc[id_client:id_client][category]
    for x in ext_source:
        data_client[x] = round(data_client[x].astype(float), 3) 

    #____________________________________________________
    #   2. CREATE DATAFAT FOR RADAR CHART AND SHOW GRAPH
    #____________________________________________________
    st.header("1. Comparaison données client versus autres clients")

    max_value_list = get_max_value(df_light, category, ext_source)

    # SPLIT TWO GROUPS OF CLIENTS
    df_good_clients = df_light[df_light['TARGET'] == 0][category]
    df_bad_clients = df_light[df_light['TARGET'] == 1][category]
    
    # CREATE A DATAFRAME CONTAINS MEAN VALUE OF EACH COLUMN
    df_good_clients_mean = pd.DataFrame(df_good_clients.mean(axis = 0), columns = ['good_client'])
    df_bad_clients_mean = pd.DataFrame(df_bad_clients.mean(axis = 0), columns = ['bad_client'])
    df_client = pd.DataFrame(data_client[category].T)
    df_group = df_good_clients_mean.merge(df_bad_clients_mean, left_index = True, right_index = True)
    df_group = df_group.merge(df_client, left_index = True, right_index = True)
    df_group.columns = ["good_client", "bad_client", "client_n°{}".format(id_client)]
    for x in int_source:
        df_group.loc[x, :] = round(df_group.loc[x,:].astype(int), )
    for x in ext_source:
        df_group.loc[x,:] = round(df_group.loc[x,:].astype(float), 3)
    
    # CREATE THREE DATA SET FOR RADAR CHART
    df_good = calcul_mean(df_good_clients, category)
    df_good = radar_value(df_good, max_value_list)
    df_bad = calcul_mean(df_bad_clients, category)
    df_bad = radar_value(df_bad, max_value_list)
    client_data_radar = convert_df_to_list(data_client, category)
    client_data_radar = radar_value(client_data_radar, max_value_list)
    
    # SHOW RADAT CHART
    st.write(radar_chart(df_good, df_bad, id_client , client_data_radar, category))    
    st.header("La table des données")
    st.write(df_group)
    
    #_____________________
    #   3. PREDICTION PART
    #_____________________
    # PREDICTION FOR ALL DATASET
    data_for_prediction, df_prediction, df_proba_prediction = process_prediction(X)
    
    # GET PREDICTION OF CLIENT
    client_pred_proba = df_proba_prediction.loc[id_client:id_client]["proba_label_1"].values[0]*100

    # NEIGHBORHOOD CLIENTS & THEIR PROBABILITY PREDICTION
    df_voisins = pd.get_dummies(df_light.iloc[:, :-1])
    tree = KDTree(df_voisins)
    nb_neighborhood = 10
    idx_voisins = tree.query(df_voisins.loc[id_client:id_client].fillna(0), k=nb_neighborhood)[1][0].tolist()
    df_index = pd.DataFrame(data=range(0, len(df_light), 1),
                            index=df_light.index, columns=['position'])
    new_idx_voisins = get_index(df_index, idx_voisins)
    data_voisins = df_proba_prediction.loc[new_idx_voisins]
    predict_voisins = 100 * (data_voisins["proba_label_1"].mean())
    
    # SHOW PROBA GRAPH
    if st.sidebar.checkbox("Afficher Proba Défaut", False, key=2):
        st.header('2. Probabilité de défaut de rembourseemnt pour le client {}'.format(id_client))
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
        st.write(fig0)
        st.markdown(
            'Proba. de défaut de remboursement pour client {0} : **{1:.1f}%**'.format(id_client, client_pred_proba))
        st.markdown("Proba défaut clients similaires : **{0:.1f}%**".format(predict_voisins))
        st.write("Critères de similarité : Âge, Sexe, Situation familiale, Niveaus d'éducation, Profession, Niveau de Revenu, Montant de crédit, Scores de sources externe")
    
    #__________________________
    #   4. FEATURES IMPORTANCES
    #__________________________

    #IMPORT THE MODEL
    model_prediction = open("model.joblib", "rb")
    model = joblib.load(model_prediction)
    model.fit(data_for_prediction, df["TARGET"])
    
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    ind = df_index.loc[id_client]["position"]
    shap.initjs()
    explainer = shap.LinearExplainer(
        model.named_steps['classifier'], data_for_prediction)
    shap_values = explainer.shap_values(data_for_prediction)
    df_prediction_array = data_for_prediction.values

    if st.sidebar.checkbox("Afficher Variables Importantes", False, key=0):
        st.header("3. Variables principales influançant l'accord de crédit")
        placeholder = st.empty()

        shap.summary_plot(
            shap_values,
            df_prediction_array,
            feature_names=data_for_prediction.columns,
            max_display=10
        )
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        pl.clf()
        with st.beta_expander("Les variables principales qui influencent l'accord de crédit sont :", expanded=False):
            st.text(
                """           
                - AGE : plus l'âge de client est important, moins le risque de défaut de remboursement existe
                - NORM EXT_SOURCE1 & NORM EXT_SOURCE2 & NORM EXT_SOURCE3
                    - plus ces trois variables de sources externes sont importants, moins le risque de défaut de remou
                """
            )
    
    #____________________
    #   5. INTERPRETATION
    #____________________
    if st.sidebar.checkbox("Afficher Explication", False, key=0):
        st.header("4. Explication du modèle")
        placeholder = st.empty()
        st.markdown("variables influançant l'accord de crédit au client {}".format(id_client))
        shap.force_plot(
            explainer.expected_value,
            shap_values[ind, :],
            df_prediction_array[ind, :],
            feature_names=data_for_prediction.columns,
            matplotlib=True,
        )
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        pl.clf()
        ind_shap_values = pd.DataFrame(data = shap_values[ind, :], index=data_for_prediction.columns)
        ind_values = pd.DataFrame(data = df_prediction_array[ind, :], index=data_for_prediction.columns)
        uderstanding_ind_values = ind_shap_values.merge(ind_values, left_index=True, right_index=True)
        uderstanding_ind_values.columns = ['shap_values', 'real_values']
        uderstanding_ind_values["diff"] = uderstanding_ind_values['shap_values'] - uderstanding_ind_values['real_values']
        uderstanding_ind_values["abs_diff"] = uderstanding_ind_values["diff"].apply(lambda x : abs(x))
        uderstanding_ind_values = uderstanding_ind_values.sort_values(by = ['abs_diff'], ascending = False)

        if id_client in new_idx_voisins:
            new_idx_voisins.remove(id_client)
        voisins = df_light.loc[new_idx_voisins]

        quanti_col = ['AGE', 'AMOUNT ANNUITY', 'AMOUNT CREDIT', 'GOODS VALUE','ANNUAL INCOME',
        'NB CHILDREN', 'RATIO CREDIT ANNUITY', 'CREDIT TERM', 'NORM. EXT_SCORE 1', 'NORM. EXT_SCORE 2',
         'NORM. EXT_SCORE 3', 'RATIO INCOME ANNUITY']
        df_voisins_explication = voisins[quanti_col]
        df_voisins_explication_mean = pd.DataFrame(df_voisins_explication.mean(axis = 0), columns = ['clients similaires'])
        
        df_explication = df_voisins_explication_mean.merge(df_light.loc[id_client:id_client][quanti_col].T, left_index=True, right_index=True)
        for x in quanti_col:
            if x in ext_source:
                df_explication.loc[x,:] = round(df_explication.loc[x,:].astype(float), 3)
            else :
                df_explication.loc[x, :] = round(df_explication.loc[x,:].astype(int),0 )
        
        st.write(uderstanding_ind_values["diff"].head(5))
        st.write(df_explication)
#   
if __name__ == "__main__":
    main()