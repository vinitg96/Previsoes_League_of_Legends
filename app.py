
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

#icon
league_icon_url = "data:image/png;base64,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"

st.set_page_config(page_title="League Previsões", page_icon=league_icon_url, layout="wide", initial_sidebar_state="auto", menu_items=None)

# Caminho do csv
file_PATH = "./data/high_diamond_ranked_10min.csv"

# Funções
@st.cache
def load_data():

    """Função para Carregar Dados Brutos e Dataframe modificado contendo features necessárias para treino"""

    df = pd.read_csv(file_PATH)

    #preditores relevantes + Target
    df_tidy = pd.DataFrame(
        {
        "Kills": df.blueKills - df.redKills,       
        "Gold": df.blueTotalGold - df.redTotalGold,
        "Exp": df.blueTotalExperience - df.redTotalExperience,
        "Dragons": df.blueDragons - df.redDragons,
        "Target": df.blueWins
    })

    return df,df_tidy

def make_slider(slider_label,var_name):

    """ Função para Criar os Sliders"""

    slider = st.slider(slider_label,int(df[var_name].min()), int(df[var_name].max()), int(df[var_name].min()),  key=var_name)

    return slider

def train_predict(df_tidy, newdata):

    """ Função para treinar modelo e retornar probabilidades"""

    X = df_tidy.drop("Target", axis = 1)

    y = df_tidy["Target"]

    clf = LogisticRegression(solver="liblinear",penalty="l1",tol = 0.001, C=0.01, max_iter= 345,random_state=32)

    pipe = make_pipeline(StandardScaler(), clf)

    pipe.fit(X,y)

    return pipe.predict_proba(newdata)

# Criando Dataframes
df,df_tidy = load_data()

# Elementos Visuais 
st.title("League of Legends Previsões")
st.write("Preencha as informações abaixo com base nos primeiros 10 minutos de game para obter a probabilidade de vitória.")
team = st.selectbox(label="Sua equipe", options=["Azul","Vermelho"])
drag = st.selectbox(label = "Primeiro Dragão realizado por", options=["Azul", "Não realizado", "Vermelho"])

if drag == "Time Vermelho":
    drag = -1
elif drag == "Time Azul":
    drag = 1
else:
    drag = 0

## Colunas para receber os sliders (1 para cada equipe)
col1,col2 = st.columns(2)
with col1:
    st.subheader("Time Azul")
    blue_kills = make_slider("Kills",  "blueKills")
    #blue_drag = st.select_slider(label="Primeiro Dragão Realizado?", options=["Sim","Não"],key="blue_drag")
    blue_gold = make_slider("Gold Total", "blueTotalGold") 
    blue_exp = make_slider("Experiência Total", "blueTotalExperience")
    

with col2:
    st.subheader("Time Vermelho")
    red_kills = make_slider("Kills",  "redKills")
    #red_drag = st.select_slider(label="Primeiro Dragão Realiazdo?", options=["Sim","Não"],key="red_drag", value="Não")
    red_gold = make_slider("Gold Total", "redTotalGold")
    red_exp = make_slider("Experiência Total", "redTotalExperience")



#Arrumar input do usuário em array numpy
blue_vars = [blue_kills, blue_gold, blue_exp]
red_vars = [red_kills,red_gold, red_exp]

new_data = []

#calculando diferença para cada feature
for blue,red in zip(blue_vars, red_vars):
    new_data.append(blue-red)

new_data.append(drag)

new_data = np.array([new_data])



# Permitir ao usuário fazer previsões ao apertar o botão
if st.button("Realizar Previsões"):

    preds = train_predict(df_tidy, new_data)

    if team == "Azul":
        st.write(f"Probabilidade de vitória: {preds[0][1]*100:.2f}%")
            
    else:
        st.write(f"Probabilidade de vitória: {preds[0][0]*100:.2f}%")


# Links
html = """<p>
    <p>
    <p>
    <p>
      <a href="https://www.linkedin.com/in/vinicius-torres-05a35695/" rel="nofollow noreferrer">
        <img src= "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABHtJREFUWEelV09olEcU/80mWEuhQmg9xOKhoSBW46EgPUaR4kUPbVOPPUqhhx68GdHtBgr14slDDx4CZl1LkZpDL202NQUPRWhNtnV3I4n5Z3ZNNFRSGt3syJuZN9/MfPNtAp3Lfvt9b9783u/93psZARoXZQ7d4wuA6FX/JQChnoJBL+lj5L2UwZzENj1LLqF1fD/yoi0weLML/W+1rMvMxcnCuOpoYzyxTZbt6gKw/Hq3QOGXJRs5o/AmZUUdY0i/c2f4s80/WnxtERCiKVAY9zglky+PvoOT7/Zg4tE6Lt+dT5wqSwEIdwrnyk9NJmxe3OB3AOgpcmggJQAxXPYjy6B1O67E6gIkRc6DYmEGaPKnB/fi5sfvp7j9/Me/MTK1krznYKNCNSyFoqTIny4qDTNQ+vVScO7D/bh8oi8F4Ovf5nCxPEs5y6gCf4q2cgS7pnMemy1EYZywKmS5HLB1/pjvTQK7vpnAy3as/LKFaL8EOU8XcCDCg2+/gcrZo9au/7vfMdXcSGorkv84LwJYnddqD0vDQeFpwNe2gEw1l52VnsqyWbxz0ihRAQM7a0gcksmdI2tBlRSl3VAXIEoBkEO+BmhabrhsBUTflSsnFflfZ3Fpcg5XPnoPJ/dsoF6vK0hs8u3deUzOr0fpSwAYawtA0a/rjPuAcsoAHHf5O7P44oN9eLY4axcPV6uu/YtzP8+4LcAA5BS4AAKhiULZbjTtoWNe9OTx+WYLy3MP9eLsR1J+Vaexg7rqHWbCpEKgUDbak2qyvGAodpVqOqFmwHRKB2S1Ws2M3GVCQuJ0aSqpKAUxEGGoAZ2CCduAwu+8+NP/XoI6Jo3unMCtwcNeFhjv6dJ9b0PfIQC9F7AG+NmN/FTpfrKgAHZ35fD9J4d8KUiJ/OQc7j1+bkH4AEwKXAXTM1WBfifQHhpQaaXFa/W6ev5pZg1X7y2lVD52pj91urn2x2Pcqj5xxFgYl25pxlRuU6AADqBaq6Fe06VGY3S6gWKlker1GoA/rk+v4Eal6WxIVgO8HTsi5N7hiPDBYK+NnJkqGgAmSXY77wSAbeMa4LowVWT7wOoCxk70pKLyGHDo3B5AsB2HIlMHFAitAdNeyakmJqlDBkDz7V4vgNufZaeAo9iRCMVXI3ZXi0U1OtVA8a+GIyz9eJs0wH1bCpDYitMrSi8MNbMMOT5S+4HhH6zzKACrAf+olLKVwGhFizCxjDYikrtAraZLza3xOIAmihXnyGbyENrSoqOmCuIpAEydC1XnvKtpAHpzGnPyyixdn27ghqLVHRJjZ45Ey7C0HQPVag31es1OPlX6s2NpKRGSBoJTmwIbHFy5D2Qy0LPRQKv5yEEu8M+LllI9VcSbu7rNc2KyuSWxudVOMUC2ajilGdoaERqLJ+boHLgyLcq/L3Q4a4WfOh3L6GpGd6R9XOcdbqapfP7/F6Ih1M14fWTLnl4jXlUE7m3CyXXG60xsHht9L7q0ROiG/NrMMiD3atHoMoyNxEHG/Ww7WnQ0DfRt9iKfb78CLOl4lgOx5t8AAAAASUVORK5CYII="> LinkedIn
      </a> &nbsp; 
      <a href="https://github.com/vinitg96/LeagueOfLegendsPredictions" rel="nofollow noreferrer">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABMpJREFUWEeVV02IHEUUfq97Z8yCIsp6iGgCih7iHhREAzNVs+shB8lBECXJKQchEZUkm+lkL7IQxOD07GYvrhoPIvgvKOrFHKI71bPgQQWJwYMREveirAEVnSTsTD2prv6p7q7uWesyM/WqXr33ve99VYNQNTAyUrqoOR/cRpt40HXkDADu0Ba8QlL23NHordXl2T/DGQAwtmVOSWyo140dzaNiO9TwGwdgBwIBUXZn8TC6XKvVd59/effvpnNzXfw9DUD5pGzUjZP9W1wpNwDwpnyUVRkaa687g+Edqyuz/9jhwBwCkVf1wbzVZQD3yDg4x8KnSoGwFHTYce0+G3qEQHay5YlLBHivPUtV2S1VziAQAhBdEl1+X95nzhMC93rrSHiXKnM6YjoZk0YcFg6UBfmr8NnO1K9RAvWVe70vAHCvXkARXPFyfWLVYVvhBQF+FvjNJ2KvSUoz7bVpifKCmbcEerrv848bJ/t3OpIuIMDtmqZmf+rvucpedaA+3fMf+a3R7h9wUL5rlm00xOm15eZF5SoJoOUFlO9b4bNMIdic2IUOHgAp3xBLrXUMS6tD5nO9u8nFww7SO71O66ckEQTgbWEeFZoCn6EiZHgAmxPPoosreUYnAZhNGwNg0qJMcSKHvB2E5M+gJOGQWGRnwwC4FxRcKE3odRoOOE5oi6sfknPMgZlEFsjhg/4omTMqqFDAmYWvt8nBxDVbv9cm6lPnTz961YSzTFAyURkUmTn+7ZR0rm3Y9MLZGE4i94IuABy3BPCq8NnzacKRjKi66yaxDJOckTnkQPA6ABzKbyBCH7knBgA4mTUSCJ9Xqk0Yzv8ohy5zQcQGCgFrLvkOsOVbfeeZnU7AvWInhKjHDLV2gJlhLlsTnnxj2IItJBptqkTAptv5y8SOTJFRZUiHCISCkvOkSpBQqrTWMTERqCBjWYca6TwHCBQJCyqVKtUWiR4mYMii5UWUQcC8yJgnrjjR0yqHwkvCZy+Oh7iCKNHmphecdoDm89c4AV7GphcccQCWbQdZO8E4z1qZZDJ9YmXrn6ZPRC+USrEKSBJs9rusXo5C1mILiLfFJiBO5H2otb2YZ7wdDAHBDRcR/QGINwPAtmSTlAeDxdbbW7oCoigannjGBXyzLHgCGAY+q2kE5noPoet8ryWWhOjyVnM+uMcZwS8ZB0QjdPHT3ivsqaJj9aARnwPRXkDdQFVCKUfwYH+J/5DoCW+L64D69UsEf/e77NaG4gfBcuIJARyC2dUuW7WLzdoeAHkutcXqm1N1ohuiy0OEE8uuhY/qU4PtN5LNo9GTYmnmE+aJJQQ4Fi8dR8wywTED/vevyfp3Zx/ezASgfrTawX5CeE9Dl15IljsuA4B5Y2pdqRq4T/jND+MVCOa7Kvw/IF5DwMMRIX8WXX5/ZTFzoWQCiEJJlBZhJeiw58wIjeKkebATa6eQZCJCRCABYB2RdhavafVWUnjpoUqgf2frTkCnAp8v2NoxnEv1Q0tq49hXD7gTtR9TqPR9UeBAjuo2DoyGm9NrZx67mPeV5UBJz7ROBO8Twb7wfyMCiC5TCZaOlicozh4RPuh12H5zcYqLiZPt8MwcAmuLo+jAHtFhj1dRjHvBOSnpy/4iP1NNRm39D5AhBpJuROh9AAAAAElFTkSuQmCC" alt="github"> Github
        <i class="fa-brands fa-medium"></i>
      </a>
      <a href = "https://medium.com/@vini.guerra87/prevendo-o-resultado-de-partidas-de-league-of-legends-com-python-3c15c10a8784" rel="nofollow noreferrer">
      <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABGRJREFUWEeVV0GIHEUU/b9nFwIKHiIRkumu3qoJ5ODJFS/mIHj2ph5ySOYkGkRQRDB3vUhEEAweJCYQJEGvgiDkpGfPbrqnq2aMC4uLiJLsmpmSquqqruqqzox9maGq/q/3/3v//24E+6D5o36kRP0LIEG6Rb3jHXKWg38Qla9225r33Jib7KEhV8kzCIBywDY0eNwVyotDkDLz14Yc6fVNAgmiNQYqS1EChnyxnD0rcfkVILzQMqZJU0QZG8kB4HLN+fdhMgc8qvDjrXiF5uVbmMEXRhUmZ8GTWJQA12reXF6nFE8D/YsRdnef2/7z4I+HgJAlHSXRhChXGZaz2Uxlpns8zuIMtJt0fOYsjrZ+9e2G7huOsg1KwrQSzQ1bQoa0rqCCSlQb7DTLYXspUo6HQTxehcsVXGjmzTd9n45O35yRUgPUl6X5jXWwjmzl63j71L3f9w5ca4n0hAi0KA4Q8On1/jaqu0islWgCDQd9gI3HExht7TkrCZAw0Nu0KE3ttU/FQ8d2nfXOAeAHFZ99YvtPgIaS8gECnPCzHjru+ikriACEvD37c8WbF7um1mUnoLPtFr5PNP3aGNjD/fQ7A6/70jG5ACO8ZSJYna+4+KlfabTVUkTnv8dFdf/+3GihBTspdl6XKG+nuLcAWJ5Pq/n8a30GAVixo4eB2UdgRXmxEs1N21t1QEkR43c1n71qxaid0aK8iwAvBQDaYtU6UBeOLYAwYxqA2s93LlZidtPy6jLgQLi+4LTlMsCKchl3PGNgM0DzfIqY8UrwuwooIzs/AizrSog3aF68jDA6owA4AQ5RAGB8tiWgzw+lS+35FACO3q5E87yhoNg9WsrZYjE/ZKT8BSR++r8A2D6g4hwUTB9ANrquKfHfTTQYxTde2gSAMq15o8eAK0NKyn0EeKbrAUby9rDOUp5PQQEIat4MD0bKVR+ACsp00rhpWR8OACvKa4DwZiRC7Cig+XiK2dZ1KVfv1kJ8Zs9OivJDifDxYAZ6lSClPKwFP9lVgSqh0zSHrZUwzT98nMrHJgMS4J+aN0/auBgpjwFge1MKViDfn3F+1QNgLkw1orjFIjBCpL+u7AZbcaISgk6o+WlfXSkhdxDwNT9+JZYuJ13949HDU/f29w/OFoSuEKt1ABwLEv6qRPOUvSN6u2LEdLf+Eze0bi4kX41bg77d0fLRycVicdgD0Kl0Qsh7EvCq30LV2HN1tx5ZeyI5rtuh1Y4tvxGZmdCmuCj1pOvu8pxt8F6WOiKllLXgWR+WoSABlhLyNwI+kdyMCPITahhUsfj89oTn3gojDfiAKCF7CDgJ7+sGistaonS1jQFyVPPmxBDmEIBHgzVQcx9HeCvlwKQ6xbUFKT+vBH8nfbk5k85A38IMno8AsiuWEq+soi8VCfLbmvOgnNdnICnaWAGKWprnlyDLXgHAcwiQSYBGSvjhePXoy98WiweuiAd8+mDsd3hbyr5Fl6K4KyTrIxlkSvXGn9n5D0MUDESskur/AAAAAElFTkSuQmCC"> Medium
    </p> 
    </p>
    </p>"""
st.markdown(html, unsafe_allow_html=True)

