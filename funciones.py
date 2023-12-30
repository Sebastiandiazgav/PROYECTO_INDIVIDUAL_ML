import pandas as  pd
import pickle
from surprise import SVD, Dataset, Reader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import Dict




#Funcion 1

# Función para calcular el año con más item_id para un género específico
def PlayTimeGenre_func(genero: str) -> Dict[str, int]:
    # Cargar los datos de juegos de Steam
    steam_games = pd.read_csv('C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Data/steam1_output.csv')
    # Cargar los datos de revisiones de usuarios
    user_reviews = pd.read_csv('C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Data/user_reviews1.csv')


    # Eliminar filas con valores faltantes en los datos de juegos
    steam_games.dropna(inplace=True)

    # Convertir el año de lanzamiento a entero
    steam_games['Year'] = steam_games['Year'].astype(int)

    # Fusionar los datos de juegos y revisiones en 'id'
    merged_data = pd.merge(user_reviews, steam_games, left_on='item_id', right_on='id', how='inner')

    # Filtrar los datos para obtener solo los juegos del género dado
    genre_data = merged_data[merged_data[f'genre_{genero}'] == 1]

    # Agrupar los datos por año y contar la cantidad de item_id
    total_items_by_year = genre_data.groupby('Year')['item_id'].count()

    # Encontrar el año con más item_id
    max_items_year = total_items_by_year.idxmax()

    return {"Año de lanzamiento con más item_id para Género seleccionado": int(max_items_year)}






#Funcion 2

def UserForGenre_func(genero:str):
    # Cargo el csv de los generos con usuarios con mas horas
    users_gen = pd.read_csv('C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Data/genero_unifi.csv')

    # Verificar si el género especificado está presente en el conjunto de datos
    if genero.lower() not in [x.lower() for x in users_gen['Género'].tolist()]:
        return "No se encontró ese genero"
    
    # Filtrar los datos para el género especificado
    gen = users_gen[users_gen['Género'].str.lower() == genero.lower()] # Busco el genero especificado
    
    # Devolver un diccionario con información sobre usuarios y horas jugadas para el género dado
    return { 
        'Usuario':gen['Usuario'].tolist(),
        'Horas jugadas':gen['Año_Horas'].tolist()      
    }



#Funciones 3 , 4, 5
# Cargar el modelo SVD entrenado

user_reviews1 = pd.read_csv('C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Data/reviews.csv')
reader_svd = Reader(rating_scale=(0, 4))
data_svd = Dataset.load_from_df(user_reviews1[['user_id', 'item_id', 'rating']], reader_svd)
with open("C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Modelo/model1.pkl", "rb") as f:
    model_svd = pickle.load(f)

# Cargar el modelo KNN entrenado
with open("C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Modelo/model_knn.pkl", "rb") as f:
    model_knn = pickle.load(f)




# Función 3 para obtener recomendaciones de usuarios
def UsersRecommend_func(year: int):
    try:
        # Filtrar reseñas para el año dado
        reviews_for_year = user_reviews1[user_reviews1['YearOnly'].astype('int64') == year]

        # Imprimir para depuración
        print("Reseñas para el año {}: {}".format(year, reviews_for_year))

        # Obtener las predicciones usando el modelo SVD
        predictions = []
        for index, row in reviews_for_year.iterrows():
            user_id, item_id, rating = row['user_id'], row['item_id'], row['rating']
            prediction = model_svd.predict(user_id, item_id)
            predictions.append((item_id, prediction.est))

        # Imprimir para depuración
        print("Predicciones: {}".format(predictions))

        # Ordenar las predicciones por rating descendente y obtener las top 3
        top_3_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]

        # Formatear el resultado como se espera en el ejemplo de retorno
        result = [{"Puesto {}".format(i + 1): game_id} for i, (game_id, rating) in enumerate(top_3_recommendations)]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la aplicación: {str(e)}")







#Funcion 4
def UsersNotRecommend_logic(year: int):
    
    try:
        # Filtrar reseñas para el año dado
        reviews_for_year = user_reviews1[(user_reviews1['YearOnly'] == year) & (user_reviews1['recommend'] == False)]

        # Imprimir para depuración
        print("Reseñas para el año {}: {}".format(year, reviews_for_year))

        # Obtener las predicciones usando el modelo SVD
        predictions = []
        for index, row in reviews_for_year.iterrows():
            prediction = model_svd.predict(row['user_id'], row['item_id'])
            predictions.append((row['item_id'], prediction.est))

        # Imprimir para depuración
        print("Predicciones: {}".format(predictions))

        # Ordenar las predicciones por rating descendente y obtener las top 3
        top_3_not_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]

        # Formatear el resultado como se espera en el ejemplo de retorno
        result = [{"Puesto {}".format(i + 1): game_id} for i, (game_id, rating) in enumerate(top_3_not_recommendations)]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la aplicación: {str(e)}")






#Funcion 5

def sentiment_analysis_func(year: int):
    # Filtrar reseñas para el año dado
    reviews_for_year = user_reviews1[user_reviews1['YearOnly'] == year]

    # Contar la cantidad de registros con análisis de sentimiento negativo, neutral y positivo
    negative_count = reviews_for_year[reviews_for_year['sentiment_analysis'] == 0].shape[0]
    neutral_count = reviews_for_year[reviews_for_year['sentiment_analysis'] == 1].shape[0]
    positive_count = reviews_for_year[reviews_for_year['sentiment_analysis'] == 2].shape[0]

    # Formatear el resultado como se espera en el ejemplo de retorno
    result = {"Negative": negative_count, "Neutral": neutral_count, "Positive": positive_count}

    return result



#funcion 6


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Lee tu conjunto de datos
user_reviews = pd.read_csv('C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Data/reviews.csv')   # Asegúrate de especificar la ruta correcta

def recommend(user_id: str) -> List[int]:
    # Filtrar reseñas para el usuario dado
    reviews_for_user = user_reviews[user_reviews['user_id'] == user_id]

    if reviews_for_user.empty:
        raise Exception(f"No hay suficientes datos para el usuario {user_id}")

    # Obtener la matriz de reseñas de usuarios
    user_reviews_matrix = pd.pivot_table(reviews_for_user, values='rating', index='user_id', columns='item_id', fill_value=0)

    # Calcular la similaridad del coseno entre el usuario dado y todos los demás usuarios
    user_similarity = cosine_similarity(user_reviews_matrix, user_reviews_matrix)

    # Obtener las recomendaciones basadas en la similaridad del coseno
    user_index = user_reviews_matrix.index.get_loc(user_id)
    similar_users_indices = user_similarity[user_index].argsort()[:-6:-1]

    # Obtener las reseñas del usuario dado
    user_reviews_items = reviews_for_user['user_id'].unique()

    # Filtrar juegos que el usuario ya ha revisado
    recommended_items = user_reviews_matrix.columns[similar_users_indices].difference(user_reviews_items)[:5]

    return recommended_items.tolist()




#funcion 7 

def user_recommendation_logic(user_id: str):
    try:
        # Cargar el modelo entrenado desde el archivo pickle
        with open('C:\\Users\\Sebastian Diaz G\\OneDrive\\Escritorio\\Proyecto_individual_ML\\Modelo\\model1.pkl', 'rb') as archivo:
            model = pickle.load(archivo)

        # Leer el conjunto de datos de reseñas de usuarios
        user_reviews = pd.read_csv('C:\\Users\\Sebastian Diaz G\\OneDrive\\Escritorio\\Proyecto_individual_ML\\Data\\reviews.csv', usecols=['user_id',  'item_id'])

        # Verificar si el usuario ha realizado alguna reseña
        if not user_reviews['user_id'].eq(user_id).any():
            user_item_max_hours = pd.read_csv('C:\\Users\\Sebastian Diaz G\\OneDrive\\Escritorio\\Proyecto_individual_ML\\Data\\max_hours.csv')
            if not user_item_max_hours['user_id'].eq(user_id).any():
                return 'Ese usuario no ha realizado reseñas.'
            else:
                item = user_item_max_hours.loc[user_item_max_hours['user_id'] == user_id, 'item_id']
                return recommend(int(item.iloc[0]))

        # Leer el conjunto de datos de juegos de Steam
        df_steam = pd.read_csv('C:\\Users\\Sebastian Diaz G\\OneDrive\\Escritorio\\Proyecto_individual_ML\\Data\\steam1_output.csv')
        user_reviews_id = user_reviews[user_reviews['user_id'] != user_id]
        user_game = pd.merge(df_steam[['id', 'app_name']], user_reviews_id, left_on='id', right_on='item_id', how='inner')
        user_rec = user_reviews[user_reviews['user_id'] == user_id]

        # Realizar predicciones y ordenarlas por puntaje descendente
        predictions = pd.DataFrame()
        predictions['app_name'] = user_game['app_name']
        predictions['score'] = user_game['id'].apply(lambda x: model.predict(user_rec.iloc[0]['user_id'], x).est)
        predictions.sort_values(by='score', ascending=False, inplace=True)

        # Eliminar valores nulos y duplicados, luego seleccionar las mejores 5 recomendaciones
        predictions.dropna(inplace=True)
        predictions.drop_duplicates(subset='app_name', inplace=True)
        top_5 = predictions.head(5)

        return {
            'Recomendacion 1': top_5['app_name'].iloc[0],
            'Recomendacion 2': top_5['app_name'].iloc[1],
            'Recomendacion 3': top_5['app_name'].iloc[2],
            'Recomendacion 4': top_5['app_name'].iloc[3],
            'Recomendacion 5': top_5['app_name'].iloc[4]
        }

    except Exception as e:
        raise RuntimeError(f"Error en la aplicación: {str(e)}")