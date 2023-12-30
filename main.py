from funciones import *
from fastapi import FastAPI, HTTPException
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader, KNNBasic, SVD
from typing import List
import pickle
from math import sqrt


app = FastAPI()

@app.get('/', tags=['Home'])
def Home():
    return {'API de consultas para un modelo de Machine Learning para la empresa Steam '}



#enpoint para PlayTimeGenre
@app.get('/playtime_genre', tags=['Play Time Genre'])
def PlayTimeGenre(genero: str):
    """
    Obtén el tiempo de juego para un género específico.

    Parameters:
    - `genero` (str): El género para el cual quieres obtener el tiempo de juego.

    Returns:
    - `dict`: Un diccionario con el resultado de la operación.
    """
    try:
        return PlayTimeGenre_func(genero)
    except Exception as e:
        return {"Error": str(e)}






#Enpoint UserForGenre
@app.get('/genero', tags=['User For Genre'])
def UserForGenre(genero: str):
    """
    Devuelve el usuario que ha acumulado más horas en un juego de un género específico, junto con las horas totales acumuladas por año desde el lanzamiento del juego.
    genero: str Genero del juego

    Ingresar en el apartado  genre_ y luego el genero ejemplo genre_Action
    """
    try:
        return UserForGenre_func(genero)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la aplicación: {str(e)}")


with open("C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Modelo/model1.pkl", "rb") as f:
    model_svd = pickle.load(f)

# Cargar el modelo KNN entrenado
with open("C:/Users/Sebastian Diaz G/OneDrive/Escritorio/Proyecto_individual_ML/Modelo/model_knn.pkl", "rb") as f:
    model_knn = pickle.load(f)



# Endpoint para UsersRecommend
@app.get("/UsersRecommend/{year}",tags=['Users Recommend'])
def users_recommend(year: int):
    """
    Obtiene las principales recomendaciones de juegos para un año específico.

    Parameters:
    - `year` (int): El año para el cual se desean obtener las recomendaciones.

    Returns:
    - `List[Dict[str, Any]]`: Una lista de diccionarios que contienen las principales recomendaciones. Cada diccionario tiene la forma: {"Puesto N": game_id}
    """
    try:
        result = UsersRecommend_func(year)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la aplicación: {str(e)}")


# Endpoint para UsersNotRecommend
@app.get("/UsersNotRecommend/{year}", tags=['Users Not Recommend'])
def get_users_not_recommend(year: int):
    """
    Obtiene las principales no recomendaciones de juegos para un año específico.

    Parameters:
    - `year` (int): El año para el cual se desean obtener las no recomendaciones.

    Returns:
    - `List[Dict[str, Any]]`: Una lista de diccionarios que contienen las principales no recomendaciones. Cada diccionario tiene la forma: {"Puesto N": game_id}
    """
    try:
        # Lógica para obtener las no recomendaciones usando el modelo SVD
        result = UsersNotRecommend_logic(year)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la aplicación: {str(e)}")


# Endpoint para sentiment_analysis
@app.get("/sentiment_analysis/{year}", tags=['Sentiment Analysis'])
def sentiment_analysis(year: int):
    """
    Realiza el análisis de sentimiento de las reseñas para un año específico.

    Parameters:
    - `year` (int): El año para el cual se desea realizar el análisis de sentimiento.

    Returns:
    - `Dict[str, Any]`: Un diccionario que contiene la cantidad de reseñas con análisis de sentimiento negativo, neutral y positivo. 
    La estructura del diccionario es: {"Negative": neg_count, "Neutral": neutral_count, "Positive": pos_count}
    """
    try:
        # Lógica para realizar el análisis de sentimiento usando el modelo SVD
        result = sentiment_analysis_func(year)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la aplicación: {str(e)}")



#Endpoint recommend_item
@app.get('/recommend_item/{user_id}', tags=['Recommend Item']) 
def recommend_item_endpoint(user_id: str):
    """
    Esta función recomienda 5 usuarios dado un ítem específico.

    Params:
    item_id: str - id del ítem para el cual se quieren recomendar usuarios.

    Returns:
    Lista con los IDs de los 5 usuarios recomendados.
    """
    try:
        # Reemplaza la siguiente línea con la llamada a la función correcta
        result = recommend(user_id)
        return {"Recomendaciones": result}
    except HTTPException as e:
        return {"Error": f"HTTP Exception - {e.detail}"}
    except Exception as e:
        return {"Error": str(e)}
    



#Enpoint para Recommend_user_games
@app.get('/recommend_user_games/{user}', tags=['Recommend User Games'])
def recommend_user_games(user_id: str):
    """
    Recomienda los 5 mejores juegos para un usuario específico.

    Parameters:
    - `user_id` (str): El ID del usuario para el cual se desea realizar la recomendación.

    Returns:
    - `Dict[str, str]`: Un diccionario con los nombres de los 5 juegos recomendados.
    """
    try:
        return user_recommendation_logic(user_id)
    except Exception as e:
        return {"Error": str(e)}