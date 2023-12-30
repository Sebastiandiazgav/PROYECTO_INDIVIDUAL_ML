# Proyecto Individual de Machine Learning




# ENLACES A CONSULTAR INFORMACION

[Notbooks](https://drive.google.com/drive/folders/1HPIinSXMW4Y-Wswt48FMcGjG4P3RjuWC?usp=sharing)

[Data-Seta](https://drive.google.com/drive/folders/10Db0d2GEuoYHdPqy6sF_AMtNnj3yBXQJ?usp=sharing)

[Modelo](https://drive.google.com/drive/folders/1KpjdO5KXqMT9--D6srKSIxW3kMbZh2Wm?usp=sharing)







## Contexto y Rol a Desarrollar

El objetivo de este proyecto es desarrollar un modelo de recomendación con métricas sólidas. El rol para el cual me voy a desempeñar implica trabajar en Steam, una plataforma multinacional de videojuegos líder en la industria. Steam proporcionará toda la información necesaria para ejecutar nuestro trabajo y crear un sistema de recomendación óptimo para sus usuarios.


#Pasos

## Paso 1: Trabajo ETL (Extract, Transform, Load)

### steam_games

- Extraemos la información.
- Revisamos la información existente para entender la estructura de las columnas.
- Identificamos y tratamos datos duplicados, nulos y faltantes.
- Creamos variables ficticias para géneros.
- Eliminamos columnas que no son necesarias.
- Realizamos una verificación exhaustiva para asegurar datos limpios.
- Verificamos los tipos de datos que cumpla con lo necesario
- Exportamos el conjunto de datos limpio, listo para su uso.


### user_reviews

- Extraigo la data para su inspección.
- Aplico `explode` en la columna `reviews`, duplicando las filas por cada elemento en la lista. Cada fila contendrá una única reseña.
- Reviso la composición de las columnas de la tabla.
- Creamos una columna de análisis de sentimientos.
- Elimino valores nulos.
- Exporto mi nuevo conjunto de datos para trabajar.


### User_item

- Leemos el archivo.
- Revisamos la composición de mi conjunto de datos.
- Verificamos valores nulos y los eliminamos.
- Eliminamos las columnas `user_url`, `playtime_2weeks`, `steam_id`.
- Exportamos la información para su posterior trabajo.



## Paso 2: Análisis Exploratorio de Datos (EDA)

### steam_games

- Visualizamos los datos que tenemos, incluyendo sus tipos, valores nulos y columnas.
- Utilizamos el método `.describe()` para visualizar estadísticamente nuestro conjunto de datos.
- Validamos el conjunto de datos, por ejemplo, analizando el rango de precios, los diez juegos más jugados, las categorías predominantes y la distribución de géneros a lo largo del año.
- Verificamos los juegos gratuitos y cuándo tuvieron mayores horas jugadas.
- Validamos que no haya valores nulos y que los datos estén limpios.

### user_reviews

- Verificamos la información del conjunto de datos, incluyendo tipos de datos y la composición de las columnas.
- Analizamos opiniones negativas y positivas de las recomendaciones de los juegos.
- Determinamos la proporción de recomendaciones positivas y negativas.
- Validamos la distribución de sentimientos en las revisiones, clasificándolas como no recomendadas, recomendadas o con opiniones neutras.

### user_items

- Validamos la información presente en el conjunto de datos.
- Revisamos la relación entre el tiempo de juego y el número de items.
- Identificamos los diez juegos más jugados.
- Exploramos y abordamos la presencia de outliers.
- Generamos un merge entre `steam_games` considerando el ID y el año, y `user_items` considerando `item_id` y `playtime_forever`.
- Validamos los juegos con mayores horas jugadas.


## Paso 3: Data API

- Aumentar el rendimiento de la API.
- Realizo un merge entre `app_name` de mi conjunto de datos `steam1_output` y `user_items`, utilizando las columnas `user_id`, `items_count`, `item_name` mediante las columnas `app_name` e `item_name`.
- Con la información anterior, creo el porcentaje de recomendación.
- Genero un DataFrame con los géneros y agrupo la información.
- Agrupo datos por año y calculo la suma de `playTime_forever`.
- Elimino filas con valores faltantes.
- Genero un DataFrame con la nueva información para géneros.
- Realizo un nuevo merge entre `user_items` y `user_reviews`.
- Agrupo los usuarios mediante el ID con el tiempo en minutos jugados.
- Exporto el DataFrame.



## Paso 4: Modelo Recomendacion

## Paso 4: Modelo de Recomendación

- Genero un modelo utilizando las librerías Surprise y scikit-learn, utilizando SVD y KNN.
- En mi conjunto de datos `user_reviews1`, extraigo la información del año de la columna llamada `posted` y genero una nueva columna solo con el año.
- Creo la columna `rating` que contendrá valoraciones de 0 a 4, siendo 0 la más baja y 4 la más alta.
- Entreno el modelo con SVD utilizando hiperparámetros específicos.
- Realizo pruebas de entrenamiento y exporto el modelo con el nombre `model1.pkl`.
- Genero otro modelo KNN, especificando el rango de clasificación.
- Selecciono un conjunto de datos pequeño del conjunto de datos para examinar su rendimiento.
- Entreno el modelo con el método `.fit()`.
- Extraigo el modelo con el nombre `model_knn.pkl`.



## Paso 5 Crear Endpoints

### def PlayTimeGenre(genero: str):
    Obtén el tiempo de juego para un género específico.

### def UserForGenre(genero: str):
    
    Devuelve el usuario que ha acumulado más horas en un juego de un género específico, junto con las horas totales acumuladas por año desde el lanzamiento del juego.


### def users_recommend(year: int):
    
    Obtiene las principales recomendaciones de juegos para un año específico.



### def get_users_not_recommend(year: int):
    
    Obtiene las principales no recomendaciones de juegos para un año específico.

    


### def sentiment_analysis(year: int):
    
    Realiza el análisis de sentimiento de las reseñas para un año específico.

   

### def recommend_item_endpoint(user_id: str):
    
    Esta función recomienda 5 usuarios dado un ítem específico.

    

### def recommend_user_games(user_id: str):
    
    Recomienda los 5 mejores juegos para un usuario específico.



