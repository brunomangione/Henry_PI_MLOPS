from fastapi import FastAPI
import pandas as pd

app = FastAPI()


# PRIMER FUNCION 

#Traemos el archivo parquet que creamos   
data = pd.read_parquet('genres.parquet')   

# Traemos la funcion creada 
def playTimeGenre(genres):
    
    filtro = data[data['genres'] == genres] 
    if filtro.empty:
        return f"El genero {genres} no se encuentra en los registros" 
    else:
        anio_max_horas = filtro.loc[filtro['hours_game'].idxmax(), 'year']
        
    return f"El Año de lanzamiento con mas horas jugadas para el genero {genres}: {anio_max_horas}"

# Creamos la funcion para Fast Api    
@app.get("/anio_mas_horas/{genre}")
async def anio_con_mas_horas_para_genero(genero: str):
    resultado = playTimeGenre(genero)
    return resultado
    


# SEGUNDA FUNCION 
    
# Traemos la funcion creada
def userForGenre (genres):
    
    filtro = data[data['genres'] == genres]
    if filtro.empty:
        return f"el genero {genres} no se encuentra en los registros"
    else:
        usuario_max_horas = filtro.loc[filtro['hours_game'].idxmax()]['user_id']
        acumulacion_horas_anio = filtro.groupby('year')['hours_game'].sum().reset_index()
    
    result = {
        "Usuario con más horas jugadas para Género " + genres: usuario_max_horas,
        "Horas jugadas": [{"Año":row['year'], "Horas": row['hours_game']} for index, row in acumulacion_horas_anio.iterrows()]
    }
    
    return result

# Creamos la funcion para FastApi
@app.get("/User_for_genre/{genre}")
async def user_for_genre(genero: str):
    resultado = userForGenre(genero)
    return resultado
    
