from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
import numpy as np
import json
import os
import dtale
from datetime import datetime
from typing import List, Optional
import h2o

# Import your converted scripts
import feature_selection_revised
import h20
import Step_1

async def read_csv(file: UploadFile) -> pd.DataFrame:
    """
    Function to read CSV file and return Pandas DataFrame.
    """
    content = await file.read()
    decoded_content = content.decode('utf-8')
    df = pd.read_csv(StringIO(decoded_content))
    return df

app = FastAPI()

h2o.init()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app"}

@app.post("/eda")
async def upload_file_eda(file: UploadFile = File(...)):
    # Stage 1: Read CSV file and convert to Pandas DataFrame
    df = await read_csv(file)
    
    # Call the main function or relevant function from automated_eda.py
    json_result_eda = feature_selection_revised.main(df)
    return json_result_eda

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Stage 1: Read CSV file and convert to Pandas DataFrame
    df = await read_csv(file)
    
    # Preprocess The Data
    df = Step_1.main(df)
    
    # Call the main function or relevant function from h2o_automl.py
    json_result = h20.main(df)
    
    #converting list of list str to df
    #df_pandas = pd.DataFrame(df, columns=["model_id", "auc", "logloss","aucpr","mean_per_class_error","rmse","mse"])
    
    # Convert H2OFrame to Pandas DataFrame
    #df_pandas = df.as_data_frame()

    # Convert Pandas DataFrame to list of dictionaries (orient='records')
    #result = df_pandas.to_dict(orient='records')
    
    # Convert dictionary to JSON string
    #json_result = json.dumps(result)
    
    # Convert H2OFrame to JSON string
    #json_result = df.as_data_frame().to_json(orient='records')
    
    
    return {"result": json_result}