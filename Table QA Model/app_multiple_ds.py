import pandas as pd
from transformers import pipeline
import glob
import os

# Load all CSVs into a dictionary of dataframes
tables = {}
try:
    csv_files = glob.glob(os.path.join('./data/', '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file).astype(str)
        table_name = os.path.basename(csv_file)
        tables[table_name] = df.to_dict(orient="records")
except Exception as e:
    print(f"Error loading CSVs: {e}")
    exit()

# TAPAS model
try:
    tqa = pipeline("table-question-answering", model="google/tapas-large-finetuned-wtq")
except Exception as e:
    print(f"Error loading TAPAS model: {e}")
    exit()

while True:
    print("\nAvailable tables:")
    for i, name in enumerate(tables.keys(), 1):
        print(f"{i}. {name}")
    
    table_choice = input("Choose a table by name (or type 'exit' to quit):\n").strip()
    if table_choice.lower() in ['exit', 'quit']:
        print("Exiting. Goodbye!")
        break

    if table_choice not in tables:
        print("Invalid table name. Try again.\n")
        continue

    question = input("Ask your question:\n").strip()
    try:
        answer = tqa(table=tables[table_choice], query=question)
        print(f"Answer: {answer['answer']} \n")
    except Exception as e:
        print(f"Couldn't answer that. Error: {e} \n")
