import pandas as pd
from transformers import pipeline

try:
    df = pd.read_csv("./data/employees1.csv")
    df = df.astype(str)
    table_data = df.to_dict(orient="records")
except Exception as e:
    print(f"Error loading CSV: {e} ")
    exit()

# TAPAS model
try:
    tqa = pipeline("table-question-answering", model="google/tapas-large-finetuned-wtq")
except Exception as e:
    print(f"Error loading TAPAS model: {e}")
    exit()

while True:
    question = input("What would you like to ask?  or Type 'exit' to quit\n").strip()
    if question.lower() in ['exit', 'quit']:
        print("Exiting. Goodbye!")
        break

    try:
        answer = tqa(table=table_data, query=question)
        print(f"Answer: {answer['answer']} \n")
    except Exception as e:
        print(f"Couldn't answer that. Error: {e} \n")
