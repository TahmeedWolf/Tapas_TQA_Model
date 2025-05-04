import pandas as pd
from transformers import pipeline
from deep_translator import GoogleTranslator
import json


try:
    df = pd.read_csv("./data/employees1.csv")
    df = df.astype(str)
    table_data = df.to_dict(orient="records")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# TAPAS model
try:
    tqa = pipeline("table-question-answering", model="google/tapas-large-finetuned-wtq")
except Exception as e:
    print(f"Error loading TAPAS model: {e}")
    exit()

def translate_to_bengali(text):
    try:
        return GoogleTranslator(source='auto', target='bn').translate(text)
    
    except Exception as e:
        return f"(Translation failed: {e})"
    
def translate_numbers_to_bengali(text):
    try:
        eng_to_bn_numbers = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
        return text.translate(eng_to_bn_numbers)
    except Exception as e:
        return f"(Number conversion failed: {e})"

try:
    with open("bengali_map.json", "r", encoding="utf-8") as f:
        job_title_map = json.load(f)
except Exception as e:
    print(f"Error loading bengali_map.json: {e}")
    job_title_map = {}
    

while True:
    question = input("What would you like to ask? (Type 'exit' to quit)\n").strip()
    if question.lower() in ['exit', 'quit']:
        print("Exiting. Goodbye!")
        break

    try:
        answer = tqa(table=table_data, query=question)
        english_answer = answer['answer']
        bengali_answer = translate_to_bengali(english_answer)
        bengali_answer = translate_numbers_to_bengali(bengali_answer)

        if translate_numbers_to_bengali(bengali_answer.lower()) == (english_answer.lower()):
            bengali_answer = job_title_map.get(english_answer, "(No translation available)")


        print(f"English Answer: {english_answer}\n")
        print(f"Bengali Answer: {bengali_answer}\n")
        
        # spaced_bengali = ' '.join(bengali_answer)
        # print(f"\nBengali Answer:\n  → {spaced_bengali}\n")
        
    except Exception as e:
        print(f"Couldn't answer that. Error: {e} \n")
