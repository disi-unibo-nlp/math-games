import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
import json
from openai import OpenAI
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List
import time
import base64
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for calculating category-wise accuracy from a dataset."
    )

    parser.add_argument(
        "--jsonl_file_path",
        type=str,
        default="",
        help="Path to the JSONL file containing model data."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Name of the dataset on Hugging Face."
    )

    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-2024-08-06", # other good judges: "gpt-4o-mini-2024-07-18",#"gemini-2.0-flash",#"gpt-4o-2024-08-06",
        help="Name of the dataset on Hugging Face."
    )

    parser.add_argument(
        "--ids_to_modify",
        type=int,
        nargs='+',
        default=[],
        help="List of IDs to modify."
    )

    return parser.parse_args()

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def make_completion(question, gold, final_answer, judge_model, id=None):
    system_prompt = """Given the primary question, compare the gold answer with the student's final answer to determine if they are equivalent. First, provide a concise rational, without trying to redo the problem, then respond with 'yes' or 'no' in the exact format below:

### Rationale: [your rationale]
### Answer: [yes/no]"""
    try:
        # Create a chat completion using the question and context
        if "gemini" in judge_model:
            response = client.chat.completions.create(
                model=judge_model, #gpt-4o-mini-2024-07-18 "gpt-4o-2024-08-06"
                messages = [
                    {"role": "system","content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\nGold answer: {gold}\nFinal Answer: {final_answer}"},
                ],
                temperature=0,
                max_tokens=10000,
                top_p=1,
            )
        else:
            if MODE != "vision":
                response = client.chat.completions.create(
                    model=judge_model, #gpt-4o-mini-2024-07-18 "gpt-4o-2024-08-06"
                    messages = [
                        {"role": "system","content": system_prompt},
                        {"role": "user", "content": f"Question: {question}\nGold answer: {gold}\nFinal Answer: {final_answer}"},
                    ],
                    temperature=0,
                    max_tokens=10000,
                    top_p=1,
                    seed=42
                )
            else:
                image_path = f"jpg_images/image_{id}.jpg"
                base64_image = encode_image(image_path)
                response = client.chat.completions.create(
                    model=judge_model, #gpt-4o-mini-2024-07-18 "gpt-4o-2024-08-06"
                    messages = [
                        {"role": "system","content": system_prompt.replace("Given the primary question", "Given the primary question and the image")},
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Question: {question}\nGold answer: {gold}\nFinal Answer: {final_answer}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ]
                        },
                    ],
                    temperature=0,
                    max_tokens=10000,
                    top_p=1,
                    seed=42
                )


        return response
    except Exception as e:
        print(e)
        return ""

def main(jsonl_file_path, dataset_name, ids_to_modify, csv_file_path, judge_model):
    dataset = load_dataset(dataset_name, split="train")
    df_dataset = pd.DataFrame(dataset)
    
    data = []
    with open(jsonl_file_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
            
    results = []

    for item in data:
        # Estrai il campo 'final_answer'
        id = item['id']
        final_answer = item['final_answer']
        #print(f"ID: {id}, Final Answer: {final_answer}")
        gold_answer = item['gold_answer']

        if is_float(final_answer):
            if final_answer in gold_answer:
                answer = 'yes'
                results.append({
                    "id":id,
                    "gold_answer": gold_answer,
                    "final_answer": final_answer,
                    "model_response": answer
                })
            else:
                answer = 'no'
                results.append({
                    "id":id,
                    "gold_answer": gold_answer,
                    "final_answer": final_answer,
                    "model_response": answer
                })
        elif final_answer == "":
            answer = 'no'
            results.append({
                "id":id,
                "gold_answer": gold_answer,
                "final_answer": final_answer,
                "model_response": answer
            })
        else:
            answer = 'llm'
            results.append({
                "id":id,
                "gold_answer": gold_answer,
                "final_answer": final_answer,
                "model_response": answer
            })
            
    # Crea un DataFrame dai risultati
    df_result = pd.DataFrame(results)
    
    # Iterate through the DataFrame and modify the 'model_response' for specified IDs
    for index, row in df_result.iterrows():
        if row['id'] in ids_to_modify:
            df_result.loc[index, 'model_response'] = 'no'
            
    df_result_2 = df_result[df_result['model_response'] == 'llm']
    df_result_2 = df_result_2.merge(df_dataset[['id', 'question']], on='id', how='left')
    
    
    NUM_ITEMS = len(df_result_2)
    
    k = -1
    evaluations = []
    for _, row in tqdm(df_result_2.iterrows(), desc="Processing items", total=NUM_ITEMS):
        k += 1
        if k < NUM_ITEMS:
            
            gold_answer = row['gold_answer']
            final_answer = row['final_answer']
            question = row['question']
            id_problem = row['id']
            #print(f"ID: {id_problem}, Final Answer: {final_answer}")

            
            response = make_completion(question=question, gold=gold_answer, final_answer=final_answer, judge_model=judge_model, id=id_problem)
            completion = response.choices[0].message.content.strip()
            
            model = response.model
            #usage = dict(response.usage)

            completion_text = completion.lower()

            if "### answer:" in completion_text:
                decision = completion.split("### Answer:")[1].replace("[", "").replace("]", "").strip()
                evaluations.append(decision)
            else:
                print(f"'### answer:' not found in:\n{completion_text}")
                evaluations.append("MISSING_ANSWER")
            out_path_eval = "/".join(jsonl_file_path.split('/')[:-1])
            with open(f'{out_path_eval}/eval_{judge_model}.jsonl', 'a') as f:
                result = {
                    "model": model,
                    "id": row['id'],
                    "decision": decision,
                    "completion": completion,
                    "gold_answer": gold_answer,
                    "final_answer": final_answer,
                    #"usage": usage if completion else {},
                }
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            if "gemini" in judge_model:
                time.sleep(5)
        else:
            break
    
    assert len(evaluations) == len(df_result_2)
    df_result_2['model_response'] = evaluations
    
    df_final = pd.concat([df_result_2, df_result[df_result['model_response'] != 'llm']], ignore_index=True)
    df_final = df_final.drop(columns=['question'])
    df_final.to_csv(csv_file_path, index=False)



if __name__ == "__main__":

    load_dotenv()
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # parse input args
    
    args = parse_args()
    if "gemini" in args.judge_model:
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        client = OpenAI(
            api_key=OPENAI_KEY
        )

    MODE = args.jsonl_file_path.split('/')[4]
    print(f"MODE: {MODE}")
    csv_file_path = args.jsonl_file_path.replace('.jsonl', f'_eval_{args.judge_model}.csv')

    main(args.jsonl_file_path, args.dataset_name, args.ids_to_modify, csv_file_path, args.judge_model)