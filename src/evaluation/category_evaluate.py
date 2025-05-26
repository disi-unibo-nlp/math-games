import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
from dataclasses import dataclass, field
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for calculating category-wise accuracy from a dataset."
    )

    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        help="Path to the CSV file containing model data."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Name of the dataset on Hugging Face."
    )

    return parser.parse_args()

def main(file_path, dataset_name):
    try:
        df_model = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Il file '{file_path}' non è stato trovato.")

    dataset = load_dataset(dataset_name, split="train")
    print(dataset)
    df_hf = pd.DataFrame(dataset)

    merged_df = pd.merge(df_model, df_hf[['id', 'difficulty', 'category', 'subject', 'year', 'type']], on='id', how='left')
    
    #os.makedirs("./out", exist_ok=True)
    model_name = file_path.split('/')[2] 
    parts = file_path.split('/')
    
    out_path = "/".join(file_path.split('/')[:-1])
    
    # if not os.path.exists(file_path):
    #     with open(file_path, 'w') as f:
    #         f.write("### Models Category accuracy file ###\n")

    with open(out_path + "/category_accuracy.txt", 'w') as f:
        if parts[4] == 'tir' or parts[4] == 'cot' or parts[4] == 'vision':
            f.write(f"\n\nModel: {model_name}\nMode: {parts[3]}, {parts[4]}\n")
        else:
            f.write(f"\n\nModel: {model_name}\nMode: {parts[3]}\n")
        
        yes_count = merged_df[merged_df['model_response'] == 'yes'].shape[0]
        global_accuracy = yes_count / merged_df.shape[0]
        accuracy_str = f"Global Accuracy: {global_accuracy:.2%}"   
        f.write(accuracy_str + '\n')

        for year in [1996, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            year_df = merged_df[merged_df['year'] == year]
            shape = year_df.shape[0]
            yes_count = year_df[year_df['model_response'] == 'yes'].shape[0]
            
            accuracy = yes_count / shape
            accuracy_str = f"({year}, {accuracy * 100:.1f})"
            
            f.write(accuracy_str + ' ')
        f.write('\n---\n')
        f.write('\n')


        
        easy_df= merged_df[merged_df['difficulty'] == 'easy']
        shape = easy_df.shape[0]
        yes_count = easy_df[easy_df['model_response'] == 'yes'].shape[0]
        
        accuracy = yes_count / shape
        accuracy_str = f"Easy Difficulty Accuracy: {accuracy:.2%}"
        
        f.write(accuracy_str + '\n')
        
        medium_df= merged_df[merged_df['difficulty'] == 'medium']
        shape = medium_df.shape[0]
        yes_count = medium_df[medium_df['model_response'] == 'yes'].shape[0]
        
        accuracy = yes_count / shape
        accuracy_str = f"Medium Difficulty Accuracy: {accuracy:.2%}"
        
        f.write(accuracy_str + '\n')
        
        hard_df= merged_df[merged_df['difficulty'] == 'hard']
        shape = hard_df.shape[0]
        yes_count = hard_df[hard_df['model_response'] == 'yes'].shape[0]
        
        accuracy = yes_count / shape
        accuracy_str = f"Hard Difficulty Accuracy: {accuracy:.2%}"
        
        f.write(accuracy_str + '\n')
        out_dict = {"model": model_name + "_" + parts[3] + "_" + parts[4]} #"overall": round(global_accuracy * 100, 1)}
        for category in ['CE', 'C1', 'C2', 'L1', 'L2', 'GP', 'HC']:
            category_df = merged_df[merged_df['category'].str.contains(category, na=False)]
            shape = category_df.shape[0]
            yes_count = category_df[category_df['model_response'] == 'yes'].shape[0]
            
            accuracy = yes_count / shape
            accuracy_str = f"{category} Category Accuracy: {accuracy:.2%}"
            out_dict[category] = round(accuracy * 100, 1)   
            f.write(accuracy_str + '\n')

        
        
        for subject in ['Arithmetic', 'Logic', 'Geometry', 'Combinatorics', 'Algebra', 'Pattern Recognition']:
            subject_df = merged_df[merged_df['subject'].str.contains(subject, na=False)]
            shape = subject_df.shape[0]
            yes_count = subject_df[subject_df['model_response'] == 'yes'].shape[0]
            
            accuracy = yes_count / shape
            accuracy_str = f"'{subject}' Accuracy: {accuracy:.2%}"
            
            f.write(accuracy_str + '\n')

        for category in ['CE', 'C1', 'C2', 'L1', 'L2', 'GP', 'HC']:
            for year in [2024]:#[1996, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
                category_df = merged_df[
                    merged_df['category'].str.contains(category, na=False) & 
                    (merged_df['year'] == year)
                ]
                shape = category_df.shape[0]
                if shape == 0:
                    continue  # Skip if no data for this year-category combination
                yes_count = category_df[category_df['model_response'] == 'yes'].shape[0]
                accuracy = yes_count / shape
                accuracy_str = f"{category} {year} Accuracy: {accuracy:.2%}"
                f.write(accuracy_str + '\n')
            f.write('---\n')

        f.write('\n')
        f.write('##### GROUP BY COMPETTION PHASE ##########\n')
        for category in ['CE', 'C1', 'C2', 'L1', 'L2', 'GP', 'HC']:
            for phase in ['autumn games', 'team games', "Rosi's games", 'quarter finals', 'semifinal', 'final', 'international final']:
                category_df = merged_df[
                    merged_df['category'].str.contains(category, na=False) &
                    (merged_df['type'].str.contains(phase, na=False))
                ]
                shape = category_df.shape[0]
                if shape == 0:
                    continue  # Skip if no data for this category-phase pair
                yes_count = category_df[category_df['model_response'] == 'yes'].shape[0]
                accuracy = yes_count / shape
                accuracy_str = f"{category} - '{phase}' Accuracy: {accuracy:.2%}"
                f.write(accuracy_str + '\n')
            f.write('---\n')
        
        f.write('\n')
        f.write('##### GROUP BY CATEGORY and SUBJECT ##########\n')
        for category in ['CE', 'C1', 'C2', 'L1', 'L2', 'GP', 'HC']:
            for subject in ['Arithmetic', 'Logic', 'Geometry', 'Combinatorics', 'Algebra', 'Pattern Recognition']:
                category_df = merged_df[
                    merged_df['category'].str.contains(category, na=False) &
                    (merged_df['subject'].str.contains(subject, na=False))
                ]
                shape = category_df.shape[0]
                if shape == 0:
                    continue  # Skip if no data for this category-subject pair
                yes_count = category_df[category_df['model_response'] == 'yes'].shape[0]
                accuracy = yes_count / shape
                accuracy_str = f"{category} - '{subject}' Accuracy: {accuracy:.2%}"
                f.write(accuracy_str + '\n')
            f.write('---\n')
    
    with open("out/completions/category_accuracy.jsonl", 'a') as f:
        json.dump(out_dict, f, ensure_ascii=False)
        f.write('\n')

if __name__ == "__main__":
    
    load_dotenv()
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    # parse input args
    
    args = parse_args()

    main(args.file_path, args.dataset_name)
