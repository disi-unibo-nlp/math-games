from openai import OpenAI
import torch
import json
import os
import base64
import re
import logging
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict 
from datetime import datetime


from google import genai
from google.genai import types



#codellama/CodeLlama-34b-Instruct-hf #bigcode/starcoder2-15b-instruct-v0.1 #mistralai/Codestral-22B-v0.1 #deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct #TheBloke/CodeLlama-70B-Instruct-AWQ #casperhansen/llama-3-70b-instruct-awq #hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gemini-2.5-flash", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="disi-unibo-nlp/MathGames",  metadata={"help": "dataset HF directory"})
    max_samples: Optional[int] = field(default=4, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=1, metadata={"help": "Number of prompts to sample for each question"})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default='cot', metadata={"help": "Inference mode: CoT or TIR", "choices":["cot", "tir"]})
    text_only: Optional[bool] = field(default=False, metadata={"help": 'whether to consider only textual question without images.'})
    img_only: Optional[bool] = field(default=True, metadata={"help": 'whether to consider only textual question combined with images.'})

    def __post_init__(self):
        if self.text_only and self.img_only:
            raise ValueError("The options 'text_only' and 'img_only' cannot both be True at the same time.")
        

if __name__ == "__main__":
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version':'v1alpha'})
    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/batch_api/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/batch_api/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    MODEL_NAME =  args.model_name 

    if args.text_only: # to use to ignore images from data
        dataset = load_dataset(args.dataset_name, split="textual")
    
    if args.img_only: # to use to ignore images from data
        dataset = load_dataset(args.dataset_name, split="multimodal")
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, args.max_samples))
    
    if args.start_idx > 0 and args.max_samples < 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, len(dataset)))

    logger.info(f"First sample:\n{dataset[0]}")
    #######################################
    #### 1. Preparing Your Batch File #####
    #######################################
    
    total_promtps = 0
    json_file_path = f'out/batch_api/{output_dir}/input_batch.json'
    request_data = []
    for i, item in enumerate(tqdm(dataset)): 

        prompt = item['question']
        id = item['id']
        
        if args.text_only:
            for k in range(args.n_sampling):
                batch_request = {"key": f"request-{id}-{k}-image", "request": {"contents": [{"parts": [{"text": f"{prompt}"}]}], "generation_config": {"temperature": 0.7}}}
                request_data.append(batch_request)

        if args.img_only:            

            image_path = f"jpg_images/image_{id}.jpg"
            IMAGE_MIME_TYPE = "image/jpg"

            image_file = client.files.upload(
                file=image_path,
            )

            for k in range(args.n_sampling):
                batch_request = {
                    "key": f"request-{id}-{k}-image",
                    "request": {
                        "contents": [{
                            "parts": [
                                {"text": prompt + "\n\nEnclose the final answer in \\boxed{}."},
                                {"file_data": {"file_uri": image_file.uri, "mime_type": image_file.mime_type}}
                            ]
                        }]
                    }
                }

                request_data.append(batch_request)


    print(f"\nCreating JSONL file: {json_file_path}")
    print("len(request_data): ", len(request_data))
    with open(json_file_path, 'w') as f:
        for req in request_data:
            f.write(json.dumps(req) + '\n')
    
        
    logger.info(f"Uploading JSONL file: {json_file_path}")
    batch_input_file = client.files.upload(
        file=json_file_path
        )
    logger.info(f"Uploaded JSONL file: {batch_input_file.name}")

    logger.info("\nCreating batch job...")
    # now time string
    
    # Format the date and time as a string
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    batch_job_from_file = client.batches.create(
        model=MODEL_NAME,
        src=batch_input_file.name,
        config=types.UploadFileConfig(display_name=f'requests-{time_str}')
    )
    logger.info(f"Created batch job from file: {batch_job_from_file.name}")
    logger.info("You can now monitor the job status using its name.")
        
    logger.info(f"UNIQUE PROMPTS: {total_promtps / args.n_sampling}")
    logger.info(f"TOTAL PROMPTS: {total_promtps}")
    