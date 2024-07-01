import os
import numpy as np

from prompt import get_prompt
from llm import infer
from utils.parse import parse_score
from utils.utils import create_dir, get_attribute, scale_bbox

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def vlm_caption(
    img_path: str,
    task: str = 'vlm_global_describe',
    bbox: list = [],
    vlm: AutoModelForCausalLM = None,
    vlm_tokenizer: AutoTokenizer = None
):
    
    prompt = get_prompt(task=task)
    if 'local' in task:
        # img = Image.open(img_path).convert("RGB")
        width, height = 1000, 1000
        xmin = int(bbox[0] * width) - 1
        ymin = int(bbox[1] * height) - 1
        xmax = int((bbox[0] + bbox[2]) * width) - 1
        ymax = int((bbox[1] + bbox[3]) * height) - 1
        prompt.format(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    
    query = vlm_tokenizer.from_list_format([
        {"image": img_path},
        {"text": prompt},
    ])
    
    response, history = vlm.chat(vlm_tokenizer, query=query, history=None)
    
    return response


def llm_align(
    text: str,
    pred: str,
    model: str = 'qwen-1.5-14b',
    llm: AutoModelForCausalLM = None,
    llm_tokenizer: AutoTokenizer = None
):
    '''
    Function:
        use llm to score
    '''
    
    # init_llm_model()
    
    prompt = get_prompt(task='llm_align').format(answer=text, pred=pred)
    response = infer(model=model, prompt=prompt, h_model=llm, h_tokenizer=llm_tokenizer)
    score = parse_score(response)
    
    return score


def clis_img(
    img_path: str, 
    text: str, 
    local_list: list,
    crop_path: str = 'crop/0.png',
    local_weights: list = [0.8, 0.2],
    weights: list = [0.5, 0.5],
    vlm: AutoModelForCausalLM = None,
    vlm_tokenizer: AutoTokenizer = None,
    llm: AutoModelForCausalLM = None,
    llm_tokenizer: AutoTokenizer = None
):
    
    global_pred = vlm_caption(img_path, task='vlm_global_describe', vlm=vlm, vlm_tokenizer=vlm_tokenizer)
    global_score = llm_align(text, global_pred, llm=llm, llm_tokenizer=llm_tokenizer)
    if global_score == -1:
        return -1, -1, []
    
    local_score_list = []
    for local_item in local_list:
        
        try:
            local_pred = vlm_caption(img_path, task='vlm_local_describe', bbox=local_item['bbox'], vlm=vlm, vlm_tokenizer=vlm_tokenizer)
            local_score = llm_align(local_item['text'], local_pred, llm=llm, llm_tokenizer=llm_tokenizer)
            if local_score == -1:
                return -1, -1, []
        except:
            local_score_list.append(0)
            continue
        
        # crop score
        with Image.open(img_path) as img:
            
            try:
                crop_img = img.crop(scale_bbox(local_item['bbox']))
                create_dir(crop_path)
                crop_img.save(crop_path)
            except:
                local_score_list.append(0)
                continue

            local_crop_pred = vlm_caption(crop_path, task='vlm_global_describe', vlm=vlm, vlm_tokenizer=vlm_tokenizer)
            local_crop_score = llm_align(local_item['text'], local_crop_pred, llm=llm, llm_tokenizer=llm_tokenizer)
            if local_crop_score == -1:
                return -1, -1, []
        
        local_score_list.append(local_score * local_weights[0] + local_crop_score * local_weights[1])
    
    score = global_score * weights[0] + np.mean(local_score_list) * weights[1]
    
    return score, global_score, local_score_list
    

def eval_image(
    syn_data: dict,
    vlm: AutoModelForCausalLM = None,
    vlm_tokenizer: AutoTokenizer = None,
    llm: AutoModelForCausalLM = None,
    llm_tokenizer: AutoTokenizer = None
):
    
    # initialize task
    suffix = 'gc5-seed0-alpha0.8/'
    img_dir = f"{syn_data['img_dir']}{suffix}"
    
    score_list = []
    
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            
            if 'xl' not in file:
                continue
            
            # get image
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert("RGB")
            
            text = syn_data['caption']
            
            local_list = []
            for i in syn_data['layout']:
                if type(i['bbox']) == list and len(i['bbox']) == 4:
                    local_list.append({
                        "bbox": i['bbox'],
                        "text": get_attribute(i['object'], syn_data)
                    })
            
            # calculate score
            score, global_score, local_score_list = clis_img(img_path, text, local_list, crop_path=f"crop/0.png", vlm=vlm, vlm_tokenizer=vlm_tokenizer, llm=llm, llm_tokenizer=llm_tokenizer)
            
            score_list.append({
                "score": score,
                "file_name": img_path,
                "img_path": img_path,
                "global_score": global_score,
                "local_score_list": local_score_list,
            })
    
    if len(score_list) > 0:
        
        # update syn_data
        max_score_item = max(score_list, key=lambda x: x['score'])
        syn_data['file_name'] = max_score_item['file_name']
        syn_data['img_path'] = max_score_item['img_path']
        syn_data['score'] = max_score_item['score']
        syn_data['global_score'] = max_score_item['global_score']
        syn_data['local_score_list'] = max_score_item['local_score_list']
        
    return syn_data, score_list
