import os
import cv2
import json
import copy
import argparse
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor

from prompt import get_prompt
from llm import infer
from utils.parse import parse_str
from utils.utils import convert_example_list, get_query, get_sg, get_attribute, scale_bbox
from utils.visualize import visualize_seg
from infer_image import infer_image
from CLIS.eval_image import eval_image
from CLIS.eval_layout import rule_eval_offline, avg_score_by_conf


# ------------------------- Inference ------------------------- # 

def infer_description(
    object_list: list,
    model: str = 'qwen-1.5-14b',
    task: str = 'create_dataset',
    example_prefix: str = 'config/',
):
    '''
    Function:
        infer description given object list
    '''
    
    # get example list
    with open(example_prefix + f"{task}.json", 'r') as f:
        ori_example_list = json.load(f)
    
    example_list = convert_example_list(task, ori_example_list)
    example_list = example_list[:-1]
    
    # get query
    ori_q = get_query(task, object_list)
    q = {
        "prompt": json.dumps(ori_q, indent=4),
        "output": ""
    }
    
    # get prompt
    prompt = get_prompt(task, example_list, q)
    
    # get response
    response = infer(model, prompt, h_model=llm, h_tokenizer=llm_tokenizer, max_tokens=1024)
    parsed_response = parse_str(response)
    
    return parsed_response
    

def infer_layout(
    desc: dict,
    model: str = 'qwen-1.5-14b',
    task: str = 'bbox',
    example_prefix: str = 'config/'
):
    '''
    Function:
        infer layout given description
    '''
    
    with open(example_prefix + f"{task}.json", 'r') as f:
        ori_example_list = json.load(f)
    
    example_list = convert_example_list(task, ori_example_list)
    example_list = example_list[:-1]
    
    ori_q = get_query(task, desc)
    q = {
        "prompt": json.dumps(ori_q, indent=4),
        "output": ""
    }
    
    # get prompt
    prompt = get_prompt(task, example_list, q)
    
    # get response
    response = infer(model, prompt, h_model=llm, h_tokenizer=llm_tokenizer, max_tokens=1024)
    parsed_response = parse_str(response)['Layout']
    
    return parsed_response


def gen_image(
    sg: dict,
    num_images: int = 8,
    save_info_dir: str = 'gen_info/',
    save_img_dir: str = 'images/',
    prefix: str = '/mnt/petrelfs/chenyicheng/workspace/code/ACP/'
):
    
    # initialize task
    caption = sg['caption'] # global caption
    annos= []
    for i in sg['layout']:
        i_bbox = i['bbox']
        i_caption = get_attribute(i['object'], sg)
        annos.append({
            "bbox": i_bbox,
            "caption": i_caption,
        })
    
    gen_info = {
        "caption": caption,
        "width": 1.0,
        "height": 1.0,
        "annos": annos
    }
    
    if not os.path.exists(save_info_dir):
        os.makedirs(save_info_dir)
    cnt = len([name for name in os.listdir(save_info_dir) if os.path.isfile(os.path.join(save_info_dir, name))])
    input_json = f"{save_info_dir}{cnt}.json"
    with open(input_json, 'w') as f:
        json.dump(gen_info, f)
    
    # generate images
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    cnt = len([name for name in os.listdir(save_img_dir) if os.path.isdir(os.path.join(save_img_dir, name))])
    output = f"{save_img_dir}{cnt}/"
    
    infer_image(
        output=f"{prefix}{output}",
        num_images=num_images,
        input_json=f"{prefix}{input_json}"
    )
    
    # update
    syn_data = copy.deepcopy(sg)
    syn_data['img_dir'] = output
    
    return syn_data


def seg(
    syn_data: dict,
):
    
    # load model 
    sam = sam_model_registry['vit_h'](checkpoint='sam/sam_vit_h_4b8939.pth')
    sam.cuda()
    sam_mask_predictor = SamPredictor(sam)
    
    img_dir = syn_data['img_dir']
    
    seg_syn_data_list = []
    
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            
            if 'xl' not in file:
                continue
            
            img_path = os.path.join(root, file)
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width = img_bgr.shape[:2]
            
            sam_mask_predictor.set_image(img_rgb)
            
            seg_syn_data = copy.deepcopy(syn_data)
            seg_syn_data['layout'] = []
            
            for i, l in enumerate(syn_data['layout']):
                
                bbox = scale_bbox(l['bbox'], width, height)
                box = np.array(bbox)
                
                # get mask
                masks, scores, logits = sam_mask_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                mask = masks[np.argmax(scores)]
                
                # get contours
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                max_contour = max(contours, key=cv2.contourArea)
                segmentation = max_contour.flatten().tolist()
                
                seg_l = copy.deepcopy(l)
                seg_l['segmentation'] = [segmentation]
                
                # update
                seg_syn_data['layout'].append(seg_l)
                
            seg_syn_data['img_path'] = img_path
            
            seg_syn_data_list.append(seg_syn_data)

    return seg_syn_data_list
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_list", type=str, default="inputs/demo.json", help="input json file of object list")
    
    args = parser.parse_args()
    
    # initialize object list
    with open(args.object_list, 'r') as f:
        object_list = json.load(f)
    
    # initialize model
    vlm_tokenizer = AutoTokenizer.from_pretrained('Qwen-VL-Chat', trust_remote_code=True)
    vlm = AutoModelForCausalLM.from_pretrained("Qwen-VL-Chat", device_map="cuda", trust_remote_code=True)
    vlm.cuda()
    vlm.eval()
    
    with open('config/model_config.json', 'r') as f:
        model_pool = json.load(f)
    
    llm_tokenizer = AutoTokenizer.from_pretrained(model_pool['qwen-1.5-14b'], trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(model_pool['qwen-1.5-14b'], torch_dtype='auto', device_map='cuda', trust_remote_code=True)
    llm.cuda()
    llm.eval()
    
    vis_flag = True
    eval_layout_flag = True
    
    # get description
    desc= infer_description(object_list)
    
    # get layout
    layout = infer_layout(desc)
    
    # combine scene graph
    sg = get_sg(desc, layout)
    
    # generate images
    syn_data = gen_image(sg, num_images=16)
    
    # seg
    seg_syn_data_list = seg(syn_data)
    
    # visualize segmentation
    if vis_flag:
        for seg_syn_data in seg_syn_data_list:
            visualize_seg(seg_syn_data)
    
    # eval
    syn_data, score_list = eval_image(syn_data, vlm=vlm, vlm_tokenizer=vlm_tokenizer, llm=llm, llm_tokenizer=llm_tokenizer)
    print(syn_data)
    
    # save
    data_dir = 'syn_data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cnt = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
    
    save_path = f"{data_dir}{cnt}.json"
    with open(save_path, 'w') as f:
        json.dump([syn_data], f)
    
    if eval_layout_flag:
        score_list, score_size_list, score_dist_list, score_dir_list, conf_list, ret_list = rule_eval_offline(pred_path=save_path, sim_threshold=0.4)
        score, score_size, score_dist, score_dir = avg_score_by_conf(score_list, score_size_list, score_dist_list, score_dir_list, conf_list)
        print(score)
    