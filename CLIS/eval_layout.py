import os
import json
import math
import argparse
import numpy as np

from tqdm import tqdm
from typing import Callable


def get_pred_layout(pred_item: dict, subject: str, object: str):
    
    s_bbox = -1
    o_bbox = -1
    
    for i in pred_item['layout']:
        if i['object'].split('-')[-1] == subject.split('-')[-1]:
            s_bbox = i['bbox']
        if i['object'].split('-')[-1] == object.split('-')[-1]:
            o_bbox = i['bbox']
        
    return s_bbox, o_bbox


'''
============================== Penalty Functions ==============================
'''

def non_penalty(score: float):
    return score


def linear_penalty(score: float, penalty_threshold: float = 0.1):
    return -(1 + penalty_threshold) * (1 - (score / penalty_threshold)) + penalty_threshold if score < penalty_threshold else score


'''
============================== Tool Functions ==============================
'''

def cal_area(bbox: list):
    '''
    Function:
        caluate the area of bbox
    
    Args:
        bbox: [x, y, w, h]
    '''
    
    return bbox[2] * bbox[3]
    
    
def cal_area_ratio(bbox1: list, bbox2: list):
    '''
    Function:
        caluate the area ratio of bbox1 and bbox2
    '''
    
    return cal_area(bbox1) / cal_area(bbox2)


def cal_iou(bbox1: list, bbox2: list):
    '''
    Function:
        calculate the iou of bbox1 and bbox2
    
    Args:
        bbox: [x, y, w, h]
    '''
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    
    # calculate inter area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # calculate area of bbox
    area1 = cal_area(bbox1)
    area2 = cal_area(bbox2)
    
    # calculate union area
    union_area = area1 + area2 - inter_area
    
    # calculate iou
    iou = inter_area / union_area
    
    return iou


def cal_diagonal(bbox: list):
    '''
    Function:
        calculate the diagonal of bbox
    '''
    
    return math.sqrt(bbox[2] ** 2 + bbox[3] ** 2)


def cal_center(bbox: list):
    '''
    Function:
        calculate the center of bbox
    '''
    
    return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    

def cal_dist(x1: float, y1: float, x2: float, y2: float):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    

def cal_rel_dist(bbox1: list, bbox2: list):
    '''
    Function:
        calculate the relative distance of bbox1 and bbox2
    '''
    
    d1 = cal_diagonal(bbox1)
    d2 = cal_diagonal(bbox2)
    d_avg = (d1 + d2) / 2
    
    c1 = cal_center(bbox1)
    c2 = cal_center(bbox2)
    
    d = cal_dist(c1[0], c1[1], c2[0], c2[1])
    
    d_norm = d / d_avg
    
    return d_norm


def cal_dir(bbox1: list, bbox2: list):
    '''
    Function:
        calculate the direction vector
    '''
    
    c1 = cal_center(bbox1)
    c2 = cal_center(bbox2)
    
    return np.array([c2[0] - c1[0], c2[1] - c1[1]])
    

def cal_sim(A: float, B: float):
    '''
    Function:
        calculate the similarity between A and B
    '''
    
    # judge 
    if A + B < 1e-10:
        return 1.0
    
    # norm
    A_norm = A / (A + B)
    B_norm = B / (A + B)
    
    # calculate relative error
    relative_error = abs(A_norm - B_norm) / max(A_norm, B_norm)
    
    # calculate sim
    sim = 1 - relative_error
    
    return sim


def update_weight(init_weight: float):
    '''
    Function:
        update weight
    '''
    
    if init_weight >= 0.9:
        weight = init_weight
    elif init_weight < 0.9 and init_weight >= 0.8:
        weight = init_weight * init_weight
    elif init_weight < 0.8 and init_weight >= 0.7:
        weight = init_weight * init_weight * init_weight
    elif init_weight < 0.7 and init_weight >= 0.6:
        weight = init_weight * init_weight * init_weight * init_weight
    else:
        weight = init_weight * init_weight * init_weight * init_weight * init_weight
    
    return weight


def avg_score_by_conf(score_list, score_size_list, score_dist_list, score_dir_list, conf_list):
    '''
    Function:
        average score by conf
    '''
    
    conf_sum = sum(conf_list)
    
    if conf_sum == 0:
        return 0, 0, 0, 0

    conf_list = [i / conf_sum for i in conf_list]
    
    for i in range(len(conf_list)):
        score_list[i] *= conf_list[i]
        score_size_list[i] *= conf_list[i]
        score_dist_list[i] *= conf_list[i]
        score_dir_list[i] *= conf_list[i]
    
    return np.sum(score_list), np.sum(score_size_list), np.sum(score_dist_list), np.sum(score_dir_list)
    

def judge_dir_symmetry(relation: str):
    '''
    Function:
        judge dir_symmetry according to relation
    '''
    
    if 'left' in relation.lower() or 'right' in relation.lower():
        return False
    
    return True


def get_eval_example_list_s_o(eval_example_list: list, subject: str, object: str, sim_map: dict, sim_threshold: float = 0.8):
    '''
    Function:
        get eval_example_list_s_o from eval_example_list
    
    Args:
        eval_example_list = [{'img_id', 'img_path', 's_bbox', 'o_bbox', 'subject', 'object'}]
    '''
    
    eval_example_list_s_o = []
    
    for i in eval_example_list:

        try:
            s_sim = sim_map[convert_name(subject)][i['subject'].lower()]
            o_sim = sim_map[convert_name(object)][i['object'].lower()]
        except:
            s_sim = 0
            o_sim = 0
        
        if s_sim >= sim_threshold and o_sim >= sim_threshold:
            
            eval_example_list_s_o.append({
                "img_id": i['info']['img_id'],
                'img_path': i['info']['img_path'],
                'caption': i['info']['caption'],
                's_bbox': i['s_bbox'],
                'o_bbox': i['o_bbox'],
                'subject': i['subject'],
                'object': i['object'],
                's_sim': s_sim,
                'o_sim': o_sim
            })
    
    return eval_example_list_s_o


def convert_name(name: str):
    '''
    Function:
        convert name
    '''

    index = name.split('-')[-1]
    name = name.replace(f'-{index}', '')
    name = name.replace('_', ' ')

    if '(' in name:
        index = name.find('(')
        if name[index - 1] != ' ':
            name = name.replace('(', ' (')

    return name


'''
============================== Eval Functions ==============================
'''

def eval_size(pred_s_bbox: list, pred_o_bbox: list, example_s_bbox: list, example_o_bbox: list, weight: float = 1.0, penalty_threshold: float = 0.1, penalty_func: Callable = linear_penalty):
    '''
    Function:
        Evaluate size similarity
    '''
    
    # calculate area ratio
    A = cal_area_ratio(pred_s_bbox, pred_o_bbox)
    B = cal_area_ratio(example_s_bbox, example_o_bbox)
    
    # calculate sim
    score = cal_sim(A, B) * weight
    
    # penalty
    score = penalty_func(score, penalty_threshold)
    
    return score
    

def eval_dist(pred_s_bbox: list, pred_o_bbox: list, example_s_bbox: list, example_o_bbox: list, weight: float = 1.0, penalty_threshold: float = 0.1, penalty_func: Callable = linear_penalty, use_balance: bool = False):
    '''
    Function:
        Evaluate dist similarity
    '''
    
    # calculate iou
    iou1 = cal_iou(pred_s_bbox, pred_o_bbox)
    iou2 = cal_iou(example_s_bbox, example_o_bbox)
    
    # calculate relative distance
    rel_dist1 = cal_rel_dist(pred_s_bbox, pred_o_bbox)
    rel_dist2 = cal_rel_dist(pred_s_bbox, pred_o_bbox)
    
    # calculate sim
    score_iou = cal_sim(iou1, iou2) * weight
    score_rel_dist = cal_sim(rel_dist1, rel_dist2) * weight
    
    # penalty
    score_iou = penalty_func(score_iou, penalty_threshold)
    score_rel_dist = penalty_func(score_iou, penalty_threshold)
    
    # assign weights
    weights = [0.5, 0.5]
    if use_balance:
        if score_iou > 0.95:
            weights = [0.2, 0.8]
        elif score_rel_dist > 0.95:
            weights = [0.8, 0.2]
    
    score = score_iou * weights[0] + score_rel_dist * weights[1]
    
    return score


def eval_dir(pred_s_bbox: list, pred_o_bbox: list, example_s_bbox: list, example_o_bbox: list, mode: str = 'oppo', weight: float = 1.0, penalty_threshold: float = 0.1, penalty_func: Callable = linear_penalty, use_symmetry: bool = False):
    '''
    Function:
        Evaluate direction similarity
    
    Args:
        mode: str, ['perp', 'oppo']. 'perp': score lower when reaching 90 degree. 'oppo': score lower when reaching 180 degree.
    '''
    
    # calculate dir vector
    dir1 = cal_dir(pred_s_bbox, pred_o_bbox)
    dir2 = cal_dir(example_s_bbox, example_o_bbox)
    if use_symmetry:
        dir3 = np.array([-dir2[0], dir2[1]])
    
    # process 0 vector
    if np.all(dir1 == 0) and np.all(dir2 == 0):
        return 1
    elif np.all(dir1 == 0) or np.all(dir2 == 0):
        return 0

    # calculate cos
    cos = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
    if use_symmetry:
        cos_symmetry = np.dot(dir1, dir3) / (np.linalg.norm(dir1) * np.linalg.norm(dir3))
        cos = max(cos, cos_symmetry)
    
    # convert
    cos = abs(cos) if mode == 'perp' else (cos + 1) / 2
    cos *= weight
    
    cos = penalty_func(cos, penalty_threshold)
    
    return cos
    
    
def rule_eval_single_item(pred_s_bbox: list, pred_o_bbox: list, example_list: list, weights: list, size_penalty_threshold: float = 0.03, dist_penalty_threshold: float = 0.03, dir_penalty_threshold: float = 0.03, use_balance: bool = True, use_symmetry: bool = True):
    '''
    Function:
        eval single item according to rules designed
    '''
    
    # initialize the task
    ret = []
    index_len = 0
    conf = 0.0
    
    # evaluate
    for example in example_list:
        
        # update weight
        s_weight = update_weight(example['s_sim'])
        o_weight = update_weight(example['o_sim'])
        weight = s_weight * o_weight
        
        # update conf
        if example['s_sim'] >= 0.8 and example['o_sim'] >= 0.8:
            conf += weight
            index_len += 1
        else:
            conf += (weight * 0.0001)
        
        # calculate score
        score_size = eval_size(pred_s_bbox, pred_o_bbox, example['s_bbox'], example['o_bbox'], weight=weight, penalty_threshold=size_penalty_threshold)
        score_dist = eval_dist(pred_s_bbox, pred_o_bbox, example['s_bbox'], example['o_bbox'], weight=weight, penalty_threshold=dist_penalty_threshold, use_balance=use_balance)
        score_dir = eval_dir(pred_s_bbox, pred_o_bbox, example['s_bbox'], example['o_bbox'], weight=weight, penalty_threshold=dir_penalty_threshold, use_symmetry=use_symmetry)
        
        score = (score_size * weights[0] + score_dist * weights[1] + score_dir * weights[2]) / sum(weights)
        
        # update ret
        ret.append({
            "img_id": example['img_id'],
            "img_path": example['img_path'],
            "pred_s_bbox": pred_s_bbox,
            "pred_o_bbox": pred_o_bbox,
            "example_s_bbox": example['s_bbox'],
            "example_o_bbox": example['o_bbox'],
            "score_size": score_size,
            "score_dist": score_dist,
            "score_dir": score_dir,
            "score": score,
            "sim": weight,
        })
    
    # order
    ret.sort(key=lambda x: x['score'], reverse=True)
    
    # calculate index
    index = int(index_len / 50)
    
    return ret, conf, ret[index]
        

def rule_eval_offline(pred_path: str, example_path: str = 'config/eval/relations_one_to_one.json', weights: list = [0.5, 1.0, 0.8], use_balance: bool = True, use_symmetry: bool = True, sim_threshold: float = 0.8, sim_map_path: str = 'config/eval/sim_map.json'):
    '''
    Function:
        eval layout according to rules designed
    
    Args:
        pred_path: predictions from LLMs
        example_path: examples to refer to in evaluation
        
    '''
    
    # get pred list
    with open(pred_path, 'r') as f:
        pred_list = json.load(f)
    
    # get example_list
    with open(example_path, 'r') as f:
        example_list = json.load(f)
    
    # get sim_map
    with open(sim_map_path, 'r') as f:
        sim_map = json.load(f)

    # initialize the task
    eval_fail_cnt = 0 # record the number of eval fail
    pred_fail_cnt = 0 # record the number of pred fail
    ret_list = []
    score_list = []
    score_size_list = []
    score_dist_list = []
    score_dir_list = []
    conf_list = []
    
    # evaluate
    for i in tqdm(range(len(pred_list)), desc="Evaluating"):
        
        # get item from list
        pred_item = pred_list[i]
        
        # get relations
        relation_list = pred_item['relations']
        
        # initialize the task
        eval_fail = False
        pred_fail = False
        score_relation_list = []
        score_size_relation_list = []
        score_dist_relation_list = []
        score_dir_relation_list = []
        conf_relation_list = []
        
        # evaluate relations
        for relation_item_list in relation_list:

            # check eval_fail and pred_fail
            if eval_fail or pred_fail:
                break
            
            for relation_item in relation_item_list['relations']:

                # check eval_fail and pred_fail
                if eval_fail or pred_fail:
                    break
                
                try:
                    # get info from relation_item
                    relation = relation_item['relation']
                    subject_list = relation_item['subject']
                    object_list = relation_item['object']

                    assert type(subject_list) == list and type(object_list) == list

                    # get example_list to evaluate
                    eval_example_list = example_list[relation]
                    print(len(eval_example_list))
                except:
                    eval_fail_cnt += 1
                    eval_fail = True
                    break

                # set dir_symmetry
                dir_symmetry = judge_dir_symmetry(relation)
                
                # evaluate subject - object pair
                for s in subject_list:
                    
                    # check eval_fail and pred_fail
                    if eval_fail or pred_fail:
                        break
                    
                    for o in object_list:
                        
                        # get example_list according to subject and object
                        eval_example_list_s_o = get_eval_example_list_s_o(eval_example_list, s, o, sim_map, sim_threshold=sim_threshold)
                        
                        # check len
                        if len(eval_example_list_s_o) == 0:
                            eval_fail_cnt += 1
                            eval_fail = True
                            break
                            
                        # get pred layout
                        pred_s_bbox, pred_o_bbox = get_pred_layout(pred_item, s, o)
                        
                        # check pred layout
                        if pred_s_bbox == -1 or pred_o_bbox == -1 or len(pred_s_bbox) != 4 or len(pred_o_bbox) != 4:
                            pred_fail_cnt += 1
                            pred_fail = True
                            break
                            
                        # evaluate
                        _, conf, ret = rule_eval_single_item(pred_s_bbox, pred_o_bbox, eval_example_list_s_o, weights=weights, use_balance=use_balance, use_symmetry=use_symmetry)
                        
                        # update ret_list
                        ret['caption'] = pred_item['caption']
                        ret_list.append(ret)
                        
                        # update score_list, conf_list
                        score_relation_list.append(ret['score'])
                        score_size_relation_list.append(ret['score_size'])
                        score_dist_relation_list.append(ret['score_dist'])
                        score_dir_relation_list.append(ret['score_dir'])
                        conf_relation_list.append(conf)
    
        if eval_fail or pred_fail or len(score_relation_list) == 0:
            score_list.append(0)
            score_size_list.append(0)
            score_dist_list.append(0)
            score_dir_list.append(0)
            conf_list.append(0)
        else:
            score, score_size, score_dist, score_dir = avg_score_by_conf(score_relation_list, score_size_relation_list, score_dist_relation_list, score_dir_relation_list, conf_relation_list)
            score_list.append(score)
            score_size_list.append(score_size)
            score_dist_list.append(score_dist)
            score_dir_list.append(score_dir)
            conf_list.append(min(conf_relation_list))
            
    print("pred_fail_cnt: ", pred_fail_cnt)
    print("eval_fail_cnt: ", eval_fail_cnt)
    
    save_rec = {
        "ret_list": ret_list,
        "score_list": score_list,
        "score_size_list": score_size_list,
        "score_dist_list": score_dist_list,
        "score_dir_list": score_dir_list,
        "pred_fail_cnt": pred_fail_cnt,
        "eval_fail_cnt": eval_fail_cnt,
        "conf_list": conf_list,
    }

    file_name = os.path.basename(pred_path)
    folder_name = os.path.dirname(pred_path)

    tar_dir = os.path.join(folder_name, 'eval/')
    os.makedirs(tar_dir, exist_ok=True)

    prefix = f"{sim_threshold}_"
    if use_balance:
        prefix += 'use_balance_'
    if use_symmetry:
        prefix += 'use_symmetry_'
    file_name = prefix + file_name

    # print(tar_dir)
    # print(file_name)

    with open(os.path.join(tar_dir, file_name), 'w') as f:
        json.dump(save_rec, f)
    
    return score_list, score_size_list, score_dist_list, score_dir_list, conf_list, ret_list


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pred_path', type=str, default='lvis/debug.json')
    parser.add_argument('--example_path', type=str, default='flickr/parsed/combine/relations_one_to_one.json')
    parser.add_argument('--use_balance', action='store_true')
    parser.add_argument('--use_symmetry', action='store_true')
    parser.add_argument('--sim_threshold', type=float, default=0.6)
    
    args = parser.parse_args()
    
    score_list, score_size_list, score_dist_list, score_dir_list, conf_list, ret_list = rule_eval_offline(
        pred_path=args.pred_path,
        example_path=args.example_path,
        use_balance=args.use_balance,
        use_symmetry=args.use_symmetry,
        sim_threshold=args.sim_threshold
    )
    
    score, score_size, score_dist, score_dir = avg_score_by_conf(score_list, score_size_list, score_dist_list, score_dir_list, conf_list)
    print(score)
