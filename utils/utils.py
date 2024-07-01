import os
import json


# ------------------------- Function ------------------------- # 

def _scale_bbox(bbox, ori_width, ori_height, tar_width, tar_height):
    '''
    Function:
        scale bbox from original width and height to target
    '''
    
    scale_width = tar_width / ori_width
    scale_height = tar_height / ori_height
    
    return [bbox[0] * scale_width, bbox[1] * scale_height, bbox[2] * scale_width, bbox[3] * scale_height]


def scale_bbox(bbox: list, width: float = 512, height: float = 512):
    '''
    Function:
        convert bbox from xywh to xyxy with 512 * 512
    '''
    
    return [max(0, bbox[0] * width), max(bbox[1] * height, 0), min((bbox[0] + bbox[2]) * width, 512), min((bbox[1] + bbox[3]) * height, 512)]


def create_dir(path):
    '''
    Function:
        check whether the parent dir of path exists, create one if not
    '''
    
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def get_sg(
    desc: dict,
    layout: list    
):
    '''
    Function:
        combine description and layout to get scene graph
    '''
    
    desc['layout'] = layout
    
    return desc


def get_attribute(
    name: str,
    sg: dict,
):
    '''
    Function:
        get attribute of name
    '''
    
    for i in sg['attributes']:
        if i['name'].lower() == name.lower():
            return i['attributes']
    
    index = int(name.split('-')[-1])
    for i in sg['attributes']:
        c_index = int(i['name'].split('-')[-1])
        if c_index == index:
            return i['attributes']
    
    return name

    
# ------------------------- Convert Format ------------------------- # 

def convert_output(ori_output):
    '''
    Function:
        convert ori_output to the desired format and value
    
    Parameter:
        ori_output: json {'width', 'height', 'Layout'}
    '''
    
    layout = {}
    layout['Layout'] = []
    
    try:
        width = ori_output['width']
        height = ori_output['height']
    except:
        return layout
    
    for i in ori_output['Layout']:
        layout['Layout'].append({'object': i['object'], 'bbox': _scale_bbox(i['bbox'], ori_width=width, ori_height=height, tar_width=1, tar_height=1)})
    
    return layout


def convert_example_list(task, ori_example_list):
    # convert ori_example_list from file to the desired format
    if task == 'name_to_caption':
        return [{"prompt": json.dumps(i['objects'], indent=4), "output": i['dense_caption']}for i in ori_example_list]
    elif task == 'name_to_all' or task == 'name_to_relations' or task == 'create_dataset':
        return [{"prompt": json.dumps(i['objects'], indent=4), "output": json.dumps(i['output'], indent=4)} for i in ori_example_list]
    elif task == 'caption':
        return [{"prompt": json.dumps(i['scene_graph'], indent=4), "output": i['dense_caption']} for i in ori_example_list]
    elif task == 'bbox' or task == 'all_to_bbox':
        return [{"prompt": json.dumps(i['prompt'], indent=4), "output": json.dumps(convert_output(i['output']), indent=4)} for i in ori_example_list]
    elif task == 'feedback_bbox':
        return [{"prompt": json.dumps(i['prompt'], indent=4), "output": json.dumps(convert_output(i['output']), indent=4), "feedback": json.dumps(i['feedback'], indent=4)} for i in ori_example_list]
    elif task == 'iter_bbox':
        return [{"prompt": json.dumps(i['prompt'], indent=4), "output_to_feedback": [{"output": json.dumps(convert_output(j['output']), indent=4), "feedback": json.dumps(j['feedback'], indent=4) if isinstance(j['feedback'], dict) else j['feedback']} for j in i['output_to_feedback']]} for i in ori_example_list]
    elif task == 'parse_caption' or task == 'parse_relations':
        return [{"prompt": json.dumps(i['prompt'], indent=4), "output": json.dumps(i['output'], indent=4)} for i in ori_example_list]
    else:
        raise ValueError("Wrong Task!")
    

def get_query(task, ori_query):
    '''
    Function:
        get query for tasks
    '''
    
    if task == 'create_dataset':
        ret = {}
        ret['objects'] = []
        
        for i in range(len(ori_query)):
            ret['objects'].append(ori_query[i] + '-' + str(i + 1))
        
        return ret
    
    elif task == 'bbox':
        q_objects = []
        for i in ori_query['attributes']:
            q_objects.append(i['name'])
        
        return {
            "objects": q_objects,
            "dense_caption": ori_query['caption']
        }
        