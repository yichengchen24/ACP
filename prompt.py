def get_content_template(task):
    '''
    Function:
    '''
    
    if task == 'create_dataset':
        content_template = """Please provide a JSON format with "attributes", "groups", "layers of depth", "relations", and "caption" based on the following prompt: {prompt}.\nDesired output:\n"""
    elif task == 'bbox':
        content_template = """Please provide a json format with 'Layout' based on the following prompt: {prompt}.\nThe layout is a list of json with 'object' and 'bbox'. 'object' refers to the object name in the prompt provided, while 'bbox' is formulated as [x,y,w,h], where "x,y" denotes the top left coordinate of the bounding box. "w" denotes the width, and "h" denotes the height. The bounding boxes should not go beyond the image boundaries. The six values "x,y,w,h,x+w,y+h" are all larger than 0 and smaller than 1."""
    
    return content_template


def get_task_description(task):
    '''
    Function:
        Generate task description given different task
    '''
    
    if task == 'create_dataset':
        # generate the data
        task_description = """We want to generate a scene graph given a list of objects. The object is in the format of '{object name}-{identifier}', where 'object name' means any object name in the world and 'identifier' is a unique number representing the difference between objects, especially those with the same name.\nPlease provide a JSON format with 'attributes', 'groups', 'layers of depth', 'relations', and 'caption'.\n1. 'attributes': should be descriptive color or texture of the corresponding object.\n2. 'groups': A group of objects exhibit strong spatial relationships that interact with each other.\n3. 'layers of depth': The scene is divided into different layers based on proximity - 'Immediate Foreground', 'Foreground', 'Midground', and 'Background'. Each layer depicts one or more groups of objects in (2) at that depth in the scene.\n4. 'relations': This section illustrates the interactions or spatial relationships between various objects or groups.\n5. 'caption': A simple and straightforward 1-2 sentence image caption. Please include all the objects in the caption and refer to them in '()'. Create the caption as if you are directly observing the image. Do not mention the use of any source data. Do not use words like 'indicate', 'suggest', 'hint', 'likely', or 'possibly'.\n\n"""
    elif task == 'bbox':
        # generate bbox given a list of objects and dense caption
        task_description = """The provided prompt is a list of object and corresponding description. The object is in the format of '{object name}-{identifier}', where 'object name' means any object name in the world and 'identifier' is a unique number representing the difference between objects, especially those with the same name. The corresponding description shows the relationship between objects."""
        
    task_description = task_description + "Please refer to the examples below.\n"
    
    return task_description


def get_norm_examples(example_list):
    
    example_template = """##Example-{index}\nPrompt: {prompt}\nDesired Output: {output}\n"""
    
    example_content = []
    for i, example in enumerate(example_list):
        example_content.append(example_template.format(index=i, prompt=example['prompt'], output=example['output']))
    
    examples = '\n'.join(example_content)

    return examples


def get_examples(task, example_list):
    
    return get_norm_examples(example_list)
    
    
def get_norm_content(task, q):
    
    content_template = get_content_template(task)
    
    content = content_template.format(prompt=q['prompt'], output=q['output']) if 'feedback' in task else content_template.format(prompt=q['prompt'])
    
    return content


def get_content(task, q):
    
    return get_norm_content(task, q)
    

def get_prompt(
    task: str, 
    example_list: list = [], 
    q: dict = None,
):
    '''
    Function:
    
    '''
    
    if task == 'vlm_global_describe':
        prompt = """You are my assistant to evaluate the correspondence of the image to a given text prompt.\nBriefly describe the image within 50 words, focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships.\n"""
        
        return prompt
    
    elif task == 'vlm_local_describe':
        prompt = """You are my assistant to identify the object and its color (shape, texture) in the <box>({xmin},{ymin}),({xmax},{ymax})</box> of the image.\nBriefly describe what it is in the specific part of the image within 50 words.\n"""
        
        return prompt
    
    elif task == 'llm_align':
        prompt = """You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match correctly based on the objects, and their actions, relationships. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the objects mentioned in the description and their actions and relationships when evaluating the meaningful match.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer.\n\nPlease Evaluate the following answer pair:\n\nCorrect Answer: {answer}\nPredicted Answer: {pred}\n\nProvide your evaluation in the JONSON format with 'score' and 'explanation' key. The score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. The explanation should be within 20 words."""
        
        return prompt    
    
    template = """Task Description:\n{task_description}\nExamples:\n{examples}\nPlease complete the following one:\n{content}"""
        
    prompt = template.format(
        task_description=get_task_description(task),
        examples=get_examples(task, example_list),
        content=get_content(task, q),
    )
    
    return prompt
    