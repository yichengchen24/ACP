import json
import re


# ------------------------- Parse ------------------------- # 

def parse_str(response, stop_word=None, prefix_word=None):
    '''
    Function:
        from response str parse target dict
    
    Args:
        stop_word: str, default = None, filter the content of response before stop_word
    '''
    
    # filter response
    if stop_word:
        stop_index = response.lower().find(stop_word.lower())
        if stop_index != -1:
            response = response[:stop_index]
    
    if prefix_word:
        prefix_index = response.lower().find(prefix_word.lower())
        if prefix_index != -1:
            response = response[prefix_index:]
    
    start = response.find('{')
    end = response.rfind('}')
    
    if start != -1 and end != -1:
        try:
            processed_response = json.loads(response[start:end+1])
        except:
            cleaned_response = re.sub(r"//.*", "", response[start:end+1])
            cleaned_response = re.sub(r'(?<!["\'])\s*#.*', '', cleaned_response)
            print(cleaned_response)
            processed_response = json.loads(cleaned_response)
    else:
        processed_response = response
    
    return processed_response


def parse_score(response: str):
    
    try:
        score = parse_str(response)['score']
    except:
        try:
            match = re.search(r"(?i)score:\s*(\d+)", response)
            score = int(match.group(1))
        except:
            return -1
    
    score *= 20
    
    return score
