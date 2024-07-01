import json
import requests


'''
define GPT parameters
'''
gpt_url = ""
gpt_headers = {
    
}

    
def infer(model, prompt, max_tokens=512, tp=2, top_p=0.8, top_k=40, temperature=0.8, h_model=None, h_tokenizer=None):
    '''
    Function:
        Infer response of content from model
    
    Args:
        model: models to infer, ['gpt-3.5-turbo', 'gpt-4', 'internlm', 'backend', 'qwen']
        content: the prompt to infer
    '''
    
    # get messages from content
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    if h_model and h_tokenizer:
        device = "cuda"
        
        if 'llama' in model:
            input_ids = h_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
            terminators = [
                h_tokenizer.eos_token_id,
                h_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = h_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response_ori = outputs[0][input_ids.shape[-1]:]
            
            response = h_tokenizer.decode(response_ori, skip_special_tokens=True)
            
            return response
        
        
        text = h_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = h_tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = h_model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = h_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    if 'gpt' in model:
        payload = {
            "model": model,
            "messages": messages
        }
        
        response = requests.post(gpt_url, headers=gpt_headers, data=json.dumps(payload)).json()['data']
    
    return response.text if 'backend' in model else response['choices'][0]['message']['content']
