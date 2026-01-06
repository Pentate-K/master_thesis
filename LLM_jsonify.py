import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

# --- 設定 ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_DIR = "models/"

def load_model():
    print(f"Loading model: {MODEL_ID} ...")
    
    # Mac/CPU環境向けの安定設定
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("Using CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = "cpu" 
        dtype = torch.float32
        print("Using CPU (Safe mode for Mac)")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True
    )
    model.eval()
    return model, tokenizer

def extract_json(text):
    """
    LLMの出力からJSON部分を抽出してパースする
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    return None

def main():
    model, tokenizer = load_model()

    # 構造化通信を想定したスキーマ定義
    system_prompt = """
You are an AI agent in a grid world. You communicate with another agent to solve tasks.
You MUST output your response in valid JSON format ONLY. Do not output any other text or explanations.

Use the following JSON schema:
{
    "intent": "One of [PROPOSE, INFORM, REQUEST, AGREE, REJECT]",
    "target_object": "The name of the object you are interested in (e.g., 'red key') or null",
    "target_coordinate": [x, y] coordinates as a list of integers, or null",
    "message": "A short natural language message to your partner"
}
"""

    test_inputs = [
        "I found a red key at (3, 5). I think we need it to open the door. I will go pick it up.",
        "I cannot find the blue ball. Can you search the northern room?",
        "That sounds like a good plan. I will wait here."
    ]

    print("\n=== JSON Output Experiment ===\n")

    for i, user_input in enumerate(test_inputs):
        print(f"--- Test Case {i+1} ---")
        print(f"Input: {user_input}\n")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a JSON response based on this thought/observation:\n{user_input}"}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                # 【修正箇所】ここを追加しました
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 生成結果の取得
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Raw Output:\n{generated_text}\n")

        parsed_json = extract_json(generated_text)
        if parsed_json:
            print("✅ JSON Parse Success:")
            print(json.dumps(parsed_json, indent=2))
        else:
            print("❌ JSON Parse Failed")
        
        print("\n")

if __name__ == "__main__":
    main()