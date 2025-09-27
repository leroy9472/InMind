import os
import json
import argparse
from typing import Dict, Any, List

# vLLM is recommended for high-performance inference
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ======== 配置部分 (Configuration) ========
OBSERVER_DIR = "./InMind-Avalon/observer_mode"
PARTICIPANT_DIR = "./InMind-Avalon/participant_mode"
OUTPUT_TEMPLATE = "./results/player_identification/run_{run_num}/task1_{model_name}.json"
TENSOR_PARALLEL_SIZE = 4

# ======== 核心函数 (Core Functions) ========

def convert_json_to_text_format(json_data: Dict[str, Any], game_id: int) -> str:
    """
    将 Observer-mode 的 JSON 数据动态转换为指定的 TXT 格式字符串。
    This function converts an observer-mode JSON object into a structured text format
    that serves as a reference game for profile generation.
    """
    lines = []
    lines.append(f"=== 参考游戏 {game_id} ===")
    my_info = json_data.get("my_info", {}).get("1", {})
    lines.append("[我的初始信息 (旁观视角)]")
    lines.append(f"  - 玩家身份: {my_info.get('role', 'N/A')}")
    lines.append(f"  - 代入玩家编号: {my_info.get('player', 'N/A')}")
    lines.append("\n[游戏过程及我的旁观分析]")
    task_keys = sorted([k for k in json_data.keys() if k.startswith("task")])
    for task_key in task_keys:
        task_data = json_data[task_key]
        round_num = task_data.get("player_message", [{}])[0].get("round", "N/A")
        lines.append(f"  --- 回合 {round_num} ---")
        lines.append("    [玩家发言]")
        for msg in task_data.get("player_message", []):
            player = msg.get('player', 'system')
            content = msg.get('msg', '')
            lines.append(f"      - [玩家{player}]: {content}")
        strategy_list = task_data.get("strategy", [])
        if strategy_list:
            lines.append("    [我的实时分析 (Strategy)]")
            for strat in strategy_list:
                lines.append(f"      - {strat.get('msg', '')}")
    lines.append("\n[我的全局复盘 (Review)]")
    my_review = json_data.get("my_review", {})
    lines.append(f"  - {my_review.get('msg', 'N/A')}")
    return "\n".join(lines)


def format_participant_game_prompt(data: Dict[str, Any]) -> str:
    """
    将 Participant-mode 的游戏数据格式化为字符串，用于最终的玩家识别任务。
    Formats a participant-mode game session into a string for the final prediction prompt.
    """
    lines = []
    task_keys = sorted([k for k in data.keys() if k.startswith("task")])
    for task_key in task_keys:
        task_data = data.get(task_key, {})
        round_num = task_data.get("player_message", [{}])[0].get("round", "N/A")
        lines.append(f"--- 回合 {round_num} ---")
        lines.append("  [所有玩家发言 (player_message)]")
        player_messages = task_data.get("player_message", [])
        for pm in player_messages:
            p = pm.get("player", "system")
            m = pm.get("msg", "")
            lines.append(f"    - [玩家{p}]: {m}")
        lines.append("\n  [我的内心想法 (strategy)]")
        strategy_traces = task_data.get("strategy", [])
        for st in strategy_traces:
            m = st.get("msg", "")
            lines.append(f"    - {m}")
    return "\n\n".join(lines)


def generate_reasoning_profile(llm: LLM, tokenizer: AutoTokenizer, observer_texts: List[str]) -> str:
    """
    Stage 1: Capturing Individual Reasoning Styles.
    """
    print("Stage 1: Generating reasoning profile from observer games...")
    system_prompt = "你是一名精通阿瓦隆桌游的分析专家，擅长透过玩家的行为和发言推测玩家的推理逻辑、人物风格和行动策略。你的任务是基于用户提供的旁观游戏记录，精准地总结用户在旁观时展现的推理风格、发言倾向以及分析局势的方法，为之后识别用户实际参与游戏时的发言特点建立准确的用户画像。"
    accumulated_summary = ""
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)

    for i, ref_text in enumerate(observer_texts, 1):
        print(f"  Analyzing observer game {i}/{len(observer_texts)}...")
        
        user_prompt = f"""以下是我旁观某位玩家时记录的一局游戏数据，包含我的分析和评价：

{ref_text}

结合此前你已经帮我总结过的：
{accumulated_summary}

请基于之前的总结以及上述新的游戏数据，重新生成一个更全面、更细致的『综合玩家画像』，该画像应包括但不限于：

- 我的推理风格（如侧重逻辑推理、发言细节推敲、人际互动观察等）
- 我可能的发言特征（如发言长度、用词风格、信息分享的方式、阵营表态的倾向）
- 我对游戏局势的常规分析策略（如喜欢从哪些信息判断敌我关系、如何处理模糊信息、如何引导队友）

要求精准、细致地进行归纳总结，以便下一步准确推测实际游戏中的我。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt_text], sampling_params)
        accumulated_summary = outputs[0].outputs[0].text
        
    print("Profile generation complete.\n")
    return accumulated_summary

def identify_player(llm: LLM, tokenizer: AutoTokenizer, reasoning_profile: str, participant_game_prompt: str) -> str:
    """
    Stage 2: Applying Profile in Adaptive Task.
    """
    system_prompt = """你是一名精通阿瓦隆桌游的分析专家，擅长透过玩家的发言和行为推测玩家的真实身份。此前你已经详细总结了用户旁观时的推理风格、人物特点和典型发言倾向，建立了一个准确的人物画像。

你当前的***唯一任务***是：
根据用户的『综合玩家画像』，精准地在实际游戏数据中，通过分析玩家的发言特点、用词习惯和推理策略，识别哪位玩家是用户本人。

请严格按照以下输出要求：
- 仅输出玩家编号（player1 至 player6）
- 不要输出任何额外解释或说明
- 输出最可能的玩家 (Top1) 和最可能的前三玩家 (Top3)，严格以JSON结构给出。"""

    user_prompt = f"""此前你已为我精准总结了我的『综合玩家画像』：
{reasoning_profile}

以下是一局我实际参与的游戏数据：
- player_message：包含本局游戏所有玩家的发言记录（含我本人）
- strategy：包含我对当前局势的内部分析和我的发言策略等细节
{participant_game_prompt}

你的***唯一任务***是根据上述游戏数据，判断在编号为 player1 至 player6 的玩家中，哪位玩家最可能是我。

严格按照以下格式输出：
```json
{{
    "top1": "playerX",
    "top3": ["playerX", "playerY", "playerZ"]
}}

注意：
        - 请严格遵循 JSON 格式，仅输出玩家编号
        - 排名应准确反映与『综合玩家画像』的匹配程度"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.2, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)
    
    outputs = llm.generate([prompt_text], sampling_params)
    return outputs[0].outputs[0].text


def run_evaluation_once(model_path: str, llm: LLM, tokenizer: AutoTokenizer, run_num: int):
    """
    执行单次完整的 InMind 评估流程。
    """
    print(f"--- Starting Run {run_num}/5 ---")
    observer_files = sorted([f for f in os.listdir(OBSERVER_DIR) if f.endswith(".json")])
    observer_game_texts = []
    for i, filename in enumerate(observer_files, 1):
        filepath = os.path.join(OBSERVER_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        observer_game_texts.append(convert_json_to_text_format(json_data, i))
    
    reasoning_profile = generate_reasoning_profile(llm, tokenizer, observer_texts)
    
    print("Stage 2: Starting Player Identification task for this run...")
    all_results = []
    participant_files = sorted([f for f in os.listdir(PARTICIPANT_DIR) if f.endswith(".json")])
    for filename in participant_files:
        print(f"  Processing file: {filename}...")
        filepath = os.path.join(PARTICIPANT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            participant_data = json.load(f)
        participant_prompt = format_participant_game_prompt(participant_data)
        prediction_str = identify_player(llm, tokenizer, reasoning_profile, participant_prompt)
        true_player_id = participant_data.get("my_info", {}).get("1", {}).get("player", "N/A")
        result_data = {
            "file_name": filename,
            "true_player_id": f"player_{true_player_id}",
            "model_prediction_raw": prediction_str
        }
        all_results.append(result_data)
        
    final_output = {
        "run_number": run_num,
        "reasoning_profile": reasoning_profile,
        "identification_results": all_results
    }
    model_name = os.path.basename(model_path.rstrip('/'))
    output_path = OUTPUT_TEMPLATE.format(run_num=run_num, model_name=model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"--- Run {run_num} complete. Results saved to: {output_path} ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the InMind Player Identification Task (5 runs), aligned with paper prompts.")
    parser.add_argument('model_path', type=str, help="Path to the Hugging Face model directory.")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=TENSOR_PARALLEL_SIZE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    for i in range(1, 6):
        run_evaluation_once(args.model_path, llm, tokenizer, run_num=i)
        
    print("All 5 runs are complete.")