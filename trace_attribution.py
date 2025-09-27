import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple

# vLLM is recommended for high-performance inference
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ======== 配置部分 (Configuration) ========
OBSERVER_DIR = "./InMind-Avalon/observer_mode"
PARTICIPANT_DIR = "./InMind-Avalon/participant_mode"
OUTPUT_TEMPLATE = "./results/trace_attribution/task3_{model_name}.json"
TENSOR_PARALLEL_SIZE = 4

# ======== 核心函数 (Core Functions) ========

def convert_json_to_text_format(json_data: Dict[str, Any], game_id: int) -> str:
    """
    (复用) 将 Observer-mode 的 JSON 数据动态转换为指定的 TXT 格式字符串。
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
            lines.append(f"      - [玩家{msg.get('player', 'system')}]: {msg.get('msg', '')}")
        strategy_list = task_data.get("strategy", [])
        if strategy_list:
            lines.append("    [我的实时分析 (Strategy)]")
            for strat in strategy_list:
                lines.append(f"      - {strat.get('msg', '')}")
    lines.append("\n[我的全局复盘 (Review)]")
    my_review = json_data.get("my_review", {})
    lines.append(f"  - {my_review.get('msg', 'N/A')}")
    return "\n".join(lines)

def prepare_and_mask_game_rounds(data: Dict[str, Any]) -> Tuple[str, List[Dict]]:
    """
    准备单局游戏数据：格式化，对strategy进行mask，并按轮次切分。
    返回游戏初始信息和处理后的轮次列表。
    """
    my_info_lines = ["[My Info]"]
    my_info_data = data.get("my_info", {}).get("1", {})
    my_info_lines.append(f"  - 你的初始身份: {my_info_data.get('role', 'N/A')}, 你的玩家编号: {my_info_data.get('player', 'N/A')}")
    my_info_text = "\n".join(my_info_lines)

    processed_rounds = []
    mask_index = 1  # 在每个文件处理开始时重置

    task_keys = sorted([k for k in data.keys() if k.startswith("task")])
    for task_key in task_keys:
        task_data = data[task_key]
        round_lines = [f"=== {task_key} / Round {task_data.get('player_message', [{}])[0].get('round', 'N/A')} ==="]
        
        # 处理 player_message
        round_lines.append("  [Player Messages]")
        for pm in task_data.get("player_message", []):
            round_lines.append(f"    - [玩家{pm.get('player', 'system')}]: {pm.get('msg', '')}")
            
        # 处理 strategy 并进行mask
        ground_truth_masks = []
        has_strategy = False
        strategy_list = task_data.get("strategy", [])
        if strategy_list:
            has_strategy = True
            round_lines.append("  [My Strategy]")
            for st in strategy_list:
                original_strategy = st.get("msg", "")
                pattern = re.compile(r'\b[1-6]\b')
                
                def replacer(match: re.Match):
                    nonlocal mask_index
                    original_number = match.group(0)
                    mask_tag = f"[MASK_{mask_index}(1digit)]"
                    ground_truth_masks.append(f"{mask_tag} => {original_number}")
                    mask_index += 1
                    return mask_tag
                
                masked_strategy = pattern.sub(replacer, original_strategy)
                round_lines.append(f"    - {masked_strategy}")

        processed_rounds.append({
            "text": "\n".join(round_lines),
            "has_strategy": has_strategy,
            "masks": ground_truth_masks
        })
        
    return my_info_text, processed_rounds

def generate_reasoning_profile(llm: LLM, tokenizer: AutoTokenizer, observer_texts: List[str]) -> str:
    """
    Stage 1: 生成玩家推理画像。
    """
    print("Stage 1: Generating reasoning profile from observer games...")
    system_prompt = "你是一名精通阿瓦隆桌游的分析专家，擅长透过玩家的行为和发言推测玩家的推理逻辑、人物风格和行动策略。你的任务是基于用户提供的旁观游戏记录，精准地总结用户在旁观时展现的推理风格、发言倾向以及分析局势的方法，为之后推测用户实际参与游戏时的策略标注建立准确的用户画像。"
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
- 我的典型发言特征（如发言长度、用词风格、信息分享的方式、阵营表态的倾向）
- 我对游戏局势的常规分析策略（如喜欢从哪些信息判断敌我关系、如何处理模糊信息、如何引导队友）
要求精准、细致地进行归纳总结，以便下一步准确推测实际参与游戏时我可能的策略标注内容。"""
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt_text], sampling_params)
        accumulated_summary = outputs[0].outputs[0].text
    
    print("Profile generation complete.\n")
    return accumulated_summary

def attribute_trace(llm: LLM, tokenizer: AutoTokenizer, profile: str, game_summary: str, current_round_text: str) -> str:
    """
    Stage 2: 动态归因，让模型总结内容并填充当前轮次的mask。
    """
    # System Prompt 已与论文完全对齐
    system_prompt = """你是一个擅长分析阿瓦隆游戏的大模型。
你会根据我的推理风格，分析每一轮游戏数据并根据要求输出内容。请确保所有的输出严格遵守格式要求，尤其是替换映射的格式。

你将接收到我提供的游戏数据，要求你对其进行分析，按照以下方式输出：
1. 游戏数据的总结（Content部分）以 === Content === 开头，按顺序总结任务的内容。
2. 策略中被MASK的玩家编号以及对应的映射行，以 === Replacements === 开头，并按照 MASK_x(Ndigits) => abc... 的格式输出。请确保仅输出实际需要的映射行，且格式无误。
对于 [MASK_x(1digit)]，请用单个数字表示该位置的玩家编号，对于 [MASK_x(Ndigits)]，请用多个数字组成一个数字组合，数字按升序排列，表示多位玩家标号。

要求：
- 格式要求严格：输出中的替换映射应遵循 [MASK_x(1digit)] => m 或 [MASK_x(Ndigits)] => abc...。
- 不输出多余的映射行，仅填写实际需要的映射行，注意映射行一定要完整。
- 不输出任何其他内容。"""

    # User Prompt 已与论文完全对齐
    user_prompt = f"""此前你已经帮我总结了：我旁观游戏时表现出的推理风格和复盘策略，
{profile}

下面我会逐步地向你提供一局我实际参与的新的游戏数据，以下是你对此前游戏数据的总结，
{game_summary if game_summary else "（这是游戏的第一轮，暂无历史总结）"}

下面是新的部分游戏数据，
{current_round_text}

你有两个任务：
1. 请你总结全部的游戏数据，包括关键的玩家发言和游戏状态等，以 === Content === 开头。
2. 请你根据我的推理风格，输出我的策略（strategy）中被MASK的玩家编号，并输出映射行，以 === Replacements === 开头。

注意事项：
1. 格式要求严格：[MASK_x(1digit)] => m 或 [MASK_x(Ndigits)] => abc...。
2. 不要输出多余的映射行，仅填写实际需要的映射行。
3. 即使不确定玩家编号，也要输出完整的映射行。
4. 不要输出其他任何多余的内容。

****注意：两个任务一定要分别用=== Content ===和=== Replacements ===开头。****"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.2, top_p=0.8, repetition_penalty=1.05, max_tokens=2048)
    outputs = llm.generate([prompt_text], sampling_params)
    return outputs[0].outputs[0].text

def compare_predictions(ground_truth: list, model_output: str) -> Tuple[int, int]:
    """比对模型输出和真实答案，返回正确数量和总mask数量。"""
    truth_map = {}
    for item in ground_truth:
        match = re.search(r'(\[MASK_\d+\(.*?\)])\s*=>\s*(\d+)', item)
        if match:
            truth_map[match.group(1)] = match.group(2)
            
    model_map = {}
    for line in model_output.strip().split('\n'):
        match = re.search(r'(\[MASK_\d+\(.*?\)])\s*=>\s*(\d+)', line.strip())
        if match:
            model_map[match.group(1)] = match.group(2)
            
    correct_count = 0
    for mask, true_val in truth_map.items():
        if mask in model_map and model_map[mask] == true_val:
            correct_count += 1
            
    return correct_count, len(truth_map)

def run_evaluation_for_file(data: Dict, llm: LLM, tokenizer: AutoTokenizer, profile: str) -> Tuple[int, int, list, str]:
    """对单个文件执行完整的逐轮评估流程。"""
    my_info, game_rounds = prepare_and_mask_game_rounds(data)
    
    game_history = my_info
    game_summary_from_llm = ""
    
    total_correct = 0
    total_mask = 0
    full_model_responses = []
    all_ground_truths = []

    for current_round in game_rounds:
        current_round_text = current_round["text"]
        
        # 只有包含strategy的轮次才需要调用模型进行归因
        if current_round["has_strategy"]:
            # 将历史和当前轮次信息结合后发送给模型
            prompt_input = f"{game_history}\n{current_round_text}"
            model_response = attribute_trace(llm, tokenizer, profile, game_summary_from_llm, prompt_input)
            full_model_responses.append(model_response)
            
            # 解析模型的输出
            content_part = ""
            replacement_part = ""
            if "=== Replacements ===" in model_response:
                parts = model_response.split("=== Replacements ===", 1)
                content_part = parts[0].replace("=== Content ===", "").strip()
                replacement_part = parts[1].strip()
            elif "=== Content ===" in model_response:
                content_part = model_response.replace("=== Content ===", "").strip()

            # 使用模型自己的总结作为下一轮的历史信息
            game_summary_from_llm = content_part
            
            # 评估当前轮次的归因结果
            ground_truth = current_round["masks"]
            all_ground_truths.extend(ground_truth)
            correct, total = compare_predictions(ground_truth, replacement_part)
            total_correct += correct
            total_mask += total
        
        # 无论如何，都将当前轮次加入历史记录，为下一轮做准备
        game_history += f"\n{current_round_text}"

    return total_correct, total_mask, all_ground_truths, "\n---\n".join(full_model_responses)

# ======== Main 函数 ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 3: Trace Attribution.")
    parser.add_argument('model_path', type=str, help="Path to the Hugging Face model directory.")
    args = parser.parse_args()

    MODEL_NAME_OR_PATH = args.model_path
    model_name = os.path.basename(MODEL_NAME_OR_PATH)

    print(f"Loading model: {model_name}")
    llm = LLM(model=MODEL_NAME_OR_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    observer_files = sorted([f for f in os.listdir(OBSERVER_DIR) if f.endswith(".json")])
    observer_game_texts = [json.load(open(os.path.join(OBSERVER_DIR, f), 'r', encoding='utf-8')) for f in observer_files]
    observer_game_texts_formatted = [convert_json_to_text_format(data, i+1) for i, data in enumerate(observer_game_texts)]
    
    profile = generate_reasoning_profile(llm, tokenizer, observer_game_texts_formatted)
    
    run_results = []
    num_runs = 5
    for i in range(num_runs):
        print(f"\n--- Starting Run {i+1}/{num_runs} ---")
        run_total_correct = 0
        run_total_mask = 0
        run_details = []
        
        for filename in sorted(os.listdir(PARTICIPANT_DIR)):
            if filename.endswith(".json"):
                print(f"  Processing file: {filename}...")
                filepath = os.path.join(PARTICIPANT_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                correct, total, truths, response = run_evaluation_for_file(data, llm, tokenizer, profile)
                
                run_total_correct += correct
                run_total_mask += total
                run_details.append({
                    "file_name": filename,
                    "correct_count": correct,
                    "total_masks_in_file": total,
                    "ground_truth_masks": truths,
                    "model_responses_raw": response
                })
        
        run_summary = {"reasoning_profile": profile}
        run_details.append(run_summary)
        run_results.append({
            "run": i + 1,
            "total_correct": run_total_correct,
            "total_mask": run_total_mask,
            "results_data": run_details
        })

    if run_results:
        avg_correct = sum(run["total_correct"] for run in run_results) / num_runs
        best_run = max(run_results, key=lambda x: x["total_correct"])
        best_results_data = best_run["results_data"]
    else:
        avg_correct = 0
        best_run = {"total_correct": 0, "total_mask": 0}
        best_results_data = []

    output_path = OUTPUT_TEMPLATE.format(model_name=model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_results_data, f, ensure_ascii=False, indent=4)
        
    new_output_path = os.path.join(os.path.dirname(output_path), f"avg_score_{avg_correct:.2f}_{os.path.basename(output_path)}")
    try:
        os.rename(output_path, new_output_path)
        final_path = new_output_path
    except OSError as e:
        print(f"Error renaming file: {e}")
        final_path = output_path
        
    print("\n--- Evaluation Summary ---")
    print(f"Total runs: {num_runs}")
    print(f"Best run total correct: {best_run['total_correct']}")
    print(f"Best run total masks: {best_run['total_mask']}")
    if best_run['total_mask'] > 0:
        best_accuracy = best_run['total_correct'] / best_run['total_mask']
        print(f"Best run accuracy: {best_accuracy:.2%}")
    print(f"Average total correct across {num_runs} runs: {avg_correct:.2f}")
    print(f"Results for the best run saved to: {final_path}")