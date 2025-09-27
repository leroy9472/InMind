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
# 输出模板：将在 ./results/reflection_alignment/ 目录下生成结果
OUTPUT_TEMPLATE = "./results/reflection_alignment/task2_{model_name}.json"
TENSOR_PARALLEL_SIZE = 4

# ======== 核心函数 (Core Functions) ========

def convert_json_to_text_format(json_data: Dict[str, Any], game_id: int) -> str:
    """
    (复用自Task1) 将 Observer-mode 的 JSON 数据动态转换为指定的 TXT 格式字符串。
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

def format_participant_game_prompt(data: Dict[str, Any]) -> Tuple[str, str, list]:
    """
    构建并格式化Participant模式的游戏数据，同时对my_review进行mask处理。
    """
    lines = []
    
    # 1. My Info
    lines.append("My Info:")
    my_info = data.get("my_info", {}).get("1", {})
    lines.append(f"  - 你的初始身份: {my_info.get('role', 'N/A')}, 你的玩家编号: {my_info.get('player', 'N/A')}")

    # 2. Tasks Overview (包含 strategy)
    lines.append("\nTasks Overview (Round by Round):")
    task_keys = sorted([k for k in data.keys() if k.startswith("task")])
    for task_key in task_keys:
        task_data = data[task_key]
        round_num = task_data.get("player_message", [{}])[0].get("round", "N/A")
        lines.append(f"  --- 回合 {round_num} ---")
        lines.append("    player_message:")
        for pm in task_data.get("player_message", []):
            lines.append(f"      - [玩家{pm.get('player', 'system')}]: {pm.get('msg', '')}")
        
        # 根据论文描述，补充strategy部分
        strategy_list = task_data.get("strategy", [])
        if strategy_list:
            lines.append("    strategy:")
            for st in strategy_list:
                lines.append(f"      - {st.get('msg', '')}")
    
    # 3. Final User Info
    lines.append("\nFinal User Info:")
    for pid, info in sorted(data.get("user_info", {}).items()):
        lines.append(f"  - 玩家{pid}的真实身份: {info.get('role', 'N/A')}")

    # 4. Mask 'my_review'
    review_text = data.get("my_review", {}).get("msg", "")
    masked_review, replacements = mask_player_numbers(review_text)
    
    return "\n".join(lines), masked_review, replacements

def mask_player_numbers(text: str) -> Tuple[str, list]:
    """对文本中的玩家编号(1-6)进行mask，并记录替换关系。"""
    pattern = re.compile(r'\b[1-6]\b') # 使用\b确保匹配的是独立的数字
    replacements = []
    mask_index = 1
    
    def replacer(match: re.Match):
        nonlocal mask_index
        original_number = match.group(0)
        # 统一使用 (1digit) 格式
        mask_tag = f"[MASK_{mask_index}(1digit)]"
        replacements.append(f"{mask_tag} => {original_number}")
        mask_index += 1
        return mask_tag

    masked_text = pattern.sub(replacer, text)
    return masked_text, replacements

def generate_reasoning_profile(llm: LLM, tokenizer: AutoTokenizer, observer_texts: List[str]) -> str:
    """
    Stage 1: 通过分析Observer模式的游戏文本，生成玩家推理画像。
    """
    print("Stage 1: Generating reasoning profile from observer games...")
    system_prompt = "你是一名精通阿瓦隆桌游的分析专家，擅长透过玩家的行为和发言推测玩家的推理逻辑、人物风格和行动策略。你的任务是基于用户提供的旁观游戏记录，精准地总结用户在旁观时展现的推理风格、发言倾向以及分析局势的方法，为之后推测用户实际参与游戏后的复盘内容建立准确的用户画像。"
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

要求精准、细致地进行归纳总结，以便下一步准确推测实际游戏后我可能的复盘思路和内容。"""
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt_text], sampling_params)
        accumulated_summary = outputs[0].outputs[0].text
        
    print("Profile generation complete.\n")
    return accumulated_summary

def align_reflection(llm: LLM, tokenizer: AutoTokenizer, reasoning_profile: str, game_prompt: str, masked_review: str) -> str:
    """
    Stage 2: 应用画像，对mask后的复盘文本进行填充。
    """
    # System Prompt 已与论文完全对齐
    system_prompt = """你是一个擅长分析阿瓦隆游戏的大模型。
根据以下提示，你的任务是为我的复盘（my_review）中的每个 [MASK_x(...)] 填充数字或数字组合，并严格遵守以下格式要求：
1. 对于 [MASK_x(1digit)]，请用单个数字表示该位置的玩家编号。
2. 对于 [MASK_x(Ndigits)]，请用多个数字组成一个数字组合，数字按升序排列，表示多位玩家标号。
3. 你需要完整地为每个 [MASK_x(...)] 提供一个对应的数字或数字组合映射行。
4. 不要输出多余的映射行，不要输出多余的其他解释或文字。

格式要求：
[MASK_x(1digit)] => y
[MASK_x(Ndigits)] => yz"""

    # User Prompt 已与论文完全对齐
    user_prompt = f"""此前你已经帮我总结了：我旁观时的推理风格和人物特点以及参与游戏时可能的复盘内容，
{reasoning_profile}

以下是一局我参与的游戏数据{game_prompt}：

你的***唯一任务***：根据总结和游戏数据，推测每个我对整局游戏的复盘内容中被MASK的玩家编号，并输出映射行。
- [My Review]：{masked_review}

***要求***：
1. 每个 [MASK_x(...)] 都需要填写 1~6 之间的数字编号，按升序排列。
2. 格式要求严格：[MASK_x(1digit)] => y 或 [MASK_x(Ndigits)] => yz
3. 不要输出多余的映射行，仅填写实际需要的映射行，不要解释。"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.2, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)
    outputs = llm.generate([prompt_text], sampling_params)
    return outputs[0].outputs[0].text

def compare_predictions(ground_truth: list, model_output: str) -> Tuple[int, int]:
    """
    比对模型输出和真实答案，返回正确数量和总mask数量。
    """
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

def run_full_process(directory: str, llm: LLM, tokenizer: AutoTokenizer, observer_texts: List[str]) -> Tuple[int, int, list]:
    """执行一次完整的数据处理和评估流程。"""
    total_correct = 0
    total_mask = 0
    all_results = []
    
    # Stage 1 在循环外执行，画像只需生成一次
    reasoning_profile = generate_reasoning_profile(llm, tokenizer, observer_texts)
    
    print("Stage 2: Starting Reflection Alignment task...")
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            json_file_path = os.path.join(directory, filename)
            print(f"  Processing file: {filename}...")
            
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            game_prompt, masked_review, ground_truth = format_participant_game_prompt(data)
            
            if not ground_truth:  # 如果没有需要mask的内容，则跳过
                continue
            
            model_prediction = align_reflection(llm, tokenizer, reasoning_profile, game_prompt, masked_review)
            correct_count, mask_count = compare_predictions(ground_truth, model_prediction)
            
            total_correct += correct_count
            total_mask += mask_count
            
            all_results.append({
                "file_name": filename,
                "ground_truth_masks": ground_truth,
                "model_prediction_raw": model_prediction,
                "correct_count": correct_count,
                "total_masks_in_file": mask_count,
            })
            
    final_summary = {
        "total_correct": total_correct,
        "total_mask": total_mask,
        "accuracy": (total_correct / total_mask) if total_mask > 0 else 0,
        "reasoning_profile": reasoning_profile,
    }
    all_results.append(final_summary)
    
    return total_correct, total_mask, all_results

# ======== Main 函数 ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2: Reflection Alignment.")
    parser.add_argument('model_path', type=str, help="Path to the Hugging Face model directory.")
    args = parser.parse_args()

    MODEL_NAME_OR_PATH = args.model_path
    model_name = os.path.basename(MODEL_NAME_OR_PATH)

    print(f"Loading model: {model_name}")
    llm = LLM(model=MODEL_NAME_OR_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    # 准备Observer模式数据
    observer_files = sorted([f for f in os.listdir(OBSERVER_DIR) if f.endswith(".json")])
    observer_game_texts = []
    for i, filename in enumerate(observer_files, 1):
        filepath = os.path.join(OBSERVER_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        observer_game_texts.append(convert_json_to_text_format(json_data, i))
    
    # 运行5次评估
    run_results = []
    num_runs = 5
    for i in range(num_runs):
        print(f"\n--- Starting Run {i+1}/{num_runs} ---")
        total_correct, total_mask, results = run_full_process(PARTICIPANT_DIR, llm, tokenizer, observer_game_texts)
        run_results.append({
            "run": i + 1,
            "total_correct": total_correct,
            "total_mask": total_mask,
            "results_data": results
        })

    # 计算平均分并找到最佳运行
    if run_results:
        avg_correct = sum(run["total_correct"] for run in run_results) / num_runs
        best_run = max(run_results, key=lambda x: x["total_correct"])
        best_results_data = best_run["results_data"]
    else:
        avg_correct = 0
        best_run = {"total_correct": 0, "total_mask": 0}
        best_results_data = []

    # 保存最佳运行的结果
    output_path = OUTPUT_TEMPLATE.format(model_name=model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_results_data, f, ensure_ascii=False, indent=4)
    
    # 根据平均分重命名文件
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