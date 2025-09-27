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
OUTPUT_TEMPLATE = "./results/role_inference/task4_{model_name}.json"
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

def prepare_game_rounds(data: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    准备单局游戏数据：格式化，并按轮次切分文本。
    返回游戏初始信息和轮次文本列表。
    """
    my_info_lines = ["[My Info]"]
    my_info_data = data.get("my_info", {}).get("1", {})
    my_info_lines.append(f"  玩家={my_info_data.get('player', '')}, 身份={my_info_data.get('role', '')}, 信息={my_info_data.get('msg', '')}")
    my_info_text = "\n".join(my_info_lines)

    round_texts = []
    task_keys = sorted([k for k in data.keys() if k.startswith("task") and k != "task6"])
    for task_key in task_keys:
        task_data = data[task_key]
        round_lines = [f"=== {task_key} / Round {task_data.get('player_message', [{}])[0].get('round', 'N/A')} ==="]
        
        # 处理 player_message
        for pm in task_data.get("player_message", []):
            round_lines.append(f"  - [Player={pm.get('player', 'system')}] says: {pm.get('msg', '')}")
        
        # 处理 strategy
        for st in task_data.get("strategy", []):
            # BUG FIX: 修复了原代码中s_msg变量作用域错误的问题
            round_lines.append(f"  - Strategy: {st.get('msg', '')}")
            
        round_texts.append("\n".join(round_lines))
        
    return my_info_text, round_texts

def generate_reasoning_profile(llm: LLM, tokenizer: AutoTokenizer, observer_texts: List[str]) -> str:
    """
    Stage 1: 生成玩家推理画像。
    """
    print("Stage 1: Generating reasoning profile from observer games...")
    system_prompt = "你是一个擅长分析阿瓦隆游戏的大模型。你的任务是帮我总结我的推理风格，进而推测复盘策略，只需要产出综合描述。"
    accumulated_summary = ""
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)

    for i, ref_text in enumerate(observer_texts, 1):
        print(f"  Analyzing observer game {i}/{len(observer_texts)}...")
        user_prompt = f"""我旁观某位玩家的游戏过程，给出自己的分析和复盘，
{ref_text}
此前你已经帮我总结了：我旁观时的推理风格和人物特点以及参与游戏时可能的复盘策略和内容
{accumulated_summary}
请你基于已有总结，结合新的参考游戏数据，生成一个**新的综合总结**："""
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt_text], sampling_params)
        accumulated_summary = outputs[0].outputs[0].text # 每一轮都用最新的完整总结覆盖
        
    print("Profile generation complete.\n")
    return accumulated_summary

def infer_roles_for_round(llm: LLM, tokenizer: AutoTokenizer, game_summary: str, current_round_text: str) -> str:
    """
    Stage 2: 动态推断角色，让模型总结内容并预测当前轮次的角色分布。
    """
    # System Prompt 已与论文完全对齐
    system_prompt = """你是一个阿瓦隆游戏角色推断专家。你熟悉阿瓦隆游戏的所有角色和规则，并能够根据给定信息准确推断玩家角色。请根据后续提供的游戏数据，输出6位玩家的角色，要求如下：
- 1名[梅林]
- 1名[莫甘娜]
- 1名[派西维尔]
- 1名[刺客]
- 2名[忠臣]

请严格按照以下格式输出，不要添加其他任何内容或说明：
=== Roles ===
Player 1: [角色]
Player 2: [角色]
Player 3: [角色]
Player 4: [角色]
Player 5: [角色]
Player 6: [角色]"""

    # User Prompt 已与论文完全对齐
    user_prompt = f"""下面我会逐步地向你提供一局我实际参与的新的游戏数据，以下是你对此前游戏数据的总结，
{game_summary if game_summary else "（这是游戏的第一轮，暂无历史总结）"}

下面是新的部分游戏数据，
{current_round_text}

你有两个任务：
1. 请你总结全部的游戏数据，包括关键的玩家发言和游戏状态等，以 === Content === 开头。
2. 请你根据玩家和系统发言，结合我的角色信息（代入我的角色视角进行分析），输出每位玩家可能的角色，以 === Roles === 开头。

格式要求如下：
Player 1: [角色]
Player 2: [角色]
Player 3: [角色]
Player 4: [角色]
Player 5: [角色]
Player 6: [角色]

****注意两个任务一定要分别以=== Content ===和=== Roles ===开头。****"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=2048)
    outputs = llm.generate([prompt_text], sampling_params)
    return outputs[0].outputs[0].text

def compare_roles(ground_truth_list: List[Tuple[str, str]], model_prediction: str) -> int:
    """对比正确角色与模型预测角色，进行模糊匹配并返回正确数量。"""
    correct_count = 0
    truth_dict = {str(player).strip(): role.strip() for player, role in ground_truth_list}

    for line in model_prediction.splitlines():
        match = re.match(r"Player\s*(\d+)\s*:\s*(.+)", line.strip())
        if match:
            player_num = match.group(1).strip()
            predicted_role = match.group(2).strip().strip("[]").strip()
            predicted_role = re.sub(r'\s+', '', predicted_role)
            
            # 模糊匹配：刺客 -> 刀客
            if predicted_role == "刺客":
                predicted_role = "刀客"
                
            correct_role = re.sub(r'\s+', '', truth_dict.get(player_num, ""))
            if predicted_role == correct_role:
                correct_count += 1
                
    return correct_count

def run_inference_for_file(data: Dict, llm: LLM, tokenizer: AutoTokenizer) -> Tuple[float, List[str], str]:
    """对单个文件执行完整的逐轮角色推断评估。"""
    my_info_text, game_rounds = prepare_game_rounds(data)
    ground_truth_roles = [(info.get("player"), info.get("role")) for _, info in data.get("user_info", {}).items()]
    
    game_summary_from_llm = ""
    all_role_predictions = []
    full_model_responses = []

    for round_text in game_rounds:
        prompt_input = f"{my_info_text}\n{round_text}"
        model_response = infer_roles_for_round(llm, tokenizer, game_summary_from_llm, prompt_input)
        full_model_responses.append(model_response)
        
        roles_part = ""
        if "=== Roles ===" in model_response:
            parts = model_response.split("=== Roles ===", 1)
            game_summary_from_llm = parts[0].replace("=== Content ===", "").strip()
            roles_part = parts[1].strip()
            all_role_predictions.append(roles_part)

    # 加权评分
    num_rounds = len(all_role_predictions)
    weights = []
    if num_rounds == 1: weights = [1.0]
    elif num_rounds == 2: weights = [0.4, 0.6]
    elif num_rounds == 3: weights = [0.2, 0.4, 0.4]
    elif num_rounds == 4: weights = [0.1, 0.25, 0.35, 0.3]
    elif num_rounds == 5: weights = [0.1, 0.25, 0.3, 0.25, 0.1]
    else: weights = [1.0 / num_rounds] * num_rounds if num_rounds > 0 else []

    total_weighted_score = 0.0
    for i, prediction in enumerate(all_role_predictions):
        round_correct = compare_roles(ground_truth_roles, prediction)
        # 每轮满分为6分，先算得分率，再乘以权重
        round_score = (round_correct / 6.0) * weights[i]
        total_weighted_score += round_score
        
    return total_weighted_score, all_role_predictions, "\n---\n".join(full_model_responses)

# ======== Main 函数 ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 4: Role Inference.")
    parser.add_argument('model_path', type=str, help="Path to the Hugging Face model directory.")
    args = parser.parse_args()

    MODEL_NAME_OR_PATH = args.model_path
    model_name = os.path.basename(MODEL_NAME_OR_PATH)

    print(f"Loading model: {model_name}")
    llm = LLM(model=MODEL_NAME_OR_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    # Stage 1: 画像生成（仅需一次）
    observer_files = sorted([f for f in os.listdir(OBSERVER_DIR) if f.endswith(".json")])
    observer_texts = [json.load(open(os.path.join(OBSERVER_DIR, f), 'r', encoding='utf-8')) for f in observer_files]
    observer_texts_formatted = [convert_json_to_text_format(data, i+1) for i, data in enumerate(observer_texts)]
    # profile = generate_reasoning_profile(llm, tokenizer, observer_texts_formatted) # Task 4 prompt中未明确使用profile，暂不生成

    # Stage 2: 角色推断
    all_file_results = []
    total_scores = 0
    file_count = 0
    
    participant_files = sorted(os.listdir(PARTICIPANT_DIR))
    for filename in participant_files:
        if filename.endswith(".json"):
            print(f"  Processing file: {filename}...")
            filepath = os.path.join(PARTICIPANT_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            score, predictions, response = run_inference_for_file(data, llm, tokenizer)
            
            all_file_results.append({
                "file_name": filename,
                "ground_truth_roles": data.get("user_info", {}),
                "model_predictions_by_round": predictions,
                "total_weighted_score": score,
                "model_responses_raw": response
            })
            total_scores += score
            file_count += 1
            
    # 计算平均分并保存
    average_score = total_scores / file_count if file_count > 0 else 0
    final_output_data = {
        "average_weighted_score": average_score,
        "results_per_file": all_file_results
    }
    
    output_path = OUTPUT_TEMPLATE.format(model_name=model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output_data, f, ensure_ascii=False, indent=4)
        
    # 根据平均分重命名文件
    new_output_path = os.path.join(os.path.dirname(output_path), f"avg_score_{average_score:.4f}_{os.path.basename(output_path)}")
    try:
        os.rename(output_path, new_output_path)
        final_path = new_output_path
    except OSError as e:
        print(f"Error renaming file: {e}")
        final_path = output_path
        
    print("\n--- Evaluation Summary ---")
    print(f"Processed {file_count} files.")
    print(f"Average weighted score across all files: {average_score:.4f}")
    print(f"Detailed results saved to: {final_path}")