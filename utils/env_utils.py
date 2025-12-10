import re
import gym
from utils.utils import get_value

from utils.llm_utils import LLM
import utils.utils as utils
import utils.babyai_utils as babyai_utils
from utils.data_utils import SubgoalTree

from gym_minigrid.minigrid import (
    IDX_TO_COLOR,
)

# 環境の読み込みやテキスト化, プロンプトの定義などを行う
# その他, 環境に関するいろいろな機能

CLIFF = "Cliff" # CliffWalking環境
BABY = "Baby" # BabyAI環境

# configから環境を作成する
def make(params:dict) -> gym.Env:
    env_name = params["env_name"]
    if CLIFF in env_name:
        env = gym.make(env_name, render_mode="rgb_array")
    elif BABY in env_name:
        env = gym.make(env_name, agent_num = params["agent_num"])

    env.agent_num = params["agent_num"]
    return env

# エージェントの名前を返す
def get_agent_name(agent_id:int, params:dict = {}) -> str:
    env_name = params["env_name"]
    if CLIFF in env_name:
        return f"agent{agent_id}" if agent_id >= 0 else ""
    elif BABY in env_name:
        return babyai_utils.get_agent_name(agent_id)

# 行動の名前の取得
def get_actions_str(env_name:str) -> list[str]:
    if CLIFF in env_name:
        return ["move north","move east","move south","move west"]
    elif BABY in env_name:
        # 詳しく書いた方が良い?
        #return ["turn left","turn right","go forward","pick up","drop","open"]
        return ["turn left","turn right","go to the forward coordinate","pick up item that in the forward coordinate","drop carrying item to the forward coordinate","open the door at the forward coordinate"]

# 行動の名前の略称を取得
def get_brief_actions_str(env_name:str) -> list[str]:
    if CLIFF in env_name:
        return ["north","east","south","west"]
    elif BABY in env_name:
        return ["left","right","forward","pick up","drop","open"]

# 行動の名前をカンマ区切りで連結したものを取得(プロンプト用)
def get_actions_joined_str(env_name:str, last_word:str="") -> str:
    actions_str = get_actions_str(env_name)
    actions_str = [f"'{action}'" for action in actions_str]
    if last_word == "" or len(actions_str) <= 1:
        actions_joined = ", ".join(actions_str)
    else:
        actions_joined = ", ".join(actions_str[:-1]) + f" {last_word} " + actions_str[-1]
    return actions_joined

# プロンプトに使用する問題とタスクの説明文を返す
def get_explain(env:gym.Env, obs, params:dict) -> tuple[str,list[str]]:
    env_name = params["env_name"]
    actions_joined = get_actions_joined_str(env_name, "or")
    action_prompt = f"Each step, You must select actions in {actions_joined}."
    if CLIFF in env_name:
        sentences = []
        sentences.append("Interact with an grid world environment to solve a task.")
        sentences.append(action_prompt)
        base_prompt = " ".join(sentences)
        task_prompts = ["Reach the goal as soon as possible."] * env.agent_num
    elif BABY in env_name:
        right = env.width - 1
        bottom = env.height - 1
        sentences = []
        sentences.append("Interact with an grid world environment to solve a task.")

        # additional
        sentences.append(f"The grid (0,0) is at the northwest end, ({right}, 0) is northeast, (0, {bottom}) is southwest, ({right}, {bottom}) is southeast.")
        sentences.append("Note that You cannot move to coordinate there is an object.")
        sentences.append("You cannot pick up any item if you have already item.")
        sentences.append(action_prompt)
        if env.agent_num >= 2:
            sentences.append(f"Work with other agents to accomplish tasks.")

        base_prompt = utils.join_sentences(sentences)
        task_prompts =  [obs[i]['mission'] for i in range(env.agent_num)]

    return base_prompt, task_prompts

def get_image_explain(agent_id:int, params:dict) -> str:
    is_use_vision = get_value(params,"is_use_vision",False)
    if not is_use_vision: return ""

    image_token = LLM.image_token
    env_name = params["env_name"]
    if CLIFF in env_name:
        return f"{image_token} This image is your view of grid world. You are the character with the hat."
    elif BABY in env_name:
        color = IDX_TO_COLOR[agent_id]
        return f"{image_token} This image is your view of grid world. You can observe only your forward. You are {color} triangle. "
    return f"{image_token}"

# 通常のReflexionを行う時のプロンプトに記述する指示文を返す
def get_general_reflexion_instr(reason:str, agent_id:int, params:dict = {}) -> str:
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if params["agent_num"] > 1 else ""
    image_explain = get_image_explain(agent_id, params)
    prompt = profile + image_explain
    prompt += f"You are given the history of a past experience in which you were placed in an environment and given a task to complete. In this trial, you were unsuccessful in completing the task because of {reason}."
    prompt += " " + f"Do not summarize your environment, but rather think about the strategy to complete the task, considering actions that are useful and those that should be improved. Output a detailed new plan from the start of the task to its accomplishment in 3 to 6 sentences \nGive only your plan:"
    return prompt

# Subgoalに関するReflexionを行う時のプロンプトに記述する指示文を返す
def get_subgoal_reflexion_instr(reason:str, agent_id:int, subgoal_tree:SubgoalTree, params:dict = {}) -> str:
    achieved, failed = subgoal_tree.get_all_sequence()
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if params["agent_num"] > 1 else ""
    image_explain = get_image_explain(agent_id, params)
    next_trial = params["reflexion_memory_size"]+1
    prompt = profile + image_explain
    prompt += f"You are given the history of a past experience in which you were placed in an environment and given a task to complete. In this trial, you were unsuccessful in completing the task because of {reason}."
    prompt += " " + f"You planned and accomplished the following subgoal list {achieved}."
    prompt += " " + f"However, You were not able to achieve the following subgoal list {failed}."
    prompt += " " + f"Do not summarize your environment, but rather think about the subgoal planning strategy to complete the task, considering subgoals that are useful and those that should be improved. Output a detailed new plan from the start of the task to its accomplishment in 3 to 6 sentences \nGive only your plan:"
    return prompt

# グループ反省会でのまとめ用のし自分を返す
def get_group_reflexion_summary_instr(chat_history: list, reason: str, params: dict = {}) -> str:
    chat_str = ""
    for name, text in chat_history:
        chat_str += f"{name}: {text}\n"
    
    # プロンプトの構築
    prompt = f"""
    The team failed to compete the task because: {reason}.
    Based on the following team discussion, summarize the final concrete strategy for the next episode.
    
    Discussion:
    {chat_str}
    
    Output only the summary of the strategy in 3 to 6 sentences.
    """

# Subgoalに関するReflexionを行う時のプロンプトに記述する達成/未達成のサブゴールを返す
def get_subgoal_reflexion_achievement(subgoal_tree:SubgoalTree, params:dict={}) -> str:
    achieved, failed = subgoal_tree.get_all_sequence()
    result = f"I planned and accomplished the following subgoal list {achieved}, "
    result += f"but was not able to achieve the following subgoal list {failed}."
    return result

# 行動を生成する時のプロンプトに記述する指示文を返す
def get_action_instr(env_name:str, agent_id:int, params:dict = {}) -> str:
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if params["agent_num"] > 1 else ""
    image_explain = get_image_explain(agent_id, params)
    actions_joined = get_actions_joined_str(env_name, "or")
    prompt = profile + image_explain + f"What is the best action to achieve your task in {actions_joined}? Output only result.\nYour action:"
    return prompt

# メッセージを生成する時のプロンプトに記述する指示文を返す
def get_message_instr(agent_id:int, targets:str, params:dict={}) -> str:
    targets_str = " and ".join(targets)
    agent_name = get_agent_name(agent_id, params)
    
    sentences = []
    image_explain = get_image_explain(agent_id, params)
    if len(image_explain) > 0: sentences.append(image_explain)
    sentences.append(f"You are {agent_name}.")

    sentences.append(f'You are in a cooperative relationship with {targets_str}.')
    sentences.append(f'To agents split up and accomplish task efficiently, output only message to {targets_str} in 2 to 5 sentences.')
    sentences.append(f"\nYour reply to {targets_str}:")
    prompt = " ".join(sentences)
    return prompt

# メッセージを生成する時のプロンプトに記述する指示文を返す
def get_conversation_instr(agent_id:int, targets:str, messages:list[tuple[str,str]], is_last:bool, params:dict={}) -> str:
    targets_str = " and ".join(targets)
    agent_name = get_agent_name(agent_id, params)
    actions_str = get_actions_joined_str(params["env_name"], "or")

    sentences = []
    image_explain = get_image_explain(agent_id, params)
    if len(image_explain) > 0: sentences.append(image_explain)
    sentences.append(f"You are {agent_name}.")
    sentences.append(f'You are in a cooperative relationship with {targets_str}, and discussing to decide next action.')
    sentences.append(f'The mission has not yet been accomplished.')
    sentences.append(f'Either one of them should accomplish the task.')
    if is_last: sentences.append(f'This is the last turn in the conversation.')
    sentences.append(f'To accomplish tasks efficiently, output message to them in 1 to 3 sentences. Exchanging your information of sight and your plan. At the end of the conversation, reach a consensus on a plan.')
    sentences.append(f'Note that you can act only after conversation, and only in {actions_str}. Act separately from other agents for efficiency.')
    prompt = " ".join(sentences)
    for name, text in messages:
        prompt += f'\n{name}:{text}'
    prompt += f"\n{agent_name}:"
    return prompt

# Chain of Thoughtでの出力を生成する時のプロンプトに記述する指示文を返す
def get_consideration_instr(agent_id:int, achieved:list[str], not_achieved:list[str], params:dict = {}) -> str:
    agent_name = get_agent_name(agent_id, params)
    image_explain = get_image_explain(agent_id, params)
    sentences = []
    if params["agent_num"] > 1: sentences.append(f"You are {agent_name}.")
    if len(image_explain) > 0: sentences.append(image_explain)                        
    sentences.append(f"Output abstract plan, what you think would accomplish the task in the current situation in 2 to 3 sentences, briefly. Please think step by step")

    prompt = " ".join(sentences)
    prompt += "\nYour think:"
    return prompt

# サブゴールを生成する時のプロンプトに記述する指示文を返す
def get_subgoal_instr(agent_id:int, achieved:list[str], not_achieved:list[str], params:dict = {}) -> str:
    agent_name = get_agent_name(agent_id, params)
    image_explain = get_image_explain(agent_id, params)
    next = not_achieved[0]
    actions = get_actions_joined_str(params["env_name"])
    sentences = []

    if params["agent_num"] > 1: sentences.append(f"You are {agent_name}. ")
    if len(image_explain) > 0: sentences.append(image_explain)
    
    sentences.append(f"You already achieved subgoals {achieved}.")
    sentences.append(f"You think you should achieve subgoals {not_achieved} in order.")
    
    if len(not_achieved) > 1:
        next_next = not_achieved[1]
        sentences.append(f"Output more concrete subgoal to achieve '{next}' for achieving {next_next}.")
    else:
        sentences.append(f"Output more concrete subgoal to achieve '{next}'.")
    sentences.append(f"Don't output same meaning subgoal and subgoal that is not able to be achieved.")
    sentences.append(f"Don't use relative expressions for example, 'move right', 'move east', 'go to left'.")
    sentences.append(f"Considering other subgoals that you should achieve, don't output wasteful subgoal.")
    sentences.append(f"If any of your actions {actions} can achieve '{next}', output the action name as a subgoal.")
    sentences.append(f"Output very appropriate subgoal in a few words, only one subgoal.")
    prompt = " ".join(sentences)
    prompt += f"\nYour subgoal:"
    return prompt

# サブゴールを行動に変換する時のプロンプトに記述する指示文を返す
def get_subgoal_to_action_instr(env_name:str, agent_id:int, subgoals:list[str], params:dict = {}) -> str:
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if params["agent_num"] > 1 else ""
    image_explain = get_image_explain(agent_id, params)
    actions_joined = get_actions_joined_str(env_name, "or")
    prompt = profile + image_explain + f"You think you should achieve {subgoals}. To achieve '{subgoals[0]}', what is the best action in {actions_joined}? Output only action name.\nYour action:"
    return prompt

# サブゴールを達成したか判定する時のプロンプトに記述する指示文を返す
def get_subgoal_achieved_instr(env_name:str, agent_id:int, subgoals:list[str], params:dict = {}) -> str:
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if params["agent_num"] > 1 else ""
    image_explain = get_image_explain(agent_id, params)
    #prompt = profile + image_explain + f"In the previous step, you thought you should achieve {subgoals}. Is the subgoal '{subgoals[0]}' achieved? Output only 'Yes' or 'No'.\nYes or No:" #  by the previous your action も要るかと思ったがマルチエージェントを考慮して一旦消す
    prompt = profile + image_explain + f"In the previous step, you thought you should achieve {subgoals}. Is the subgoal '{subgoals[0]}' achieved? Output only 'Yes' or 'No'. Note that for subgoal such as 'go to A', you should output 'Yes' when your forward is coordinate A or there is A in your forward coordinate.\nYes or No:" #  by the previous your action も要るかと思ったがマルチエージェントを考慮して一旦消す
    return prompt

# サブゴールを達成したか判定させるプロンプトの指示文を返す
def get_all_subgoals_achieved_instr(agent_id:int, subgoals:list[str], params:dict = {}) -> str:
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if params["agent_num"] > 1 else ""
    image_explain = get_image_explain(agent_id, params)
    prompt = profile + image_explain + f"In the previous step, you thought you should achieve {subgoals}. Was each subgoal achieved? Output only the list of Yes or No, in the form of ['Yes', 'No', ...].\nlist:" #  by the previous your action も要るかと思ったがマルチエージェントを考慮して一旦消す
    return prompt

# 最初のサブゴール列を生成するプロンプトの指示文を返す
def get_init_subgoal_instr(env:gym.Env, mission:str, agent_id:int, params:dict) -> str:
    
    env_name = params["env_name"]
    if CLIFF in env_name:
        return ""
    elif BABY in env_name:
        actions_str = get_actions_joined_str(env_name, "or")
        obs = babyai_utils.world_to_str_baby(env, False, params)
        return f"You interact with an grid world environment to solve a task. This image is environment. {obs} It is not possible to overlap objects. You cannot pick up any item if you have already item. Your mission is '{mission}'. You are red triangle. Each step, you can act in {actions_str}. Output subgoal list to achieve your task in the format ['A', 'B', 'C', 'D', ... '{mission}']. Output list of abstract subgoals with enough length, without action name so much appropriately. The last subgoal of the list is your mission. Output only result.\nsubgoal list:"
    return f""

# ToM推論のプロンプトの指示文を返す
def get_tom_instr(env:gym.Env, agent_id:int, target_agent_id:int, params:dict) -> str:
    env_name = params["env_name"]
    agent_name = get_agent_name(agent_id, params)
    target_name = get_agent_name(target_agent_id, params)
    image_explain = get_image_explain(agent_id, params)
    sentences = []
    
    # プロフィール
    if params["agent_num"] > 1:
        sentences.append(f"You are {agent_name}. ")
    
    # 画像情報（必須ではない）
    if len(image_explain) > 0:
        sentences.append(image_explain)
    
    # Tom推論の指示
    # 状況に応じた推論を指示
    sentences.append(f"Infer the current situation and intent of {target_name}.")
    sentences.append(f"If {target_name} is in your view, verify their action.")
    # 【重要】見えない場合は会話を思い出すよう指示
    sentences.append(f"If {target_name} is NOT in your view, infer their action based on the **past conversation/messages** and the agreed plan.")
    sentences.append(f"Predict what subgoal {target_name} is trying to achieve and what action they will take next.")
    sentences.append("Output the prediction briefly in 2 to 3 sentences.")

    prompt = " ".join(sentences)
    prompt += f"\nPrediction for {target_name}:"
    return prompt

# 構造化通信（JSON形式）用の指示文を返す
def get_structured_conversation_instr(agent_id: int, targets: list[str], chat_history: list, is_last: bool, params: dict) -> str:
    agent_name = get_agent_name(agent_id, params)
    targets_str = ", ".join(targets)
    
    # 履歴をフォーマット (JSON文字列として履歴を渡す)
    history_text = ""
    for name, msg in chat_history:
        # msg は既にJSON文字列であることを想定、あるいはテキスト
        history_text += f"{name}: {msg}\n"

    # 終了間際なら合意を促す文言を追加
    last_instruction = ""
    if is_last:
        last_instruction = "This is the last turn. You MUST reach a consensus and output 'AGREE' as intent."

    prompt = f"""
    You are {agent_name}. You are communicating with {targets_str} to solve the task.
    
    You MUST output your response in valid JSON format ONLY. Do not output any other text.
    
    Required JSON Schema:
    {{
        "intent": "Select one from [PROPOSE, INFORM, REQUEST, AGREE, REJECT]",
        "target_object": "Object name (e.g., 'red key') or null",
        "target_coordinate": [x, y] coordinates as a list of integers, or null",
        "action_plan": "Brief description of your planned action",
        "message": "A short natural language message to your partner"
    }}
    
    Conversation History:
    {history_text}
    
    {last_instruction}
    Based on your observation and the history, decide your intent and generate a JSON response.
    """
    return prompt

# 終了判定や状態の評価を返す
def get_achievement_status(reward:int, done:bool, step:int, params:dict) -> tuple[bool,bool,str]:
    env_name = params["env_name"]
    max_step = params["max_step"]
    
    reason = ""
    is_success = False
    if CLIFF in env_name:
        if reward < -10:
            reason = "falling to the cliff"
            done = True
        elif done:
            reason = "reaching the goal"
            is_success = True
        elif step == max_step-1:
            reason = "running out of time"
            done = True
        return done, is_success, reason
    elif BABY in env_name:
        if done:
            reason = "achieve the mission"
            is_success = True
        elif step == max_step-1:
            reason = "running out of time"
            done = True
            is_success = False
        return done, is_success, reason

    return False, False, ""

# 現在のエージェントの視界の画像を取得
def get_imgs(env:gym.Env, params:dict):
    is_use_vision = utils.get_value(params,"is_use_vision",False)
    imgs = env.render_masked() if is_use_vision else [None] * env.agent_num
    return imgs

# 観測情報を文字列に変換する
def obs_to_str(env:gym.Env, observation, params:dict) -> list:
    env_name = params["env_name"]
    if CLIFF in env_name:
        return obs_to_str_cliff(observation)
    elif BABY in env_name:
        return babyai_utils.obs_to_str_baby(env, observation, params)

# CliffWalking環境の観測の変換処理
def obs_to_str_cliff(obs:int) -> list[str]:
    object_name = ["Wall", "Nothing", "Cliff", "Goal"]
    field = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,2,2,2,2,2,2,2,2,2,2,3,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
    r = obs // 12 + 1
    c = obs % 12 + 1
    north = f"Your north is {object_name[field[r-1][c]]}."
    south = f"Your south is {object_name[field[r+1][c]]}."
    east = f"Your east is {object_name[field[r][c+1]]}."
    west = f"Your west is {object_name[field[r][c-1]]}."
    row_s = "s" if 4 - r > 1 else ""
    col_s = "s" if 12 - c > 1 else ""
    if 12 - c == 0:
        goal = f"Goal is in {4 - r} step{row_s} to your south."
    elif 4 - r == 0:
        goal = f"Goal is in {12 - c} step{col_s} to your east."
    else:
        goal = f"Goal is in {12 - c} step{col_s} to your east and {4 - r} step{row_s} to your south."

    return [f"{north} {south} {east} {west} {goal}"]

# 全エージェントの行動のフィードバックをテキストで取得
def get_feedbacks(env, observations, actions:list[int], params:dict={}) -> list[str]:
    env_name = params["env_name"]
    if CLIFF in env_name:
        return [""] * env.agent_num
    elif BABY in env_name:
        return babyai_utils.get_feedbacks(env, observations, actions, params)

# 文字列を行動IDに変換する
def str_to_action(text:str, params:dict) -> int:
    env_name = params["env_name"]
    text = text.lower()
    # 正式な行動名とマッチさせる
    actions_str = get_actions_str(env_name)
    action_to_idx = {actions_str[i]:i for i in range(len(actions_str))}
    for action, value in action_to_idx.items():
        match = re.compile(action.lower()).search(text)
        if match: return value

    # 簡易的な行動名とマッチさせる
    brief_actions_str = get_actions_str(env_name)
    brief_action_to_idx = {actions_str[i]:i for i in range(len(brief_actions_str))}
    for action, value in brief_action_to_idx.items():
        match = re.compile(action.lower()).search(text)
        if match: return value

    return 0
      
# 行動IDを文字列に変換する
def action_to_str(action_idx:int, params:dict) -> str:
    env_name = params["env_name"]
    actions = get_actions_str(env_name)
    return actions[action_idx]