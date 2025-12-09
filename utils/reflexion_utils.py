import utils.env_utils as env_utils
from utils.llm_utils import LLM
import utils.utils as utils
from utils.data_utils import History, SubgoalTree, Memory

# Reflexionに関する全般処理を行うクラス
class Reflexion:
    def __init__(self, env, obs, params:dict):
        self.env_name = params["env_name"]
        self.agent_num = params["agent_num"]
        self.histories: list[History] = []
        self.subgoal_trees: list[SubgoalTree] = []
        self.tasks:list[str] = []

        memory_size = utils.get_value(params, "reflexion_memory_size", 0)
        self.memories: list[Memory] = [Memory(memory_size) for _ in range(self.agent_num)]
        self.reset(env, obs, params)

    # エピソードが終わった時に反省文だけ引き継いで履歴をリセットする処理
    def reset(self, env, obs, params:dict):
        self.env = env
        base, tasks = env_utils.get_explain(env, obs, params)
        self.tasks = tasks
        self.histories.clear()
        self.subgoal_trees.clear()
        for i in range(self.agent_num):
            self.histories.append(History(base, tasks[i], self.memories[i].contents))
            self.subgoal_trees.append(SubgoalTree(tasks[i]))

    # 履歴を追加
    def add_histories(self, label:str, contents):
        if not isinstance(contents, list):
            contents = [contents] * self.agent_num
        for i in range(self.agent_num):
            self.histories[i].add(label, contents[i])

    # メッセージを履歴に追加
    def add_message(self, agent_id:int, sender_name:str, content:str):
        self.histories[agent_id].add("message", f"{sender_name}:{content}")

    # 結果を履歴に追加
    def add_result(self, is_success:bool):
        result = "SUCCESS" if is_success else f"FALIED"
        self.add_histories("result", result)
        if is_success: return
        for tree in self.subgoal_trees:
            tree.append_failed_node()

    # 現在のサブゴールを履歴に追加
    def add_now_subgoal(self):
        subgoals_list = []
        for agent_id in range(self.agent_num):
            subgoals = self.subgoal_trees[agent_id].get_subgoals()
            subgoals_list.append(subgoals)
        self.add_histories("subgoal", subgoals_list)
    
    # 特定のラベルを履歴から消去
    def remove_label(self, label:str):
        for agent_id in range(self.agent_num):
            self.histories[agent_id].remove(label)

    # Reflexionのメイン処理
    def run(self, is_success: bool, reason: str, params:dict):
        queries = []
        reflexion_type =  utils.get_value(params, "reflexion_type", "general")
        reflexion_mode = utils.get_value(params, "reflexion_mode", "individual") # 新規パラメータ
        
        # 成功時または "none"の場合は反省しない
        if is_success or reflexion_type == "none": return queries

        imgs = env_utils.get_imgs(self.env, params)
        # 個別に反省文を出力するかグループで反省会を開くかの分岐を実装する
        if reflexion_mode == "individual":
            # 各エージェントが反省文を出力
            for agent_id in range(self.agent_num):
                prompt = self.get_reflexion_prompt(agent_id, reflexion_type, reason, params)
                text, _ = LLM.generate(prompt, imgs[agent_id])
                self.memories[agent_id].add_memory(text)
                queries.append(prompt)
        elif reflexion_mode == "group":
            # 全員の行動履歴を準備
            all_history = ""
            for i in range(self.agent_num):
                h = self.get_history_str(i, True, params)
                all_history += f"\n--- Agent {i}'s Perspective ---\n{h}\n"
            
            # 議論フェーズ
            chat_history = []
            discussion_rounds = utils.get_value(params, "reflexion_discussion_rounds", 4)
            
            for _ in range(discussion_rounds):
                for agent_id in range(self.agent_num):
                    instr = env_utils.get_group_reflexion_instr(agent_id, reflexion_type, reason, params)
                    prompt = f"{all_history}\n\n{instr}"
                    
                    text, _ = LLM.generate(prompt, imgs[agent_id])
                    # 履歴に追加
                    name = env_utils.get_agent_name(agent_id, params)
                    chat_history.append( (name, text) )
                    queries.append(prompt)
            summary_instr = env_utils.get_group_reflexion_summary_instr(chat_history, reason, params)
            summary_prompt = summary_instr # 追加で必要な履歴などを結合できる
            final_strategy, _ = LLM.generate(summary_prompt, imgs[0])
            for agent_id in range(self.agent_num):
                self.memories[agent_id].add_memory(f"Team Strategy: {final_strategy}")

        return queries
    
    # サブゴールを初期化する
    def init_subgoal(self, log:dict, params:dict):
        if log == None:
            return self.get_initial_subgoals_from_llm(params)
        else:
            return self.get_initial_subgoals_from_log(log, params)
    
    # 最初のサブゴール列を取得する
    def get_initial_subgoals_from_llm(self, params:dict):
        img = self.env.render_no_highlight()
        info = {
            "queries":[],
            "responses":[],
            "lists":[]
        }

        for agent_id in range(self.agent_num):
            prompt = self.get_init_subgoal_prompt(agent_id, params)
            text, _ = LLM.generate_high(prompt, img)
            subgoals = utils.text_to_str_list(text)
            if len(subgoals) >= 1:
                subgoals = subgoals[:-1]
                subgoals.reverse()
                for subgoal in subgoals:
                    self.subgoal_trees[agent_id].append(subgoal.lower())
            info["queries"].append(prompt)
            info["responses"].append(text)
            info["lists"].append(subgoals)
        return info

    # 最初のサブゴール列を過去ログから取得
    def get_initial_subgoals_from_log(self, log:dict, params:dict):
        trees = log[1]["subgoal_tree"]
        subgoal_lists = []
        for i in range(len(trees)):
            tree = self.subgoal_trees[i]
            tree.set_dict(trees[i])
            tree.reset_access()
            tree.reduction()
            tree.extract()
            tree.remove_failed_node()
            tree.move_to_leaf()
            subgoal_lists.append(tree.get_subgoals())
        return subgoal_lists

    # 簡易的に履歴の文字列を取得する
    def get_history_str(self, agent_id:int, is_reflexion:bool, params:dict) -> str:
        if is_reflexion:
            length = utils.get_value(params, "reflexion_history_size", -1)
            labels = utils.get_value(params, "reflexion_history_labels", [])
            labels_len = utils.get_value(params, "reflexion_history_labels_len", {})
        else:
            length = utils.get_value(params, "history_size", -1)
            labels = utils.get_value(params, "history_labels", [])
            labels_len = utils.get_value(params, "history_labels_len", {})

        history = self.histories[agent_id].get_str(length, labels, labels_len)
        return history
    
    # 履歴と指示を合わせたプロンプトを返す
    def get_prompt(self, agent_id:int, instr:str, params:dict) -> str:
        history = self.get_history_str(agent_id, False, params)
        prompt = f"{history}\n\n{instr}"
        return prompt

    # Reflexionを行う時のプロンプトを返す
    def get_reflexion_prompt(self, agent_id:int, reflexion_type:str, reason:str, params) -> str:
        history = self.get_history_str(agent_id, True, params)
        if reflexion_type == "subgoal":
            instr = env_utils.get_subgoal_reflexion_instr(reason, agent_id, self.subgoal_trees[agent_id], params)
        elif reflexion_type == "general":
            instr = env_utils.get_general_reflexion_instr(reason, agent_id, params)
        else:
            instr = ""
        prompt = f"{history}\n\n{instr}"
        return prompt

    # メッセージを生成する時のプロンプトを返す
    def get_message_prompt(self, agent_id:int, targets:list[str], params:dict) -> str:
        instr = env_utils.get_message_instr(agent_id, targets, params)
        prompt = self.get_prompt(agent_id, instr, params)
        return prompt
    
    # 会話文を生成する時のプロンプトを返す
    def get_conversation_prompt(self, agent_id:int, targets:list[str], messages:list[tuple[str,str]], is_last:bool, params:dict) -> str:
        instr = env_utils.get_conversation_instr(agent_id, targets, messages, is_last, params)
        prompt = self.get_prompt(agent_id, instr, params)
        return prompt    
    
    # 行動を生成する際のプロンプトを返す
    def get_action_prompt(self, agent_id:int, params:dict) -> str:
        instr = env_utils.get_action_instr(self.env_name, agent_id, params)
        prompt = self.get_prompt(agent_id, instr, params)
        return prompt
    
    # 思考を生成する際のプロンプトを返す
    def get_consideration_prompt(self, agent_id:int, params:dict) -> str:
        subgoal_tree = self.subgoal_trees[agent_id]
        achieved, not_achieved = subgoal_tree.get_achieved_not_achieved()
        instr = env_utils.get_consideration_instr(agent_id, achieved, not_achieved, params)
        prompt = self.get_prompt(agent_id, instr, params)
        return prompt
    
    # サブゴールを生成する際のプロンプトを返す
    def get_subgoal_prompt(self, agent_id:int, achieved:list[str], not_achieved:list[str], params:dict) -> str:
        instr = env_utils.get_subgoal_instr(agent_id, achieved, not_achieved, params)
        prompt = self.get_prompt(agent_id, instr, params)
        return prompt
    
    # サブゴールを行動に変換する際のプロンプトを返す
    def get_subgoal_to_action_prompt(self, agent_id:int, subgoals:list[str], params:dict) -> str:
        instr = env_utils.get_subgoal_to_action_instr(self.env_name, agent_id, subgoals, params)
        
        temp = utils.get_value(params["history_labels_len"], "consideration", 0)
        params["history_labels_len"]["consideration"] = 0
        prompt = self.get_prompt(agent_id, instr, params)
        params["history_labels_len"]["consideration"] = temp
        
        return prompt
    
    # サブゴールを達成したかどうか判定する際のプロンプトを返す
    def get_subgoal_achieved_prompt(self, agent_id:int, subgoals:list[str], params:dict) -> str:
        instr = env_utils.get_subgoal_achieved_instr(self.env_name, agent_id, subgoals, params)
        prompt = self.get_prompt(agent_id, instr, params)
        return prompt
    def get_all_subgoals_achieved_prompt(self, agent_id:int, subgoals:list[str], params:dict) -> str:
        instr = env_utils.get_all_subgoals_achieved_instr(agent_id, subgoals, params)
        prompt = self.get_prompt(agent_id, instr, params)
        return prompt

    # エピソードの初めにサブゴールをまとめて生成する際のプロンプトを返す
    def get_init_subgoal_prompt(self, agent_id:int, params:dict) -> str:
        prompt = env_utils.get_init_subgoal_instr(self.env, self.tasks[agent_id], agent_id, params)
        return prompt
    
    # ToM推論用のプロンプトを返す
    def get_tom_prompt(self, agent_id:int, target_agent_id:int, params:dict) -> str:
        # 通常の履歴設定を読み込む
        length = utils.get_value(params, "history_size", -1)
        labels = utils.get_value(params, "history_labels", [])
        labels_len = utils.get_value(params, "history_labels_len", {})
        
        # 【重要】メッセージの履歴だけ強制的に長くする
        # これによりずっと会っていなくても過去の会話を思い出せるようにする
        tom_msg_len = utils.get_value(params, "tom_message_history_size", 50)
        labels_len['message'] = tom_msg_len
        
        # 特別な設定で履歴を取得
        history = self.histories[agent_id].get_str(length, labels, labels_len)
        instr = env_utils.get_tom_instr(agent_id, target_agent_id, params)
        prompt = f"{history}\n\n{instr}"
        return prompt