from sentence_transformers import SentenceTransformer, util
from utils.utils import get_value, extraction_numbers
from torch import Tensor

# 文章の類似度を求める際に必要な機能
# SubgoalTreeの手法でのみ用いる

class Embedder:
    is_loaded : bool = False
    __model : SentenceTransformer = None
    __model_name : str = ""
    __cache : dict[str, Tensor] = {}

    # Embedding用のモデルを読み込む
    @classmethod
    def load(cls, config:dict={}):
        model_name = get_value(config, "embedding_model", "paraphrase-MiniLM-L6-v2")
        if cls.is_loaded and cls.__model_name == model_name: return
        cls.__model = SentenceTransformer(model_name)#, device="cpu")
        cls.__model_name = model_name
        cls.is_loaded = True

    # 2つのテキストの類似度を求める
    @classmethod
    def get_similarity(cls, text1:str, text2:str) -> float:
        emb1 = cls.__get_embedding(text1)
        emb2 = cls.__get_embedding(text2)

        # 含まれる数字が違ったら異なるものとして判定する(今回は座標の違いなどが大事なので)
        if extraction_numbers(text1) != extraction_numbers(text2):
            return 0
        cosine_score = util.pytorch_cos_sim(emb1, emb2)[0][0]
        return float(cosine_score)

    @classmethod
    def __get_embedding(cls, text:str) -> Tensor:
        if text in cls.__cache.keys():
            return cls.__cache[text]
        embeddings = cls.__model.encode(text, convert_to_tensor=True)
        cls.__cache[text] = embeddings
        return embeddings

# テスト用
if __name__ == "__main__":
    Embedder.load()
    while True:
        print("text1:")
        text1 = input()
        print("text2:")
        text2 = input()
        similarity = Embedder.get_similarity(text1, text2)
        print(similarity)