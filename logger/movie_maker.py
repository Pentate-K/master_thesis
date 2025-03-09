import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Gym環境の実行時にステップ毎に画像を取得して
# Gifアニメーションを生成する処理
# 動作確認やプレゼン用

class MovieMaker:
    def __init__(self, env, path:str = "./"):
        self.env = env
        self.rgb_data = []
        self.path = path
    
    # 保持した画像を消去
    def reset(self):
        self.rgb_data = []

    # 画像を記録して保持しておく
    def render(self):
        #rgb = self.env.render(mode='rgb_array') # 環境によっては引数が必要かも
        rgb = self.env.render()

        img = Image.fromarray(np.array(rgb).astype(np.uint8))
        rgb = np.array(img)

        self.rgb_data.append(rgb)
    
    # 画像に文字を追加する
    def add_text_to_image(self, pil_img, right, top, color, text, size=13):
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("arial.ttf", size)
        draw.text((right, top), text, color, font=font, align="left")

    # 画像に経過時間を描画する
    def add_step_to_image(self, image, step):
        self.add_text_to_image(image, 3, 3, "white", f"t={step}")

    # 現時点で保持している画像からGifアニメーションを生成する
    def make(self, name:str = "tmp"):
        images = []
        for i, rgb in enumerate(self.rgb_data):
            rgb = np.array(rgb)
            image = Image.fromarray(rgb.astype('uint8')).convert('RGB')
            self.add_step_to_image(image, i)
            images.append(image)
        images[0].save(f'{self.path}{name}.gif', save_all=True, append_images=images[1:],optimize=False, duration=100, loop=1)
    
    # 最終フレームだけ出力する(実行中もリアルタイムに状況を見たい場合)
    def make_last_frame(self, name:str = "tmp"):
        rgb = np.array(self.rgb_data[-1])
        image = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        self.add_step_to_image(image, len(self.rgb_data)-1)
        image.save(f'{self.path}{name}.png')