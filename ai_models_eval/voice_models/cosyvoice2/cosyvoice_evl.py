#%%
import sys
cosyvoice_code_path = "/home/andrewzhu/storage_1t_1/github_repos/CosyVoice"
sys.path.append(cosyvoice_code_path)

import torch
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import json

class CosyVoice2Eval:
    def __init__(self):
        self.cosyvoice = CosyVoice2(
            'pretrained_models/CosyVoice2-0.5B'
            , load_jit  = True
            , load_onnx = False
            , load_trt  = False
        )
        
        self.sample_meta_path = "samples/samples.json"
        with open(self.sample_meta_path, 'r', encoding="utf-8") as f:
            self.sample_data = json.loads(f.read())
    
    def reload_sample_data(self):
        with open(self.sample_meta_path, 'r', encoding="utf-8") as f:
            self.sample_data = json.loads(f.read())
    
    def get_ref_voice(self,ref_voice_name:str):
        prompt_speech_16k   = load_wav(
            f'samples/{self.sample_data[ref_voice_name]["file_name"]}'
            , self.sample_data[ref_voice_name]["frequency"]
        )
        prompt_speech_text  = self.sample_data[ref_voice_name]["prompt_text"]
        return prompt_speech_16k, prompt_speech_text
    
    def gen_voice(
        self
        , ref_voice_name:str
        , input_text:str
        , speed:float = 1.0
        , text_frontend:bool = True
        , output_audio_name:str = None
    ):
        prompt_speech_16k, prompt_speech_text = self.get_ref_voice(ref_voice_name)
        result_iter         = self.cosyvoice.inference_zero_shot(
            tts_text            = input_text
            , prompt_text       = prompt_speech_text
            , prompt_speech_16k = prompt_speech_16k
            , stream            = False
            , speed             = speed
            , text_frontend     = text_frontend
        )

        sound_tensor_list = []
        for i, j in enumerate(result_iter):
            sound_tensor_list.append(j['tts_speech'])
        
        if output_audio_name is None:
            output_audio_name = "default_output.wav"
        all_sound_tensor = torch.cat(sound_tensor_list, dim=1)
        torchaudio.save(
            f'outputs/{output_audio_name}'
            , all_sound_tensor
            , self.cosyvoice.sample_rate
        )
        print("audio generation done")
        
cosyvoice_pipe = CosyVoice2Eval()

#%%
input_text = """\
"""
input_text = input_text.replace("……","...")
input_text = input_text.replace("？","?")
input_text = input_text.replace('”','"')
input_text = input_text.replace('“','"')
input_text = input_text.replace('。','.')
input_text = input_text.replace('、',',')
print(input_text)
cosyvoice_pipe.gen_voice(
    ref_voice_name      = "cn_woman_2"
    , input_text        = input_text
    , output_audio_name = "test_c.wav"
)

#%%
input_text = """\
妆儿有些焦躁地说：“你说话的时候不要用自己的名字！”然后转向他指着他的额头说：“还有你！我说了多少次让你给她改个名字，听起来好像我和她是一种层次一样！”

蝶儿吐了吐舌头，“主人这么设定的，蝶儿也没办法。夫人若是有密码，帮蝶儿改了就好。”

三人一边说一边进了屋子，房门自动打开关上后，里面的灯具立刻亮起，金碧辉煌的大厅，却像极了他记忆中曾见过的样子，并不科幻。

妆儿把身上的金属外衣脱下来丢在一边，双脚甩掉了高跟鞋，揉了揉脚跟，“874，带巴特先生去他房间用餐，我累了，这里的装潢看起来老气得要死，我回我的房间洗澡去了，吃饭不用等我，这次是为你买的型号，不对我的胃口，我晚些再过去。”

他伸了伸手想叫住妆儿，但心中对她实在有些害怕，就没喊出口。心想实在不行就问蝶儿好了，看起来这个可爱的机器人与自己的关系更为密切。

上到三楼，蝶儿指着一道门说：“主人，这里是您的房间，虽然卧室在夫人那边，不过你也有阵子没去睡过了，所以夫人变得爱唠叨，你也要负责任哦。”

他支支吾吾的应了声，说：“我想不起来的事情很多，你能告诉我么？”

蝶儿嘻嘻一笑，说：“主人什么时候说话这么客气了？蝶儿好不习惯呢。蝶儿去换主人喜欢的衣服，主人先进去用餐吧。这次的可是高价货，只比主人最喜欢的那种便宜一点点哦……不过夫人肯定不会喜欢，主人就尽管多吃些吧。”

蝶儿一边走一边笑着说：“其实这种高级食物，还是夫妇分开吃才好。”

他听得一头雾水，但还是推门走了进去，他确实感到饿了，有吃的东西的话就先填饱肚子再解决脑子的问题吧。

门内是很大的一间屋子，另外还有好几道门，进门的房间似乎是餐厅，几张长沙发中间是摆满了盘碗刀叉的矮长餐桌，尽头是液晶屏幕一样的巨大墙壁，其余再没有什么，陈设十分简单。餐桌边站着一个十四五岁的少女，身材的火辣程度和她看起来天真的笑脸十分不符。看见他进来，那少女立刻恭敬的走了过来，鞠躬说：“巴特先生晚上好，我是LOF——TF19052号，今晚由我负责您的晚餐。”

他这才注意那少女仅穿了一件围裙，这一躬身，赤裸的粉红身体尽收眼底。

尤其是那对丰满的乳房，晃动得十分诱人。

他勉强把视线拉到一边，走过去坐到餐桌边，说：“好……我知道了。”

那少女走到他身后，拿起餐巾垫在他膝上，大半个身子就这么摆在他眼前，挑战着他的理智，他清楚的感觉到，自己的阴茎已经完全勃起，正顶着他裤裆，磨得甚至有些发痛。

那少女把餐具摆好，然后脱下了围裙，端来了一大盆清水，然后站进去就那么赤身裸体的用水洗起了身子。

他奇怪的看着这个女孩子仔细的清洗着身上的每一个部分，从高耸的乳房到红嫩的阴唇间，当她转过身背对着他弯下腰清洗着那双小脚的时候，他终于忍耐不住，猛地站起身子，走过去，从背后一把抱住了那个少女，双手罩住那一双乳房，拼命的揉搓起来。

自己是巴特……孟蝶是不存在的……那么，自己干什么，也不算对不起谁，他自我安慰着，张嘴在那少女身子上乱吻起来。那少女的肌肤无比滑嫩，还带着好闻的清香，让他一吻之下就再难收手，连忙把少女推到餐桌边趴下，开始解自己的裤子。

那少女很疑惑的回头说：“巴特先生，用餐前您还需要特殊服务么？”

他想着自己的身份，一个有钱的阔佬，说不定这女孩子便是故意勾引自己，好多赚些外快，便说：“没错，你让我来一次，我舒服了，少不了你的好处。”

那少女也没有挣扎或反抗，而是皱着眉头说：“巴特先生，因此影响到您的用餐质量的话，我们公司是不会负责的。”

他哪里还顾得了这些，站到少女身后把她的双脚打开，用手指拨开鲜嫩的阴唇，用龟头在阴门处蹭了蹭，便毫不犹豫地插了进去。

虽然没有多少汁液润滑，他粗大的肉棒仍然顺畅的尽根而入，这女孩子果然已经不是处女，他不免有些莫名的恼怒，双手掐住水嫩的屁股，大力奸淫起来。

那少女皱着眉说道：“巴特先生……您的夫人订餐的时候……是说夫妇一起的，所以，我们没准备有特殊服务的……”

夫妇？他猛地醒过神来，一会儿妆儿洗完澡似乎就要过来，尽管心里觉得被她看见也没什么，但还是有些害怕。只是这少女阴道内炽热的腔肉一缩一缩的吸着插进去的肉棒，让他就这么拔出来，却是怎么也做不到。

“她不会说什么的，你把屁股翘高了就可以！别的废话少说！”他有一些焦躁，口气也变得凶暴起来，但说出这些话的时候却有种这才是自己的错觉。

那少女嗯了一声，把屁股高高撅起，同时也开始摇摆着细腰，熟练的取悦着他。

阴道内层层叠叠的嫩肉磨擦得他浑身颤抖，虽然这少女阴道似乎颇长，怎么也顶不到尽头，但仅仅是被紧紧勒着的阴茎本身，就已经非常爽快了。

他双手紧握住少女挤在桌面上的丰腴乳房，那乳肉软嫩得超出了他的想象，简直像用力过大就会被掐断一样。他拼命的掐着揉着，阴茎也大力地抽动着，但那少女只是脸颊有些泛红，既不痛呼，也没有愉悦的呻吟，让他不禁有些沮丧。

他把少女的身子反转过来，把那双小脚捧在胸前，孟蝶的脚就十分得好看，每次夏天她穿凉鞋的时候，自己总会忍不住盯着她的脚发呆上半天。而现在这双脚，更加的小巧秀美，粉粉嫩嫩的让人看了就忍不住咬上一口一样。他一口含住一只脚趾，吸吮起那饱满柔软的趾肚，心中油然一股兴奋和满足。

那少女的脚被含进嘴里，突然全身绷紧，嘴里也大声地呻吟起来，好像这比阴道中抽插的阴茎更加刺激一样，那充满弹性的阴道也剧烈的收缩起来，只是不知道为何仍然没有什么爱液。

但那少女的呻吟已经极大的刺激了他，他只觉阴茎根部一阵发紧，知道高潮将至，狠狠的最后插了几下，把浓稠的精液射进了那少女的阴道深处。

他趴在那少女柔软的身子上喘着粗气，不愿意起身。那少女推了推他，“巴特先生，您起来一下，我还要清洗干净，不然会被您夫人投诉的。”

“投诉？她要投诉什么？”他笑了起来，突然觉得自己能当着她的面把这个少女再奸上一次，似乎会更加刺激。那张和孟蝶一样的脸，却是完全不同性格，让他心里十分的不舒服。

没想到，门那边马上传来了他熟悉的声音，“她要是不洗干净，我一定会投诉。你愿意吃精液是你的事，我可没兴趣晚餐时间吃那种恶心的玩意。”
"""
input_text = input_text.replace("……","...")
input_text = input_text.replace("？","?")
input_text = input_text.replace('”','"')
input_text = input_text.replace('“','"')
input_text = input_text.replace('。','.')
input_text = input_text.replace('、',',')
print(input_text)
cosyvoice_pipe.gen_voice(
    ref_voice_name      = "cn_woman_2"
    , input_text        = input_text
    , output_audio_name = "test5.wav"
)