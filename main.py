import os
import logging
import httpx # 记得 pip install httpx
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent
from astrbot.api.all import *

@register("hf_whisper_plugin", "YourName", "基于 HF Router 的异步语音识别插件", "1.0.0")
class HFWhisperPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        # 从 metadata.yaml 定义的配置中读取
        self.token = self.config.get("hf_token", "")
        # 使用你截图中的 fal-ai 路由地址
        self.api_url = "https://router.huggingface.co/fal-ai/fal-ai/whisper"
        
    @on_asr()
    async def handle_asr(self, event: AstrMessageEvent):
        # 1. 获取语音二进制数据
        audio_data = event.get_asr_data() 
        if not audio_data or not self.token:
            return

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "audio/flac" # 需确保与发送格式对应
        }

        try:
            # 2. 使用 httpx 发送异步 POST 请求
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    content=audio_data,
                    timeout=60.0 # 语音推理较慢，设置长一点的超时
                )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "") # 提取识别出的文字
                if text:
                    logging.info(f"HF Whisper 识别成功: {text}")
                    # 3. 设置识别结果，供大模型继续处理
                    event.set_asr_text(text)
            else:
                logging.error(f"HF API 报错: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"异步语音识别请求失败: {e}")
