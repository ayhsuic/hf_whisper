import logging
import httpx
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent
# 显式导入所有需要的组件
from astrbot.api.event.filter import on_asr 
from astrbot.api.all import *

@register("hf_whisper", "ai", "基于 HF Router 的高性能语音识别", "1.0.0")
class HFWhisperPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.token = self.config.get("hf_token", "")
        self.api_url = "https://router.huggingface.co/fal-ai/fal-ai/whisper"
        
    @on_asr() # 监听语音识别事件
    async def handle_asr(self, event: AstrMessageEvent):
        # 获取音频二进制数据
        audio_data = event.get_asr_data() 
        if not audio_data or not self.token:
            return

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "audio/flac" 
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    content=audio_data,
                    timeout=60.0 
                )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                
                if text:
                    logging.info(f"HF Whisper 识别成功: {text}")
                    event.set_asr_text(text) # 设置识别出的文本
            else:
                logging.error(f"HF API 异常: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"语音识别插件请求异常: {e}")
            
