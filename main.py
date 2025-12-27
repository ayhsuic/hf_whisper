import logging
import httpx
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent, EventType # 导入事件类型
from astrbot.api.all import *

@register("hf_whisper", "ai", "基于 HF Router 的高性能语音识别", "1.0.0")
class HFWhisperPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.token = self.config.get("hf_token", "")
        self.api_url = "https://router.huggingface.co/fal-ai/fal-ai/whisper"
        
    # 使用通用的事件监听装饰器，并指定监听语音识别事件
    @on_event(EventType.ASR) 
    async def handle_asr(self, event: AstrMessageEvent):
        # 1. 获取音频二进制数据
        audio_data = event.get_asr_data() 
        if not audio_data or not self.token:
            return

        # 按照你提供的截图要求配置 headers
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "audio/flac" 
        }

        try:
            # 2. 异步发送请求
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
                    # 3. 将文本设置到事件中供后续逻辑（如回复）使用
                    event.set_asr_text(text) 
            else:
                logging.error(f"HF API 报错: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"语音识别插件运行异常: {e}")
