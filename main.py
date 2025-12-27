import logging
import httpx
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent
from astrbot.api.all import * # 这里通常包含了 filter

@register("hf_whisper", "ai", "基于 HF Router 的高性能语音识别", "1.0.0")
class HFWhisperPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        # 从配置中读取 Token
        self.token = self.config.get("hf_token", "")
        # 你截图中的 fal-ai 路由地址
        self.api_url = "https://router.huggingface.co/fal-ai/fal-ai/whisper"
        
    @filter.on_asr() # 根据最新文档，使用 filter.on_asr()
    async def handle_asr(self, event: AstrMessageEvent):
        # 1. 获取语音二进制数据
        audio_data = event.get_asr_data() 
        if not audio_data or not self.token:
            return

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "audio/flac" # 对应你之前截图的代码要求
        }

        try:
            # 2. 异步请求 Hugging Face Router
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
                    # 3. 设置识别出的文本
                    event.set_asr_text(text) 
            else:
                logging.error(f"HF API 异常: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"语音识别插件请求异常: {e}")
