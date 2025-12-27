import logging
import httpx
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent
from astrbot.api.all import *

@register("hf_whisper", "ai", "基于 HF Router 的高性能语音识别", "1.0.0")
class HFWhisperPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        # 从配置中读取 Token
        self.token = self.config.get("hf_token", "")
        # 你截图中的 fal-ai 路由地址
        self.api_url = "https://router.huggingface.co/fal-ai/fal-ai/whisper"
        
    @on_asr()
    async def handle_asr(self, event: AstrMessageEvent):
        # 1. 获取音频二进制数据
        audio_data = event.get_asr_data() 
        if not audio_data:
            return
        
        if not self.token:
            logging.error("HF Whisper 插件报错: 未在配置中填写 hf_token")
            return

        # 2. 构造请求头
        # 注意：如果识别失败，尝试将 audio/flac 改为 audio/wav
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "audio/flac" 
        }

        try:
            # 3. 异步请求 Hugging Face Router
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    content=audio_data,
                    timeout=60.0 # 推理需要时间，超时设置长一些
                )
            
            if response.status_code == 200:
                result = response.json()
                # 根据你截图中的响应逻辑提取文本
                text = result.get("text", "").strip()
                
                if text:
                    logging.info(f"HF Whisper 识别成功: {text}")
                    # 将识别结果告知 Astrbot 核心
                    event.set_asr_text(text)
                else:
                    logging.warning("HF Whisper 返回了空文本")
            elif response.status_code == 503:
                logging.warning("HF 模型正在加载中，请稍后再试（冷启动）")
            else:
                logging.error(f"HF API 异常: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"语音识别插件请求异常: {e}")
