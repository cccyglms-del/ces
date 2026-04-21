import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    llm_api_key: str = ""
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: int = 45
    default_sample_size: int = 100
    pdf_render_dpi: int = 180

    @classmethod
    def from_env(cls):
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
            default_sample_size=int(os.getenv("DEFAULT_SAMPLE_SIZE", "100")),
            pdf_render_dpi=int(os.getenv("PDF_RENDER_DPI", "180")),
        )

    @property
    def llm_enabled(self):
        return bool(self.llm_api_key and self.llm_model)
