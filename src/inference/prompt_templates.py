from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional


class PromptFormat(Enum):
    """Supported prompt formats"""

    ALPACA = "alpaca"
    CHATML = "chatml"
    VICUNA = "vicuna"
    LLAMA2_CHAT = "llama2_chat"
    PHI3 = "phi3"
    MISTRAL = "mistral"
    PLAIN = "plain"


class PromptTemplate:
    """Base class for prompt templates"""

    def __init__(self, format_type: PromptFormat):
        self.format_type = format_type

    def format_prompt(self, prompt: str, **kwargs) -> str:
        """Format a prompt according to the template"""
        raise NotImplementedError

    def format_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format a chat conversation"""
        raise NotImplementedError


class AlpacaTemplate(PromptTemplate):
    """Alpaca prompt template"""

    def __init__(self):
        super().__init__(PromptFormat.ALPACA)

    def format_prompt(self, prompt: str, instruction: str = "", **kwargs) -> str:
        """Format prompt in Alpaca style"""
        if instruction:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{prompt}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{prompt}\n\n### Response:\n"

    def format_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format chat in Alpaca style"""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_messages.append(f"### System:\n{content}")
            elif role == "user":
                formatted_messages.append(f"### User:\n{content}")
            elif role == "assistant":
                formatted_messages.append(f"### Assistant:\n{content}")

        return "\n\n".join(formatted_messages) + "\n\n### Assistant:\n"


class ChatMLTemplate(PromptTemplate):
    """ChatML prompt template"""

    def __init__(self):
        super().__init__(PromptFormat.CHATML)

    def format_prompt(self, prompt: str, system_message: str = "", **kwargs) -> str:
        """Format prompt in ChatML style"""
        formatted = ""
        if system_message:
            formatted += f"<|im_start|>system\n{system_message}<|im_end|>\n"
        formatted += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return formatted

    def format_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format chat in ChatML style"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted


class Phi3Template(PromptTemplate):
    """Phi-3 prompt template"""

    def __init__(self):
        super().__init__(PromptFormat.PHI3)

    def format_prompt(self, prompt: str, system_message: str = "", **kwargs) -> str:
        """Format prompt in Phi-3 style"""
        formatted = ""
        if system_message:
            formatted += f"<|system|>\n{system_message}<|end|>\n"
        formatted += f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        return formatted

    def format_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format chat in Phi-3 style"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|{role}|>\n{content}<|end|>\n"
        formatted += "<|assistant|>\n"
        return formatted


class Llama2ChatTemplate(PromptTemplate):
    """Llama 2 Chat prompt template"""

    def __init__(self):
        super().__init__(PromptFormat.LLAMA2_CHAT)

    def format_prompt(self, prompt: str, system_message: str = "", **kwargs) -> str:
        """Format prompt in Llama 2 Chat style"""
        if system_message:
            return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"

    def format_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format chat in Llama 2 Chat style"""
        formatted = "<s>"
        system_message = ""

        # Extract system message if present
        if messages and messages[0].get("role") == "system":
            system_message = messages[0].get("content", "")
            messages = messages[1:]

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                if i == 0 and system_message:
                    formatted += f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{content} [/INST]"
                else:
                    formatted += f"[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content} </s><s>"

        return formatted


class PlainTemplate(PromptTemplate):
    """Plain text prompt template"""

    def __init__(self):
        super().__init__(PromptFormat.PLAIN)

    def format_prompt(self, prompt: str, **kwargs) -> str:
        """Format prompt as plain text"""
        return prompt

    def format_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format chat as plain text"""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_messages.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_messages) + "\nAssistant:"


class PromptTemplateManager:
    """Manager for prompt templates"""

    def __init__(self):
        self.templates: Dict[PromptFormat, PromptTemplate] = {
            PromptFormat.ALPACA: AlpacaTemplate(),
            PromptFormat.CHATML: ChatMLTemplate(),
            PromptFormat.PHI3: Phi3Template(),
            PromptFormat.LLAMA2_CHAT: Llama2ChatTemplate(),
            PromptFormat.PLAIN: PlainTemplate(),
        }

    def get_template(self, format_type: PromptFormat) -> PromptTemplate:
        """Get a prompt template by format type"""
        return self.templates.get(format_type, self.templates[PromptFormat.PLAIN])

    def detect_format(self, model_name: str) -> PromptFormat:
        """Detect prompt format based on model name"""
        model_name_lower = model_name.lower()

        if "phi-3" in model_name_lower or "phi3" in model_name_lower:
            return PromptFormat.PHI3
        elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
            if "chat" in model_name_lower:
                return PromptFormat.LLAMA2_CHAT
        elif "alpaca" in model_name_lower:
            return PromptFormat.ALPACA
        elif "chatml" in model_name_lower:
            return PromptFormat.CHATML
        elif "mistral" in model_name_lower:
            return PromptFormat.CHATML  # Mistral often uses ChatML

        return PromptFormat.PLAIN

    def format_prompt(
        self,
        prompt: str,
        format_type: Optional[PromptFormat] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Format a prompt using the appropriate template"""
        if format_type is None and model_name:
            format_type = self.detect_format(model_name)
        elif format_type is None:
            format_type = PromptFormat.PLAIN

        template = self.get_template(format_type)
        return template.format_prompt(prompt, **kwargs)

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        format_type: Optional[PromptFormat] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Format a chat conversation using the appropriate template"""
        if format_type is None and model_name:
            format_type = self.detect_format(model_name)
        elif format_type is None:
            format_type = PromptFormat.PLAIN

        template = self.get_template(format_type)
        return template.format_chat(messages, **kwargs)


# Global instance
prompt_manager = PromptTemplateManager()


def format_prompt(
    prompt: str,
    format_type: Optional[PromptFormat] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> str:
    """Convenience function to format a prompt"""
    return prompt_manager.format_prompt(prompt, format_type, model_name, **kwargs)


def format_chat(
    messages: List[Dict[str, str]],
    format_type: Optional[PromptFormat] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> str:
    """Convenience function to format a chat conversation"""
    return prompt_manager.format_chat(messages, format_type, model_name, **kwargs)
