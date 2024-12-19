import json
import random
from pathlib import Path
from typing import Optional

import emoji
from loguru import logger

SERVICE_NAME = "TStickerService"
USERNAME = "telegram"

# 读取规则，本文件目录下的rules.json
rule_file = Path(__file__).parent / "rules.json"
EMOJI_RULES = json.loads(rule_file.read_text(encoding='utf-8'))


def add_emoji_rule(rule: str, emoji_char: str):
    """
    添加一个emoji规则
    :param rule: 规则
    :param emoji_char: emoji字符
    :raises ValueError: 如果生成的emoji不受支持
    """
    # 判断是否是 emoji
    if not emoji.is_emoji(emoji_char):
        raise ValueError(f"Emoji {emoji_char} is not supported")
    EMOJI_RULES[rule] = emoji_char


def target_emoji(text: str) -> Optional[str]:
    """
    判断是否是目标emoji
    :param text: 输入的字符串
    :return: 生成的emoji字符
    """
    # 遍历规则，查找键是否在文本中
    for rule, emj in EMOJI_RULES.items():
        if rule.lower() in text.lower():
            return emj
    return None


def get_random_emoji_from_text(
        text: str,
        fallback_emoji: str = None
) -> str:
    """
    从给定的文本中提取字母并根据映射规则生成随机emoji。
    如果文本中没有匹配的字符，则返回默认emoji（❤️）。如果生成的emoji不受支持，则引发ValueError。
    :param text: 输入的字符串
    :param fallback_emoji: 默认的emoji字符
    :return: 生成的emoji字符
    """
    if fallback_emoji is None:
        fallback_emoji = emoji.emojize(":heart:")
    emoji_candidates = []
    # 仅处理文本中下划线后的部分
    if "_" in text:
        text = text.split("_")[-1]

    # 遍历规则，查找键是否在文本中
    find_emoji = target_emoji(text)
    if find_emoji:
        emoji_candidates.append(find_emoji)

    # 未找到匹配字符使用默认emoji
    if not emoji_candidates:
        selected_emoji = fallback_emoji
    else:
        selected_emoji = random.choice(emoji_candidates)

    # 处理和确认emoji是有效的
    selected_emoji = emoji.emojize(emoji.demojize(selected_emoji.strip()))
    if not emoji.is_emoji(selected_emoji):
        logger.warning(f"Emoji {selected_emoji} is not supported")
        selected_emoji = fallback_emoji
    return selected_emoji
