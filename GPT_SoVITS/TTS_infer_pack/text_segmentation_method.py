import re
from typing import Callable

punctuation = set(["!", "?", "…", ",", ".", "-", " "])
METHODS = dict()


def get_method(name: str) -> Callable:
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method


def get_method_names() -> list:
    return list(METHODS.keys())


def register_method(name):
    def decorator(func):
        METHODS[name] = func
        return func

    return decorator


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def split_big_text(text, max_len=510):
    # 定义全角和半角标点符号
    punctuation = "".join(splits)

    # 切割文本
    segments = re.split("([" + punctuation + "])", text)

    # 初始化结果列表和当前片段
    result = []
    current_segment = ""

    for segment in segments:
        # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment

    # 将最后一个片段加入结果列表
    if current_segment:
        result.append(current_segment)

    return result


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    words = todo_text.split()
    if len(words) >= 3 and all(word.isalpha() for word in words[-3:]):
        todo_text += "."
    elif todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


# 不切
@register_method("cut0")
def cut0(inp):
    if not set(inp).issubset(punctuation):
        return inp
    else:
        return "/n"


# 凑四句一切
@register_method("cut1")
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 凑50字一切
@register_method("cut2")
def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 按中文句号。切
@register_method("cut3")
def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 按英文句号.切
@register_method("cut4")
def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 按标点符号切
# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
@register_method("cut5")
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

# 按行切，再按中文句号。！？切，再按英文句号.!?切，短句合并
@register_method("cut6")
def cut6(inp):
    inp = inp.strip("\n")
    lines = inp.split("\n")
    final_segments = []
    # Define punctuation set for filtering
    punctuation_set = set("。！？.!?，,;；:：—…~、 ")
    # Define punctuation for splitting long chunks
    all_punctuation = "。！？.!?，,;；:：—…~、"

    for line in lines:
        if not line.strip():  # Skip empty lines
            continue

        # Split by Chinese and English end-of-sentence punctuation, keeping delimiters
        parts = re.split(r'([。！？.!?])', line)

        merged_parts = []
        current_merge = ""

        for i, part in enumerate(parts):
            # Check if adding this part would make the chunk too long
            potential_chunk = current_merge + part
            
            # Count non-alphabetical characters
            non_alpha_count = len(re.sub(r'[a-zA-Z0-9\s]', '', potential_chunk))
            
            # Count unique spaces (continuous spaces count as 1)
            unique_space_count = len(re.findall(r'\S+', potential_chunk)) - 1 if potential_chunk.strip() else 0
            
            # Check if chunk would be too long
            is_too_long = non_alpha_count > 20 or unique_space_count > 10
            
            if is_too_long and current_merge:
                # Try to split at the nearest punctuation
                found_split = False
                for char in all_punctuation:
                    last_punct_pos = current_merge.rfind(char)
                    if last_punct_pos > 0:
                        # Split at punctuation
                        merged_parts.append(current_merge[:last_punct_pos+1])
                        current_merge = current_merge[last_punct_pos+1:] + part
                        found_split = True
                        break
                
                # If no punctuation found, just add the current chunk
                if not found_split:
                    merged_parts.append(current_merge)
                    current_merge = part
            else:
                current_merge = potential_chunk
            
            # Original length-based checks
            is_long_enough = len(current_merge) >= 6 or current_merge.count(' ') >= 3
            
            # If it's long enough or it's the very last part
            if is_long_enough and not is_too_long or i == len(parts) - 1:
                # Filter out segments that are only punctuation
                if not all(char in punctuation_set for char in current_merge.strip()):
                    merged_parts.append(current_merge)
                current_merge = ""  # Reset for the next segment

        # Add any remaining part if not empty
        if current_merge and not all(char in punctuation_set for char in current_merge.strip()):
            merged_parts.append(current_merge)

        final_segments.extend(merged_parts)

    # Filter out any empty strings
    final_segments = [seg for seg in final_segments if seg.strip()]

    return "\n".join(final_segments)

if __name__ == "__main__":
    method = get_method("cut5")
    print(method("你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。"))
