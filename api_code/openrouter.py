import os
from openai import OpenAI

def run_openrouter(api_key, model, prompt):
    client = OpenAI(
        base_url=os.environ.get("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )

    # ストリーミングモードで補完を作成
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": os.environ.get("OPENROUTER_ROLE_USER", "user"),
                "content": prompt
            }
        ],
        stream=True  # ストリーミングを有効化
    )

    # ストリーミングレスポンスを連結して返す
    result = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content

    return result
