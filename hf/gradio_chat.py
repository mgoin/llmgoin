from openai import OpenAI
import gradio as gr

base_url = "http://localhost:8000/v1"
api_key = "dummy"
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# Get hosted model name
model = client.models.list().data[0].id
print(f"Found model '{model}' hosted on {base_url}")

def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})
  
    response = client.chat.completions.create(
        model=model,
        messages=history_openai_format,
        temperature=0.0,
        stream=True,
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message

gr.ChatInterface(predict).launch(share=True)
