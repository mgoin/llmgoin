import uuid
import gradio as gr
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# Initialize the language model
engine_args = AsyncEngineArgs(model="facebook/opt-125m")
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_response(message, history):
    SAMPLING_PARAM = SamplingParams(max_tokens=100)
    print("PROMPT:", message)
    stream = await engine.add_request(uuid.uuid4().hex, message, SAMPLING_PARAM)
    async for request_output in stream:
        text = request_output.outputs[0].text
        print("STREAM:", repr(text))
        yield text

# Launch Gradio interface
gr_interface = gr.ChatInterface(fn=generate_response)
gr_interface.launch(share=True)
