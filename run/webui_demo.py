#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import mdtex2html
import gradio as gr
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import sys

sys.path.append("..")
from run import predict_rag,predict,build_chat_pipeline_rag,build_chat_pipeline

@dataclass
class RunArguments:
    model_path: str = field(default = "../merged_model")
    log_name: str = field(default="log")
    RAG: bool = field(default=False)
    

run_args = HfArgumentParser(RunArguments).parse_args_into_dataclasses()[0]


"""重载聊天机器人的数据预处理"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

if run_args.RAG:
    build_chat_pipeline_rag(run_args.model_path)
else:
    build_chat_pipeline(run_args.model_path)
if True:
        
    with gr.Blocks() as demo:

        gr.HTML("""
        <h1 align="center">
                Chat-客服机器人
        </h1>
        """)

        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Input...", lines=10).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(
                    0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                                label="Top P", interactive=True)
                temperature = gr.Slider(
                    0, 1.5, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

       
        if not run_args.RAG:
            submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], 
                            [chatbot, history], show_progress=True)
            print('local chat')
        else:
            submitBtn.click(predict_rag, [user_input,chatbot], [chatbot], show_progress=True)
            print('rag chat')
            
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    demo.queue().launch(server_name="0.0.0.0", share=False,
                        inbrowser=False, server_port=6006)
    
    
    

