# Chat-medical

**Chat-medical** 是基于ChatGLM-6B基座进行LORA微调和推理检索增强（RAG）的医疗对话系统，代码主要来自https://github.com/KMnO4-zx/huanhuan-chat

**环境** <br>

python==3.12.3 <br>
torch==2.3.0+cu121 <br>
llama-index==0.10.29 <br>

**获取基座模型**<br>

从 https://huggingface.co/THUDM/chatglm2-6b 下载基座模型到data/model/base <br>


**下载数据集**<br>

cd data <br>
python data.py <br>

**LORA微调** <br>

cd fine_tune/lora <br>
./train.medical.sh <br>

**合并LORA旁路到基座** <br>

python merge.py --adapter_model_path 'data/adapt'--merged_model_save_path 'merged_model' <br>

**推理** <br>

cd run <br>
./run_gui.sh <br>
本地浏览器访问 localhost:6006 即可开始聊天

![示例](chat.png)
