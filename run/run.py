import asyncio
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from transformers import AutoTokenizer, AutoModel


retriever, llama_model, loop = None, None, None
model, tokenizer = None, None


async def build_llama_llm(local_model_path):
    
    from llama_index.llms.huggingface import HuggingFaceLLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
   
    print('load model')
    model = AutoModelForCausalLM.from_pretrained(local_model_path,trust_remote_code=True,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path,trust_remote_code=True)
    print('build hugging face llm')
    # Create the HuggingFaceLLM object
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=3900,
        max_new_tokens=256,
        device_map="cuda"
    )
    print('build hugging face embedding')
    embeding = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
        trust_remote_code=True
    )
    Settings.embed_model = embeding

    print('Start building vector store...')
    config = {"VECTOR_SIZE":512, "COLLECTION_NAME":"kefu"}
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "kefu"
    )
    print('collection_info')
    if collection_info.points_count == 0:
        data = read_data("../data/rag/abc")
        pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "kefu",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "kefu",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))
    print('Start building retriever...')
    retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=1)
    return retriever, llm


def parse_text(text):  
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def build_chat_pipeline_rag(model_path):
    print('build local rag chat pipeline')
    _loop = asyncio.get_event_loop()  # Initialize the event loop
    if _loop.is_running():
        future = asyncio.run_coroutine_threadsafe(build_llama_llm(model_path), _loop)
        _retriever, _llm = future.result()  # Wait for the coroutine to complete and get the return value
    else:
        _retriever, _llm = _loop.run_until_complete(build_llama_llm(model_path))
    global retriever, llama_model, loop
    retriever, llama_model, loop = _retriever, _llm,_loop
    
    
def build_chat_pipeline(model_path):
    print('build local chat pipeline')
    _model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    global model, tokenizer
    model,tokenizer=_model,_tokenizer
    

def predict(input, chatbot, max_length, top_p, temperature, history):
    global model, tokenizer
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))
        print(parse_text(response))
        
        yield chatbot, history  
 
def predict_rag(input, chatbot):
    chatbot.append((parse_text(input), ""))
    response = loop.run_until_complete(generation_with_knowledge_retrieval(input, retriever, llama_model))
    chatbot[-1] = (parse_text(input), parse_text(response.text))
    print(parse_text(response.text))
    yield chatbot


if __name__ == "__main__":
    pass
    
