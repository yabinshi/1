import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
sys.path.append("notebook/C3 搭建知识库") # 将父目录放入系统路径中
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma

# 检查 Secrets 是否配置成功
def check_secrets():
    st.write("### 🔑 运行环境检查")
    # 检查智谱 API Key
    if "ZHIPUAI_API_KEY" in st.secrets:
        st.success("✅ ZHIPUAI_API_KEY 已配置")
    else:
        st.error("❌ 缺少 ZHIPUAI_API_KEY！请在 Streamlit Cloud 的 Settings -> Secrets 中添加。")
        
    # 检查通义千问 API Key (Dashscope)
    if "DASHSCOPE_API_KEY" in st.secrets:
        st.success("✅ DASHSCOPE_API_KEY 已配置")
    else:
        st.warning("⚠️ 缺少 DASHSCOPE_API_KEY，通义千问模型可能无法调用。")

# 在页面最上方运行检查
check_secrets()
st.title("Debug Mode: App is Loading...")
st.write(f"当前工作目录: {os.getcwd()}")
def get_retriever():
    # 1. 确保绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(current_dir, 'data_base/vector_db/chroma')
    
    # 2. 修正变量定义位置：确保 embedding 始终被定义
    st.write("🔄 正在初始化 Embedding 模型...")
    # 直接显式传入 key，避免环境变量读取延迟
    embedding = ZhipuAIEmbeddings(api_key=st.secrets["ZHIPUAI_API_KEY"])
    
    # 3. 检查目录
    if not os.path.exists(persist_directory):
        st.error(f"❌ 数据库目录丢失: {persist_directory}")
        st.stop()

    try:
        st.write("📂 正在打开 Chroma 索引文件...")
        # 增加一个简单的超时提示或日志输出
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        st.success("✅ 数据库加载成功！")
        return vectordb.as_retriever()
    except Exception as e:
        st.error(f"❌ Chroma 加载失败: {e}")
        st.write("提示：请检查 GitHub 上的 data_base/vector_db/chroma 文件夹是否包含 .bin 和 sqlite3 文件。")
        st.stop()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(model="qwen-max", temperature=0,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs,
        ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]
            
@st.cache_resource
def load_chain():
    return get_qa_history_chain()

def main():
    st.write("Debug: App is starting...") # 如果页面能显示这行，说明问题出在下面的初始化
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))















