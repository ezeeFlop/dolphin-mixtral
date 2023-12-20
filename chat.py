import streamlit as st
import os
from llama_cpp import Llama
import uuid

MODEL = "dolphin-2.5-mixtral-8x7b.Q4_K_M.gguf"
MODEL_PATH = "./models"
SESSIONS_PATH = "./sessions"
MODEL_PARAMS = f"{MODEL}.params"

def save_session():
    if not os.path.exists(SESSIONS_PATH):
        os.makedirs(SESSIONS_PATH)
    (sessionTitles, sessionsList) = list_sessions()

    sessionUuid = f"{uuid.uuid4()}"
    sessionFile = f"{MODEL}.${sessionUuid}.session"
    sessionFile = os.path.join(SESSIONS_PATH, sessionFile)

    for s in sessionsList:
        if s["title"] == st.session_state.messages[1]["content"]:
            sessionFile = os.path.join(SESSIONS_PATH, f"{s['file']}")
            break

    with open(sessionFile, "w") as f:
        f.write(str(st.session_state.messages))

def import_session(session: str):
    if not os.path.exists(SESSIONS_PATH):
        os.makedirs(SESSIONS_PATH)
    if os.path.exists(os.path.join(SESSIONS_PATH, f"{session}")):
        with open(os.path.join(SESSIONS_PATH, f"{session}"), "r") as f:
            content = eval(f.read())
            return content
    return None

def load_session(session: str):
    if not os.path.exists(SESSIONS_PATH):
        os.makedirs(SESSIONS_PATH)
    if os.path.exists(os.path.join(SESSIONS_PATH, f"{session}")):
        with open(os.path.join(SESSIONS_PATH, f"{session}"), "r") as f:
            return eval(f.read())
    return None

def list_sessions():
    if not os.path.exists(SESSIONS_PATH):
        os.makedirs(SESSIONS_PATH)
    sessionsList = []
    sessionTitles = []
    for f in os.listdir(SESSIONS_PATH):
        if f.endswith(".session"):
            session = load_session(f)
            if session is not None:
                sessionTitle = session[1]["content"]
                sessionFile = f
                sessionsList.append({
                    "title": sessionTitle,
                    "file": sessionFile,
                })
                sessionTitles.append(sessionTitle)
    return (sessionTitles, sessionsList)


llm = Llama(os.path.join(MODEL_PATH, MODEL),
            # The max sequence length to use - note that longer sequence lengths require much more resources
            verbose=False, n_ctx=32768,
            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_threads=8,
            # The number of layers to offload to GPU, if you have GPU acceleration available
            n_gpu_layers=35,
            )

st.title("Chat with Me 🤖")
st.subheader(f"Model : {MODEL}")

with st.sidebar:
    st.subheader("Chat sessions")
    session = st.selectbox('Sessions', ["New Chat"] + list_sessions()[0], index=0, key=f"select chat session to load")
    if session == "New Chat":
        st.session_state.messages = [
            {"role": "system", "content": "You are an helpful honest assistant."}]
    else:
        for s in list_sessions()[1]:
            if s["title"] == session:
                messages = import_session(s["file"])
                st.session_state.update({"messages": messages})

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an helpful honest assistant."}]

for message in st.session_state["messages"]:
    role = message["role"]
    if message["role"] == "system":
        role = "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])

# user input
if user_prompt := st.chat_input("Your prompt"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # generate responses
    with st.chat_message("assistant"):
        message_placeholder = st.text("Thinking...")
        full_response = ""

        for response in llm.create_chat_completion(
                        messages = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                        max_tokens=4000,
                        temperature=0.7,
                        repeat_penalty=1.1,
                    ):
            if "content" in response["choices"][0]["delta"]:
                full_response += response["choices"][0]["delta"]["content"]
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_session()
