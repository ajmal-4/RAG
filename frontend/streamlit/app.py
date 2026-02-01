import streamlit as st
import requests

st.set_page_config(page_title="Qwen AI Chat", page_icon="🤖")
st.title("Qwen Agentic Chat")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        try:
            # Make the streaming request to your FastAPI endpoint
            with requests.post(
                "http://localhost:8000/api/v1/chat",
                json={
                    "question": prompt,
                    "model_name": "qwen"
                },
                stream=True,
                timeout=60
            ) as r:
                # Check for successful connection
                r.raise_for_status()
                
                # Iterate over the raw stream chunks
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        # Decode the byte chunk to string
                        token = chunk.decode("utf-8")
                        full_response += token
                        # Update the UI in real-time
                        placeholder.markdown(full_response + "▌")
                
                # Final update to remove the cursor
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")