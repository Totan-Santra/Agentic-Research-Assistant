import streamlit as st
from main import app as agent_app
from langchain_core.messages import HumanMessage


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Agentic Research Assistant")

st.write(
    "Ask questions and let the AI agent decide whether to use "
    "RAG, web search, calculator, or direct reasoning."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input("Ask me anything...")


if user_input:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------------
    # Agent response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Researching..."):

            try:

                config = {
                    "configurable": {
                        "thread_id": "streamlit_user"
                    }
                }

                result = agent_app.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    },
                    config=config
                )

                final_message = result["messages"][-1]

                response = final_message.content

                st.markdown(response)


                # ------------------------------------------------
                # Save assistant response
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )


            except Exception as e:

                error_message = f"Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )