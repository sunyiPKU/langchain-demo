import os
import re

import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMMathChain
from langchain.schema import HumanMessage, AIMessage
from langchain.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

embedding = OpenAIEmbeddings()
persistDirectory = 'db'


def initialize():
    llm = ChatOpenAI(temperature=0)

    search = GoogleSerperAPIWrapper()
    wiki = WikipediaAPIWrapper(top_k_results=1)

    llmMathChain = LLMMathChain.from_llm(llm=llm)
    tools = [
        Tool(
            name="Calculator",
            func=llmMathChain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name="wikipedia",
            func=wiki.run,
            description="useful for when you need to answer questions about historical entity. the input to this should be a single search term."
        ),
        Tool(
            name="Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world, also useful if there is no wikipedia result. the input to this should be a single search term."
        )
    ]

    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chatbotEngine = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory)
    return chatbotEngine


chatbotEngine = initialize()


def chatWithAgent(chatHistory, message=""):
    if not message.strip():
        return chatHistory, chatHistory, ""

    try:
        result = chatbotEngine.run(message.strip())
    except ValueError:
        result = "I can't handle this request, please try something else."

    chatHistory.append((message, result))
    return chatHistory, chatHistory, ""


def chatWithOpenAI(question, chatHistory):
    chatOpenAi = ChatOpenAI()
    if len(chatHistory) == 0:
        ans = chatOpenAi([HumanMessage(content=question)])
    else:
        messages = []
        for i in range(len(chatHistory[0])):
            if i % 2 == 0:
                messages.append(HumanMessage(content=chatHistory[0][i]))
            else:
                messages.append(AIMessage(content=chatHistory[0][i]))
        messages.append(HumanMessage(content=question))
        ans = chatOpenAi(messages)
    chatHistory.append((question, ans.content))
    return "", chatHistory

def splitParagraph(text, pdf_name, maxLength=300):
    text = text.replace('\n', '')
    text = text.replace('\n\n', '')
    text = re.sub(r'\s+', ' ', text)

    sentences = re.split('(；|。|！|\!|\.|？|\?)', text)

    newSents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        newSents.append(sent)
    if len(sentences) % 2 == 1:
        newSents.append(sentences[len(sentences) - 1])

    paragraphs = []
    current_length = 0
    current_paragraph = ""
    for sentence in newSents:
        sentence_length = len(sentence)
        if current_length + sentence_length <= maxLength:
            current_paragraph += sentence
            current_length += sentence_length
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence
            current_length = sentence_length
    paragraphs.append(current_paragraph.strip())
    documents = []
    metadata = {"source": pdf_name}
    for paragraph in paragraphs:
        new_doc = Document(page_content=paragraph, metadata=metadata)
        documents.append(new_doc)
    return documents


def askWithEmbedding(question, chatHistory):
    # Empty msg
    if not question.strip():
        return "", chatHistory

    vectordb = Chroma(persist_directory=persistDirectory, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_type="mmr")
    docs = retriever.get_relevant_documents(question)
    chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    ans = result['output_text']
    chatHistory.append((question, ans))

    return "", chatHistory


with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>Chat with OpenAI!</center></h1>""")
    # Declaring states
    chatWithOpenAIHistory = gr.State([])
    chatWithOpenAIChatBot = gr.Chatbot()
    chatWithOpenAIMessage = gr.Textbox()
    chatWithOpenAISubmit = gr.Button("SEND")
    chatWithOpenAISubmit.click(chatWithOpenAI, inputs=[chatWithOpenAIMessage, chatWithOpenAIChatBot],
                               outputs=[chatWithOpenAIMessage, chatWithOpenAIChatBot])

    gr.Markdown("""<h1><center>Chat with your online-connected bot!</center></h1>""")
    # Declaring states
    chatWithAgentHistory = gr.State([])
    chatbot = gr.Chatbot()
    message = gr.Textbox()
    submit = gr.Button("SEND")
    submit.click(chatWithAgent, inputs=[chatWithAgentHistory, message],
                 outputs=[chatbot, chatWithAgentHistory, message])

    gr.Markdown("""<h1><center>Ask anything about 《边城》！</center></h1>""")
    embeddingChatHistory = gr.State([])
    embeddingChatBot = gr.Chatbot()
    embeddingMessage = gr.Textbox()
    embeddingSubmit = gr.Button("ASK")
    embeddingSubmit.click(askWithEmbedding, inputs=[embeddingMessage, embeddingChatBot],
                          outputs=[embeddingMessage, embeddingChatBot])

if not os.path.exists(persistDirectory):
    with open("./边城.txt") as f:
        state_of_the_union = f.read()
        documents = splitParagraph(state_of_the_union, "边城.txt")
        vectordb = Chroma.from_documents(documents=documents, embedding=embedding,
                                         persist_directory=persistDirectory)
        vectordb.persist()
        vectordb = None

if __name__ == "__main__":
    demo.launch(debug=True)

