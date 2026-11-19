import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

print("Initializing components......")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    model="gpt-5.5",
    temperature=0,
)

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the questions based only on the following contexts:
    {context}

    Question: {question}

    Provide a detailed answer:    
    """
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# Implementation 1: Without LCEL - Simple approach
# =====================================================
def retrival_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats and then generates a response.
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)
    # Step 2: Format documents into context string
    context = format_docs(docs)
    # Step 3: Format the prompt with context and question
    message = prompt_template.format_messages(context=context, question=query)
    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(message)
    # Step 5: Return the content
    return response.content


# =====================================================
# Implementation 2: With LCEL - better approach
# =====================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrival chain using LCEL.'
    Returns a chain that can be invoked with {question:"..."}
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    query = "What is Pinecode in machine learning?"

    # ==========================================
    # Option 0: Raw invocation without RAG
    # ===========================================
    print("\n" + "=" * 70)
    print("Implementtion 0: Raw LLM Invocation without RAG:")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ==========================================
    # Option 1: Use implementation WITHOUT LCEL
    # ===========================================
    print("\n" + "=" * 70)
    print("Implementtion 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrival_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ==========================================
    # Option 2: Use implementation WITH LCEL
    # ===========================================
    print("\n" + "=" * 70)
    print("Implementtion 2: With LCEL")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
