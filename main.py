import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# .env 読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 1. ドキュメントをロード
loader = TextLoader("documents/example.txt", encoding="utf-8")
documents = loader.load()

# 2. 埋め込みモデルとベクトルストアを構築
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = Chroma.from_documents(documents, embedding=embeddings)

# 3. チャットモデルとRAGチェーン
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# 4. 質問
query = "この文書の主なポイントは何ですか？"
result = qa_chain.invoke(query)

# 5. 結果表示
print("🧠 回答:", result["result"])
print("📄 使用された文書:")
for doc in result["source_documents"]:
    print("-", doc.metadata["source"])
