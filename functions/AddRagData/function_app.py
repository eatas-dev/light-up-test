import json
import logging
import os

import azure.functions as func
import openai
from bson import ObjectId
from langchain.docstore.document import Document
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from langchain_openai import AzureOpenAIEmbeddings
from pymongo import MongoClient
from pymongo.collection import Collection

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
# Azure Cosmos DBの接続情報を設定
database_name = os.getenv("AZURE_COSMOS_DB_NAME", "")
collection_name = os.getenv("AZURE_COSMOS_COLLECTION_NAME", "")
connection_string = os.getenv("AZURE_COSMOS_CONNECTION_STRING", "")
mongo_username = os.getenv("AZURE_COSMOS_USERNAME", "")
mongo_password = os.getenv("AZURE_COSMOS_PASSWORD", "")
connection_string = connection_string.replace("<user>", mongo_username).replace("<password>", mongo_password)

# MongoDBクライアントを作成し、指定されたデータベースとコレクションに接続
mongo_client: MongoClient = MongoClient(connection_string)
db = mongo_client[database_name]
collection = db[collection_name]


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


async def generate_embeddings_and_add_data(
    documents: list[Document],
    collection: Collection,
    index_name: str,
    embeddings: AzureOpenAIEmbeddings,
) -> AzureCosmosDBVectorSearch:
    # データから埋め込みを生成し、データベースに保存してMongoDB vCoreへの接続を返す
    return await AzureCosmosDBVectorSearch.afrom_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=index_name,
    )


async def add_data(azure_openai_embeddings) -> None:
    # コレクション内のすべてのドキュメントを取得し、リストに追加
    documents = []
    for idx, item in enumerate(collection.find()):
        documents.append(
            Document(
                page_content=json.dumps(item, ensure_ascii=False, cls=JSONEncoder),
                metadata={"source": "mongodb", "seq_num": idx + 1},
            )
        )

    # データから埋め込みを生成し、データベースに保存してMongoDB vCoreへの接続を返す
    vector_store = await generate_embeddings_and_add_data(
        documents=documents,
        collection=collection,
        index_name=os.getenv("AZURE_COSMOS_INDEX_NAME", ""),
        embeddings=azure_openai_embeddings,
    )

    # クエリ条件: metadataキーが存在しないドキュメントを検索
    query = {"metadata": {"$exists": False}}

    # 条件に一致するすべてのドキュメントを削除
    collection.delete_many(query)

    # これらの変数の詳細については、以下のリンクを参照してください。 https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search
    num_lists = 100
    dimensions = 1536
    similarity_algorithm = CosmosDBSimilarityType.COS
    kind = CosmosDBVectorSearchType.VECTOR_HNSW
    m = 16
    ef_construction = 64

    # コレクションに対してHNSWインデックスを作成
    vector_store.create_index(num_lists, dimensions, similarity_algorithm, kind, m, ef_construction)


@app.route(route="Add_Rag_Data_To_ETL")
async def Add_Rag_Data_To_ETL(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # OpenAIの埋め込みモデルとデプロイメント名を設定
    openai.api_type = "azure"
    openai.base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    openai.api_version = os.getenv("OPENAI_API_VERSION", "2023-09-15-preview")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

    openai_embeddings_model = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")
    openai_embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding")

    # Azure OpenAIの埋め込みクライアントを初期化
    azure_openai_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        model=openai_embeddings_model,
        azure_deployment=openai_embeddings_deployment,
    )

    await add_data(azure_openai_embeddings)

    return func.HttpResponse("Add Data to ETL")


@app.route(route="Delete_Data", auth_level=func.AuthLevel.FUNCTION)
def Delete_Data(req: func.HttpRequest) -> func.HttpResponse:
    # コレクション内のすべてのドキュメントを削除
    collection.delete_many({})
    return func.HttpResponse("Detele Data")
