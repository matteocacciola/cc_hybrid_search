from cat.looking_glass.stray_cat import StrayCat
from cat.mad_hatter.decorators import hook
from cat.log import log
from typing import Dict
import time
from langchain.docstore.document import Document as LangChainDocument

from cat.memory.utils import Document
from cat.utils import run_sync_or_async

# global variables
hybrid_collection_name = "declarative_hybrid"
k = 5
k_prefetched = 10
threshold = 0.5


@hook(priority=99)
def before_cat_reads_message(user_message_json, cat):
    global k, threshold, k_prefetched
    settings = cat.mad_hatter.get_plugin().load_settings()
    k = settings["number_of_hybrid_items"]
    k_prefetched = settings["number_of_prefetched_items"]
    threshold = settings["hybrid_threshold"]
    return user_message_json


@hook(priority=99)
def agent_fast_reply(fast_reply: Dict, cat: StrayCat) -> Dict:
    global hybrid_collection_name

    user_message: str = cat.working_memory.user_message_json.text
    if not user_message.startswith("@hybrid"):
        return fast_reply

    if user_message == "@hybrid init":
        run_sync_or_async(delete_hybrid_collection_if_exists, cat, hybrid_collection_name)
        run_sync_or_async(create_hybrid_collection_if_not_exists, cat, hybrid_collection_name)

        fast_reply["output"] = "Hybrid collection initialized."
    elif user_message == "@hybrid migrate":
        points = run_sync_or_async(get_declarative_points, cat)
        run_sync_or_async(populate_hybrid_collection, points, cat)

        # add 5-second wait time to ensure data is committed
        time.sleep(5)

        fast_reply["output"] = "Hybrid collection populted."

    return fast_reply


async def get_declarative_points(cat):
    return await cat.vector_memory_handler.get_all_points(collection_name="declarative", with_vectors=True)


# hybrid collection management
@hook
def after_cat_bootstrap(cat: StrayCat):
    run_sync_or_async(create_hybrid_collection_if_not_exists, cat, hybrid_collection_name)


async def create_hybrid_collection_if_not_exists(cat, collection_name):
    dense_vector_name = "dense"
    sparse_vector_name = "sparse"
    await cat.vector_memory_handler.create_hybrid_collection(collection_name, dense_vector_name, sparse_vector_name)
    log.info("Hybrid collection created")


async def delete_hybrid_collection_if_exists(cat, collection_name):
    await cat.vector_memory_handler.delete_collection(collection_name=collection_name)
    log.info("Hybrid collection deleted")


@hook
def after_rabbithole_stored_documents(source, stored_points, cat):
    run_sync_or_async(populate_hybrid_collection, stored_points, cat)


async def populate_hybrid_collection(stored_points, cat):
    global hybrid_collection_name

    points_ids = []
    points_vectors = []
    points_payloads = []
    for p in stored_points:
        points_ids.append(p.id)
        points_vectors.append(
            {"dense": p.vector, "sparse": Document(text=p.payload.get("page_content", ""), model="Qdrant/bm25")}
        )
        points_payloads.append(p.payload)

    await cat.vector_memory_handler.add_points(
        collection_name=hybrid_collection_name, payloads=points_payloads, vectors=points_vectors, ids=points_ids
    )

    log.info(f"Added {len(points_ids)} points to hybrid collection")


async def search_hybrid_collection(query, k, k_prefetched, threshold, metadata, cat):
    global hybrid_collection_name
    client = cat.vector_memory_handler
    dense_embedding = cat.embedder.embed_query(query)
    search_result = await client.search_prefetched(
        collection_name=hybrid_collection_name,
        query=query,
        query_vector=dense_embedding,
        query_filter=client.filter_from_dict(metadata),
        k=k,
        k_prefetched=k_prefetched,
        threshold=threshold,
    )

    return search_result


@hook(priority=99)
def before_cat_recalls_memories(
    declarative_recall_config: dict, cat
) -> dict:
    global k, threshold
    declarative_recall_config["k"] = k
    declarative_recall_config["threshold"] = threshold
    return declarative_recall_config


@hook(priority=99)
def after_cat_recalls_memories(cat) -> None:
    global k, k_prefetched, threshold
    metadata = {}
    ## if there are tags in the user message, use them as metadata filter
    if (
        hasattr(cat.working_memory.user_message_json, "tags")
        and cat.working_memory.user_message_json.tags
    ):
        metadata = cat.working_memory.user_message_json.tags
    memories = run_sync_or_async(
        search_hybrid_collection,
        cat.working_memory.recall_query,
        k,
        k_prefetched,
        threshold,
        metadata,
        cat,
    )
    # convert Qdrant points to langchain.Document
    langchain_documents_from_points = []
    for m in memories:
        langchain_documents_from_points.append(
            (
                LangChainDocument(
                    page_content=m.payload.get("page_content"),
                    metadata=m.payload.get("metadata") or {},
                ),
                m.score,
                m.vector,
                m.id,
            )
        )
    cat.working_memory.declarative_memories = langchain_documents_from_points
