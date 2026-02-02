import asyncio
from typing import List
from langchain_core.documents import Document as LangChainDocument

from cat import hook, StrayCat, log, UserMessage, AgenticWorkflowOutput, RecallSettings, CheshireCat
from cat.services.memory.models import PointStruct, DocumentRecall
from cat.services.mixin import BotMixin

# global variables
hybrid_collection_names = ["declarative_hybrid", "episodic_hybrid"]
k = 5
k_prefetched = 10
threshold = 0.5


async def create_hybrid_collection_if_not_exists(collection_name: str, cat: BotMixin):
    dense_vector_name = "dense"
    sparse_vector_name = "sparse"
    await cat.vector_memory_handler.create_hybrid_collection(collection_name, dense_vector_name, sparse_vector_name)
    log.info("Hybrid collection created")


async def populate_hybrid_collection(hybrid_collection_name: str, stored_points: List[PointStruct], cat: BotMixin):
    global hybrid_collection_names
    if hybrid_collection_name not in hybrid_collection_names:
        return

    if not stored_points:
        return

    await cat.vector_memory_handler.add_points_to_tenant(collection_name=hybrid_collection_name, points=stored_points)
    log.info(f"Added {len(stored_points)} points to hybrid collection")


@hook(priority=99)
def before_cat_reads_message(user_message: UserMessage, cat) -> UserMessage:
    global k, threshold, k_prefetched

    settings = cat.mad_hatter.get_plugin().load_settings()
    k = settings["number_of_hybrid_items"]
    k_prefetched = settings["number_of_prefetched_items"]
    threshold = settings["hybrid_threshold"]

    return user_message


@hook(priority=99)
async def agent_fast_reply(cat: StrayCat) -> AgenticWorkflowOutput | None:
    global hybrid_collection_names

    user_message: str = cat.working_memory.user_message.text
    if not user_message.startswith("@hybrid"):
        return None

    if user_message == "@hybrid init":
        for hybrid_collection_name in hybrid_collection_names:
            # delete the hybrid collection if it exists
            await cat.vector_memory_handler.delete_collection(collection_name=hybrid_collection_name)
            log.info("Hybrid collection deleted")
            await create_hybrid_collection_if_not_exists(hybrid_collection_name, cat)

        return AgenticWorkflowOutput(output="Hybrid collection initialized.")

    if user_message == "@hybrid migrate":
        for hybrid_collection_name in hybrid_collection_names:
            points, _ = await cat.vector_memory_handler.get_all_tenant_points(
                collection_name=hybrid_collection_name.replace("_hybrid", ""), with_vectors=True
            )
            points = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points]
            await populate_hybrid_collection(hybrid_collection_name, points, cat)

        # add a 5-second wait time to ensure data is committed
        await asyncio.sleep(5)

        return AgenticWorkflowOutput(output="Hybrid collection populated.")

    return None


# hybrid collection management
@hook
async def after_cat_bootstrap(cat: CheshireCat):
    for hybrid_collection_name in hybrid_collection_names:
        await create_hybrid_collection_if_not_exists(hybrid_collection_name, cat)


@hook
async def after_rabbithole_stored_documents(source: str, stored_points: List[PointStruct], cat: BotMixin):
    global hybrid_collection_names

    collection_name = "declarative_hybrid" if isinstance(cat, CheshireCat) else "episodic_hybrid"
    await populate_hybrid_collection(collection_name, stored_points, cat)


@hook(priority=99)
def before_cat_recalls_memories(recall_config: RecallSettings, cat) -> RecallSettings:
    global k, threshold
    recall_config.k = k
    recall_config.threshold = threshold

    return recall_config


@hook(priority=99)
async def after_cat_recalls_memories(config: RecallSettings, cat) -> None:
    global hybrid_collection_names, k_prefetched, threshold, k

    user_message: UserMessage = cat.working_memory.user_message

    metadata = {}
    ## if there are tags in the user message, use them as a metadata filter
    if (
        hasattr(cat.working_memory.user_message, "tags")
        and cat.working_memory.user_message.tags
    ):
        metadata = user_message.tags

    finalized_metadata = config.metadata | metadata if config.metadata else metadata

    client = cat.vector_memory_handler
    memories = []
    for hybrid_collection_name in hybrid_collection_names:
        memories.extend(await client.search_prefetched_in_tenant(
            collection_name=hybrid_collection_name,
            query=cat.working_memory.user_message.text,
            query_vector=config.embedding,
            query_filter=client.filter_from_dict(finalized_metadata),
            k=k,
            k_prefetched=k_prefetched,
            threshold=threshold,
        ))

    # convert Qdrant points to langchain.Document
    langchain_documents_from_points = []
    for m in memories:
        langchain_documents_from_points.append(
            DocumentRecall(
                document=LangChainDocument(
                    page_content=m.payload.get("page_content"),
                    metadata=m.payload.get("metadata") or {},
                ),
                score=m.score,
                vector=m.vector,
                id=m.id,
            )
        )
    cat.working_memory.declarative_memories = langchain_documents_from_points
