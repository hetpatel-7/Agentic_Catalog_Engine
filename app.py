"""Gradio entrypoint for Agentic Catalog Engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import tempfile
import uuid
from typing import Any, Literal, Protocol, TypedDict, cast
from typing import List

import gradio as gr
from pydantic import BaseModel

from utils.config import AppConfig, load_config


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

CatalogJson = dict[str, Any]
MAX_CHAT_TURNS = 5
RETRIEVER_REGISTRY: dict[str, "RetrieverLike"] = {}


class UploadedFileLike(Protocol):
    """Minimal upload interface expected by the ingestion router."""

    name: str
    type: str
    size: int

    def getvalue(self) -> bytes:
        """Return uploaded file contents as bytes."""
        ...


class RetrieverLike(Protocol):
    """Minimal LangChain retriever interface used by the chat sandbox."""

    def invoke(
        self,
        input: str,
        config: Any | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Retrieve documents relevant to the user query."""
        ...


class ChatMessage(TypedDict):
    """Gradio chatbot message in messages format."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class UploadedFilePath:
    """Adapter that gives a Gradio filepath the upload interface ACE expects."""

    path: Path

    @property
    def name(self) -> str:
        """Return the uploaded file name."""
        return self.path.name

    @property
    def type(self) -> str:
        """Return a lightweight MIME hint based on the file suffix."""
        suffix_to_type = {
            ".pdf": "application/pdf",
            ".docx": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            ".txt": "text/plain",
        }
        return suffix_to_type.get(self.path.suffix.lower(), "application/octet-stream")

    @property
    def size(self) -> int:
        """Return the uploaded file size in bytes."""
        return self.path.stat().st_size

    def getvalue(self) -> bytes:
        """Return uploaded file contents as bytes."""
        return self.path.read_bytes()


class CameraGearCatalogEntry(BaseModel):
    """Structured FilmTools camera gear catalog payload."""

    brand: str
    model: str
    sensor_size: str
    lens_mount: str
    supported_resolutions: List[str]
    weight_grams: float
    power_consumption_watts: float


def _write_upload_to_temp_file(uploaded_file: UploadedFileLike, suffix: str) -> Path:
    """Persist uploaded bytes to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return Path(temp_file.name)


def route_and_parse_file(uploaded_file: UploadedFileLike) -> str:
    """Route an uploaded file to the appropriate parser and return clean text."""
    config = load_config()
    file_name = uploaded_file.name
    suffix = Path(file_name).suffix.lower()
    temp_path: Path | None = None

    if suffix not in {".pdf", ".docx", ".txt"}:
        raise ValueError(f"Unsupported file type for `{file_name}`.")

    try:
        temp_path = _write_upload_to_temp_file(uploaded_file, suffix)

        if suffix == ".pdf":
            if not config.llama_cloud_api_key:
                raise ValueError("LLAMA_CLOUD_API_KEY is required to parse PDF files.")

            from llama_parse import LlamaParse

            LOGGER.info("Detected PDF. Routing to LlamaParse with markdown output.")
            parser = LlamaParse(
                api_key=config.llama_cloud_api_key,
                result_type="markdown",
            )
            documents = parser.load_data(str(temp_path))
            parsed_text = "\n\n".join(
                getattr(document, "text", str(document)) for document in documents
            ).strip()
        else:
            from unstructured.partition.auto import partition

            LOGGER.info(
                "Detected %s file. Routing to Unstructured partitioner.",
                suffix.upper(),
            )
            elements = partition(filename=str(temp_path))
            parsed_text = "\n".join(str(element) for element in elements).strip()

        if not parsed_text:
            raise ValueError(f"No text could be extracted from `{file_name}`.")

        return parsed_text
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def extract_structured_catalog(text_data: str) -> CatalogJson:
    """Extract a Pydantic-validated catalog JSON payload from parsed text."""
    config = load_config()

    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for structured extraction.")

    if not text_data.strip():
        raise ValueError("Parsed text is empty; structured extraction cannot run.")

    import instructor
    from openai import OpenAI

    LOGGER.info("Initializing Instructor-patched OpenAI client.")
    client = instructor.from_openai(OpenAI(api_key=config.openai_api_key))

    LOGGER.info("Requesting schema-validated camera gear attributes.")
    extracted_entry = client.chat.completions.create(
        model=config.openai_chat_model,
        response_model=CameraGearCatalogEntry,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert technical data extractor. Your task is "
                    "to accurately extract the exact camera specifications from "
                    "the provided unstructured manufacturer manual text."
                ),
            },
            {
                "role": "user",
                "content": text_data,
            },
        ],
    )

    return extracted_entry.model_dump()


def _format_catalog_value(value: Any) -> str:
    """Convert a catalog field value into compact text for embedding."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)

    return str(value)


def _catalog_json_to_snippets(json_data: Mapping[str, Any]) -> list[str]:
    """Create retrieval-friendly document strings from catalog JSON."""
    field_snippets = [
        f"{field_name.replace('_', ' ').title()}: {_format_catalog_value(value)}"
        for field_name, value in json_data.items()
    ]
    summary = "Camera catalog entry. " + " ".join(field_snippets)
    return [summary, *field_snippets]


def _pinecone_retriever_or_none(
    json_data: Mapping[str, Any],
    config: AppConfig,
    embeddings: Any,
) -> RetrieverLike | None:
    """Create a Pinecone retriever when a configured index already exists."""
    if not config.pinecone_api_key:
        return None

    try:
        from langchain_core.documents import Document
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        pinecone_client = Pinecone(api_key=config.pinecone_api_key)
        index_names = pinecone_client.list_indexes().names()

        if config.pinecone_index_name not in index_names:
            LOGGER.info(
                "Pinecone credentials found, but index %s does not exist. "
                "Falling back to local Chroma.",
                config.pinecone_index_name,
            )
            return None

        documents = [
            Document(
                page_content=snippet,
                metadata={
                    "source": "catalog_json",
                    "field_count": len(json_data),
                },
            )
            for snippet in _catalog_json_to_snippets(json_data)
        ]
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=config.pinecone_index_name,
        )
        LOGGER.info("Indexed catalog snippets in Pinecone %s.", config.pinecone_index_name)
        return vector_store.as_retriever(search_kwargs={"k": min(4, len(documents))})
    except Exception as error:
        LOGGER.warning("Pinecone initialization failed; falling back to Chroma: %s", error)
        return None


def initialize_semantic_chatbot(json_data: Mapping[str, Any]) -> RetrieverLike:
    """Initialize a retriever over the structured catalog JSON."""
    config = load_config()

    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize semantic search.")

    if not json_data:
        raise ValueError("Catalog JSON is empty; semantic search cannot be initialized.")

    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    snippets = _catalog_json_to_snippets(json_data)
    documents = [
        Document(
            page_content=snippet,
            metadata={
                "source": "catalog_json",
                "snippet_index": index,
            },
        )
        for index, snippet in enumerate(snippets)
    ]

    LOGGER.info("Embedding catalog snippets with OpenAI.")
    embeddings = OpenAIEmbeddings(
        model=config.openai_embedding_model,
        api_key=config.openai_api_key,
    )

    pinecone_retriever = _pinecone_retriever_or_none(
        json_data=json_data,
        config=config,
        embeddings=embeddings,
    )
    if pinecone_retriever is not None:
        return pinecone_retriever

    LOGGER.info("Using in-memory Chroma vector store for local sandboxing.")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"ace_catalog_sandbox_{uuid.uuid4().hex}",
    )
    return vector_store.as_retriever(search_kwargs={"k": min(4, len(documents))})


def _format_retrieved_context(documents: list[Any]) -> str:
    """Format retrieved LangChain documents for model context."""
    return "\n\n".join(
        getattr(document, "page_content", str(document)) for document in documents
    )


def _format_chat_history(chat_history: list[ChatMessage]) -> str:
    """Format bounded chat history for the RAG prompt."""
    if not chat_history:
        return "No previous conversation."

    return "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in chat_history
    )


def _enforce_fifo_chat_limit(chat_history: list[ChatMessage]) -> list[ChatMessage]:
    """Keep only the newest complete interaction turns."""
    limited_history = list(chat_history)

    while (
        sum(1 for message in limited_history if message["role"] == "user")
        > MAX_CHAT_TURNS
    ):
        if (
            len(limited_history) >= 2
            and limited_history[0]["role"] == "user"
            and limited_history[1]["role"] == "assistant"
        ):
            del limited_history[:2]
        else:
            del limited_history[0]

    return limited_history


def answer_catalog_question(
    question: str,
    retriever: RetrieverLike,
    chat_history: list[ChatMessage],
    config: AppConfig,
) -> str:
    """Answer a catalog question using retrieved context and bounded history."""
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for chat generation.")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    documents = retriever.invoke(question)
    context = _format_retrieved_context(documents)
    formatted_history = _format_chat_history(chat_history)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are ACE, a precise technical catalog assistant. Answer only "
                    "from the provided catalog context and bounded chat history. If the "
                    "answer is unavailable, say that the extracted catalog entry does "
                    "not contain that specification.\n\n"
                    "Catalog context:\n{context}\n\n"
                    "Bounded chat history:\n{chat_history}"
                ),
            ),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(
        model=config.openai_chat_model,
        api_key=config.openai_api_key,
        temperature=0,
    )
    chain = prompt | llm | StrOutputParser()
    return cast(
        str,
        chain.invoke(
            {
                "context": context,
                "chat_history": formatted_history,
                "question": question,
            }
        ),
    )


def _coerce_file_path(uploaded_file: str | Path | None) -> Path:
    """Normalize a Gradio file upload into a filesystem path."""
    if uploaded_file is None:
        raise ValueError("Upload a PDF, DOCX, or TXT manual before running extraction.")

    upload_path = Path(uploaded_file)
    if not upload_path.exists():
        raise ValueError(f"Uploaded file is unavailable: {upload_path}")

    return upload_path


def run_extraction_pipeline(
    uploaded_file: str | Path | None,
) -> tuple[CatalogJson | None, str | None, list[ChatMessage], str]:
    """Run ingestion, structured extraction, and semantic indexing."""
    try:
        upload = UploadedFilePath(_coerce_file_path(uploaded_file))
        parsed_text = route_and_parse_file(upload)
        catalog_json = extract_structured_catalog(parsed_text)
        retriever = initialize_semantic_chatbot(catalog_json)
        retriever_id = uuid.uuid4().hex
        RETRIEVER_REGISTRY[retriever_id] = retriever
        return (
            catalog_json,
            retriever_id,
            [],
            "Extraction complete. Semantic chat sandbox is ready.",
        )
    except Exception as error:
        return None, None, [], f"Pipeline failed: {error}"


def submit_chat_message(
    question: str,
    retriever_id: str | None,
    chat_history: list[ChatMessage] | None,
) -> tuple[str, list[ChatMessage], list[ChatMessage], str]:
    """Answer a Gradio chat message with FIFO bounded memory."""
    current_history = chat_history or []
    trimmed_history = _enforce_fifo_chat_limit(current_history)

    if not question.strip():
        return "", trimmed_history, trimmed_history, "Enter a question to chat."

    retriever = RETRIEVER_REGISTRY.get(retriever_id or "")
    if retriever is None:
        assistant_message = "Run the extraction pipeline before using the chat sandbox."
        next_history = _enforce_fifo_chat_limit(
            [
                *trimmed_history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        return "", next_history, next_history, assistant_message

    try:
        answer = answer_catalog_question(
            question=question,
            retriever=retriever,
            chat_history=trimmed_history,
            config=load_config(),
        )
    except Exception as error:
        answer = f"Chat generation failed: {error}"

    next_history = _enforce_fifo_chat_limit(
        [
            *trimmed_history,
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    return "", next_history, next_history, answer


def reset_chat() -> tuple[list[ChatMessage], list[ChatMessage], str]:
    """Clear the visible chat and backing chat state."""
    return [], [], "Chat history cleared."


def build_interface() -> gr.Blocks:
    """Build the Hugging Face Spaces Gradio interface."""
    config = load_config()

    with gr.Blocks(title="Agentic Catalog Engine") as demo:
        retriever_state = gr.State(value=None)
        chat_history_state = gr.State(value=[])

        gr.Markdown("# Agentic Catalog Engine")
        gr.Markdown(
            "Upload a manufacturer manual, extract a validated camera catalog entry, "
            "then ask bounded RAG questions over the result."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Ingestion & Extraction")
                upload_input = gr.File(
                    label="Equipment manual",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="filepath",
                )
                extract_button = gr.Button("Run extraction pipeline", variant="primary")
                pipeline_status = gr.Textbox(
                    label="Pipeline status",
                    value=(
                        "Ready. "
                        f"Chat model: {config.openai_chat_model}; "
                        f"Embeddings: {config.openai_embedding_model}; "
                        f"Pinecone index: {config.pinecone_index_name}."
                    ),
                    interactive=False,
                )
                catalog_output = gr.JSON(label="Validated JSON Database Output")

            with gr.Column(scale=1):
                gr.Markdown("## Semantic Chat Sandbox")
                chatbot = gr.Chatbot(
                    label="ACE Catalog Chat",
                    type="messages",
                    height=520,
                )
                question_input = gr.Textbox(
                    label="Ask about this gear catalog entry",
                    placeholder=(
                        "What is the lens mount of this camera? "
                        "How much power does this model draw?"
                    ),
                    lines=2,
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear chat")
                chat_status = gr.Textbox(label="Chat status", interactive=False)

        extract_button.click(
            fn=run_extraction_pipeline,
            inputs=[upload_input],
            outputs=[
                catalog_output,
                retriever_state,
                chat_history_state,
                pipeline_status,
            ],
        ).then(
            fn=lambda: [],
            inputs=None,
            outputs=[chatbot],
        )

        question_input.submit(
            fn=submit_chat_message,
            inputs=[question_input, retriever_state, chat_history_state],
            outputs=[question_input, chatbot, chat_history_state, chat_status],
        )
        send_button.click(
            fn=submit_chat_message,
            inputs=[question_input, retriever_state, chat_history_state],
            outputs=[question_input, chatbot, chat_history_state, chat_status],
        )
        clear_button.click(
            fn=reset_chat,
            inputs=None,
            outputs=[chatbot, chat_history_state, chat_status],
        )

    return demo


if __name__ == "__main__":
    build_interface().launch()
