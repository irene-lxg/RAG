import json
import re
from pathlib import Path

import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


DATA_PATH = Path("ads_program_chunks.json")
EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM = "llama3"

ADMISSION_BOOST_TERMS = (
    "application requirements",
    "how to apply",
    "candidate statement",
    "resume",
    "transcript",
    "recommendation",
    "programming supplement",
    "virtual portfolio",
    "bachelor",
    "letters of recommendation",
)

EXPANSION_PHRASES = (
    "expand",
    "more detail",
    "more details",
    "tell me more",
    "elaborate",
    "go on",
    "what else",
    "anything else",
    "explain more",
    "can you say more",
    "say more",
)


def load_chunk_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if raw and isinstance(raw[0], dict):
        return raw

    return [{"text": item, "source": ""} for item in raw]


@st.cache_resource
def build_vector_db(chunk_records):
    texts = [item["text"] for item in chunk_records]
    metadatas = [{"source": item.get("source", "")} for item in chunk_records]
    embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embedding_model)


@st.cache_resource
def get_llm(model_name: str):
    return Ollama(model=model_name)


def doc_source(doc):
    return (doc.metadata or {}).get("source", "").lower()


def is_admission_question(question: str):
    q = question.lower()
    return any(
        term in q
        for term in (
            "admission",
            "apply",
            "application",
            "requirement",
            "eligible",
            "deadline",
            "transcript",
            "gre",
            "gmat",
            "toefl",
            "ielts",
        )
    )


def is_expansion_request(question: str):
    q = question.lower().strip()
    if q in {"more", "details", "more?", "details?"}:
        return True
    return any(phrase in q for phrase in EXPANSION_PHRASES)


def get_conversation_topic(history):
    last_user = ""
    last_assistant = ""
    for turn in reversed(history):
        if turn["role"] == "user" and not last_user:
            last_user = turn["content"]
        elif turn["role"] == "assistant" and not last_assistant:
            last_assistant = turn["content"][:300]
        if last_user and last_assistant:
            break
    return f"{last_user} {last_assistant}".strip()


def retrieve_context(question, vector_db, k=8, fetch_k=24):
    candidates = list(vector_db.similarity_search(question, k=fetch_k))

    if is_admission_question(question):
        admission_query = (
            "MS Applied Data Science application requirements transcripts resume "
            "candidate statement letters of recommendation programming supplement "
            "virtual portfolio how to apply"
        )
        candidates.extend(vector_db.similarity_search(admission_query, k=15))

    seen = set()
    unique = []
    for doc in candidates:
        key = doc.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(doc)

    query_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}

    def relevance(doc):
        text = doc.page_content.lower()
        source = doc_source(doc)
        score = sum(1 for w in query_words if w in text)

        if is_admission_question(question):
            if "how-to-apply" in source:
                score += 6
            if "faqs" in source and "application" in text:
                score += 2
            score += sum(2 for term in ADMISSION_BOOST_TERMS if term in text)

        return score

    unique.sort(key=relevance, reverse=True)
    return unique[:k]


def format_chat_history(history, max_turns=4):
    if not history:
        return "(No previous conversation.)"
    lines = []
    for turn in history[-(max_turns * 2) :]:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def make_retrieval_query(question, history, max_turns=4):
    if is_expansion_request(question) and history:
        topic = get_conversation_topic(history)
        return f"{topic} detailed requirements information"

    if not history:
        return question

    recent_user = [
        turn["content"]
        for turn in history[-(max_turns * 2) :]
        if turn["role"] == "user"
    ]
    return " ".join(recent_user[-2:] + [question])


def format_program_context(docs):
    return "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())


def extract_verbatim_course_titles(context):
    titles = set()
    for match in re.finditer(
        r"(?:Core|Elective|Seminar|Foundational)\s+(.+?)(?:\s+Letter Grade|\s+Pass/Fail|\s+noncredit|:|\s+The |\s+You )",
        context,
        flags=re.IGNORECASE,
    ):
        title = match.group(1).strip()
        if 5 < len(title) < 90:
            titles.add(title)
    for match in re.finditer(
        r"(Machine Learning I{1,2}|Time Series Analysis and Forecasting|Career Seminar)",
        context,
        flags=re.IGNORECASE,
    ):
        titles.add(match.group(1).strip())
    return sorted(titles)


def build_answer_prompt(question, docs, history):
    context = format_program_context(docs)
    history_text = format_chat_history(history)
    course_titles = extract_verbatim_course_titles(context)
    course_hint = (
        ", ".join(course_titles)
        if course_titles
        else "(no specific course titles found in retrieved text)"
    )

    expansion_note = ""
    if is_expansion_request(question) and history:
        expansion_note = (
            "The user wants MORE DETAIL on the previous topic. "
            "Add new specifics from the program information and avoid repeating the prior response.\n"
        )

    return f"""You are a helpful assistant for the University of Chicago MS in Applied Data Science program.

Answer using ONLY the program information below. Do not use outside knowledge.
{expansion_note}
Style (very important):
- Write naturally for a prospective student.
- NEVER mention passages, context, documents, retrieval, or numbered sources.
- NEVER say "based on the provided information", "the passages mention", "according to the program information", etc.
- Do not describe your sources; just state the facts.
- Use the recent conversation for follow-ups (e.g. "expand more details", "what about part-time?").
- If something is missing, say: "I don't see that detail on the program website."
- If you cannot answer at all, say: "I don't have that information on the program website."

Course names (anti-hallucination):
- ONLY name courses that appear verbatim in the program information OR in this allowed list: {course_hint}
- Never invent electives (for example, do not make up "Financial Instruments and Markets" unless it appears above).
- For career advice, describe core/elective categories if specific titles are unavailable.

Program information:
{context}

Recent conversation:
{history_text}

Current question: {question}

Answer:"""


def answer_question(question, history, llm, vector_db, k, fetch_k):
    search_query = make_retrieval_query(question, history)
    docs = retrieve_context(search_query, vector_db=vector_db, k=k, fetch_k=fetch_k)
    prompt = build_answer_prompt(question, docs, history)
    response = llm.invoke(prompt).strip()
    return response, docs


def main():
    st.set_page_config(page_title="UChicago ADS RAG Chatbot", page_icon="🎓", layout="wide")
    st.title("🎓 UChicago MS ADS RAG Chatbot")
    st.caption("Ask questions about the UChicago MS in Applied Data Science program.")

    if not DATA_PATH.exists():
        st.error("`ads_program_chunks.json` not found. Run notebook Part 1 first.")
        st.stop()

    model_name = DEFAULT_LLM
    k = 8
    fetch_k = 24
    show_debug = False

    chunk_records = load_chunk_records(DATA_PATH)
    vector_db = build_vector_db(chunk_records)
    llm = get_llm(model_name.strip() or DEFAULT_LLM)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_prompt = st.chat_input("Ask about admissions, courses, deadlines, program format...")
    if not user_prompt:
        return

    with st.chat_message("user"):
        st.write(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    history_before_answer = st.session_state.messages[:-1]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, docs = answer_question(
                question=user_prompt,
                history=history_before_answer,
                llm=llm,
                vector_db=vector_db,
                k=k,
                fetch_k=fetch_k,
            )
        st.write(response)
        if show_debug:
            with st.expander("Retrieved context (debug)"):
                st.text(format_program_context(docs))
            with st.expander("Retrieved source URLs"):
                for i, doc in enumerate(docs, start=1):
                    src = (doc.metadata or {}).get("source", "(unknown)")
                    st.write(f"{i}. {src}")

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
