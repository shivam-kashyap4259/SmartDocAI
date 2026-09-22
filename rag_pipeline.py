import numpy as np

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# 1. LOCAL EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# 2. LOCAL LLM
# ============================================================

LLM_MODEL = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):
    """
    Convert text into numerical embeddings.
    """

    if not texts:
        return []

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings.tolist()


# ============================================================
# 4. COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    vector_a = np.array(vector_a)

    vector_b = np.array(vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / denominator
    )


# ============================================================
# 5. SEARCH RELEVANT CHUNKS
# ============================================================

def search_chunks(
    query,
    chunks,
    top_k=3
):
    """
    Find the most relevant document chunks
    for the user's question.
    """

    if not query or not chunks:
        return []

    query_embedding = create_embeddings(
        [query]
    )[0]

    results = []

    for chunk in chunks:

        chunk_embedding = chunk.get(
            "embedding"
        )

        if chunk_embedding is None:
            continue

        score = cosine_similarity(
            query_embedding,
            chunk_embedding
        )

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": score
        })


    # Highest similarity first

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# 6. GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):
    """
    Generate a grounded answer using retrieved
    document content.
    """

    # --------------------------------------------------------
    # No retrieved information
    # --------------------------------------------------------

    if not retrieved_chunks:

        return (
            "I couldn't find this information "
            "in the uploaded documents."
        )


    # --------------------------------------------------------
    # Minimum relevance check
    # --------------------------------------------------------

    best_score = retrieved_chunks[0]["score"]

    MINIMUM_SCORE = 0.35

    if best_score < MINIMUM_SCORE:

        return (
            "I couldn't find this information "
            "in the uploaded documents."
        )


    # --------------------------------------------------------
    # Question-content relevance check
    # --------------------------------------------------------

    stop_words = {
        "what", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on",
        "to", "for", "and", "or", "do", "does",
        "did", "how", "why", "who", "which",
        "where", "when", "can", "could",
        "this", "that"
    }

    question_words = {
        word.strip(".,?!").lower()
        for word in question.split()
        if word.strip(".,?!").lower()
        not in stop_words
    }

    document_text = " ".join(
        chunk["text"]
        for chunk in retrieved_chunks
    ).lower()

    matching_words = [
        word
        for word in question_words
        if word in document_text
    ]

    if question_words and len(matching_words) == 0:

        return (
            "I couldn't find this information "
            "in the uploaded documents."
        )


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            chunk["text"]
        )

    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Focused prompt
    # --------------------------------------------------------

    prompt = f"""
Answer the question using only the context.

Write one complete sentence that directly answers
the question.

Do not answer with only a keyword.
Do not copy unrelated information.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=60,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )


    # --------------------------------------------------------
    # Decode answer
    # --------------------------------------------------------

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


    # --------------------------------------------------------
    # Validate generated answer
    # --------------------------------------------------------

    words = answer.split()


    # If the model produces a very short answer,
    # use the most relevant sentence from the document.

    if len(words) < 6:

        best_text = retrieved_chunks[0]["text"]

        sentences = best_text.replace(
            "•",
            "."
        ).split(".")


        question_words_for_overlap = {
            word.strip(".,?!").lower()
            for word in question.split()
            if word.strip(".,?!").lower()
            not in stop_words
        }


        best_sentence = ""

        best_overlap = 0


        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue


            sentence_words = set(
                sentence.lower().split()
            )


            overlap = len(
                question_words_for_overlap.intersection(
                    sentence_words
                )
            )


            if overlap > best_overlap:

                best_overlap = overlap

                best_sentence = sentence


        if best_sentence:

            answer = best_sentence


    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if not answer:

        return (
            "I couldn't find this information "
            "in the uploaded documents."
        )


    return answer
    # ============================================================
# 7. GENERATE DOCUMENT SUMMARY
# ============================================================

# ============================================================
# 7. GENERATE DOCUMENT SUMMARY
# ============================================================

def generate_summary(chunks):
    """
    Generate a concise summary using information
    from different parts of the uploaded document.
    """

    if not chunks:
        return (
            "No document content is available "
            "to generate a summary."
        )


    # --------------------------------------------------------
    # Select content from different parts of document
    # --------------------------------------------------------

    total_chunks = len(chunks)

    if total_chunks <= 6:

        selected_chunks = chunks

    else:

        # Take chunks from beginning, middle and end
        indices = [
            0,
            total_chunks // 5,
            (2 * total_chunks) // 5,
            (3 * total_chunks) // 5,
            (4 * total_chunks) // 5,
            total_chunks - 1
        ]

        selected_chunks = []

        used_indices = set()

        for index in indices:

            if (
                0 <= index < total_chunks
                and index not in used_indices
            ):

                selected_chunks.append(
                    chunks[index]
                )

                used_indices.add(index)


    # --------------------------------------------------------
    # Build summary context
    # --------------------------------------------------------

    context_parts = []

    for index, chunk in enumerate(
        selected_chunks,
        start=1
    ):

        context_parts.append(
            f"Section {index}:\n"
            f"{chunk['text']}"
        )


    context = "\n\n".join(
        context_parts
    )


    # Keep prompt within local model limits
    context = context[:6000]


    # --------------------------------------------------------
    # Summary prompt
    # --------------------------------------------------------

    prompt = f"""
Create a concise summary of the document content below.

Use ONLY the information provided in the document.

Focus on the main topics, concepts, processes,
and important points.

Do not mention the document title, author,
college name, or other header information
unless it is an important part of the content.

Do not invent information.

Write 4 to 6 clear sentences.

Document content:

{context}

Summary:
"""


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not summary:

        return (
            "Unable to generate a summary "
            "from the uploaded document."
        )


    return summary