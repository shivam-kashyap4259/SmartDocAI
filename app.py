import streamlit as st

from document_processor import extract_document
from chunking import chunk_text

from rag_pipeline import (
    create_embeddings,
    search_chunks,
    generate_answer,
    generate_summary
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📚 SmartDoc AI")

st.markdown(
    """
    **AI-powered document question answering using RAG**

    Upload one or more PDF/TXT documents, ask questions,
    and get grounded answers with source and page references.
    """
)

st.info(
    "💡 SmartDoc AI answers questions using only the "
    "information found in your uploaded documents."
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} document(s) selected."
    )

    all_chunks = []


    for uploaded_file in uploaded_files:

        st.write(
            f"### 📄 {uploaded_file.name}"
        )

        try:

            # ------------------------------------------------
            # 1. Read uploaded file
            # ------------------------------------------------

            file_bytes = uploaded_file.getvalue()


            # ------------------------------------------------
            # 2. Extract text
            # ------------------------------------------------

            pages = extract_document(
                file_bytes,
                uploaded_file.name
            )

            st.write(
                f"Extracted {len(pages)} page/text section(s)."
            )


            # ------------------------------------------------
            # 3. Create searchable chunks
            # ------------------------------------------------

            document_chunks = []

            for page in pages:

                page_chunks = chunk_text(
                    page["text"]
                )

                for chunk in page_chunks:

                    document_chunks.append({
                        "text": chunk,
                        "source": page["source"],
                        "page": page["page"]
                    })


            st.write(
                f"Created {len(document_chunks)} searchable chunks."
            )


            # ------------------------------------------------
            # 4. Create embeddings
            # ------------------------------------------------

            if document_chunks:

                chunk_texts = [
                    chunk["text"]
                    for chunk in document_chunks
                ]

                embeddings = create_embeddings(
                    chunk_texts
                )


                for index, embedding in enumerate(
                    embeddings
                ):

                    document_chunks[index][
                        "embedding"
                    ] = embedding


            all_chunks.extend(
                document_chunks
            )


            # ------------------------------------------------
            # 5. Extracted text
            # ------------------------------------------------

            with st.expander(
                f"📖 View extracted text — {uploaded_file.name}"
            ):

                if pages:

                    for page in pages:

                        if page["page"]:

                            st.markdown(
                                f"**Page {page['page']}**"
                            )

                        st.write(
                            page["text"]
                        )

                        st.divider()

                else:

                    st.warning(
                        "No text could be extracted from this document."
                    )


            # ------------------------------------------------
            # 6. Searchable chunks
            # ------------------------------------------------

            with st.expander(
                f"🔎 View searchable chunks — {uploaded_file.name}"
            ):

                for index, chunk in enumerate(
                    document_chunks,
                    start=1
                ):

                    page_number = chunk["page"]

                    if page_number:

                        location = (
                            f"Page {page_number}"
                        )

                    else:

                        location = "Text document"


                    st.markdown(
                        f"**Chunk {index} — {location}**"
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()


        except Exception as error:

            st.error(
                f"❌ Error processing "
                f"{uploaded_file.name}: {error}"
            )


    # ========================================================
    # EMBEDDING STATUS
    # ========================================================

    if all_chunks:

        st.success(
            f"✅ {len(all_chunks)} chunks converted "
            "into semantic embeddings."
        )


    # ========================================================
    # DOCUMENT SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📝 Document Summary")

    st.caption(
        "Generate a concise summary using the uploaded "
        "document content."
    )


    if st.button(
        "✨ Generate Summary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating document summary..."
        ):

            summary = generate_summary(
                all_chunks
            )


        st.markdown(
            "### 📋 Summary"
        )

        st.write(
            summary
        )


    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    st.divider()

    st.subheader("🔎 Ask a Question")

    question = st.text_input(
        "Enter your question about the uploaded documents:",
        placeholder="Example: What role do ministries of labour play in e-waste management?"
    )


    # ========================================================
    # RETRIEVAL + ANSWER
    # ========================================================

    if question.strip():

        with st.spinner(
            "🔍 Searching the documents..."
        ):

            results = search_chunks(
                question,
                all_chunks,
                top_k=3
            )


        # ----------------------------------------------------
        # Generate grounded answer
        # ----------------------------------------------------

        answer = generate_answer(
            question,
            results
        )


        # ----------------------------------------------------
        # AI ANSWER
        # ----------------------------------------------------

        st.subheader("🤖 AI Answer")

        st.success(
            answer
        )


        # ====================================================
        # EVIDENCE / SOURCES
        # ====================================================

        if not results:

            st.warning(
                "No relevant information was found "
                "in the uploaded documents."
            )


        else:

            st.subheader(
                "📚 Evidence & Sources"
            )

            st.caption(
                "The answer is based on the following "
                "retrieved document sections."
            )


            for index, result in enumerate(
                results,
                start=1
            ):

                score = result["score"]

                page = result["page"]

                source = result["source"]


                # ------------------------------------------------
                # Location
                # ------------------------------------------------

                if page:

                    location = (
                        f"Page {page}"
                    )

                else:

                    location = (
                        "Text document"
                    )


                # ------------------------------------------------
                # Evidence heading
                # ------------------------------------------------

                st.markdown(
                    f"### 📌 Evidence {index}"
                )


                # ------------------------------------------------
                # Source information
                # ------------------------------------------------

                col1, col2 = st.columns(2)


                with col1:

                    st.write(
                        f"📄 **Document:** {source}"
                    )


                with col2:

                    st.write(
                        f"📍 **Location:** {location}"
                    )


                # ------------------------------------------------
                # Similarity score
                # ------------------------------------------------

                safe_score = min(
                    max(
                        float(score),
                        0.0
                    ),
                    1.0
                )


                st.progress(
                    safe_score
                )


                st.caption(
                    f"🎯 Semantic similarity: {score:.3f}"
                )


                # ------------------------------------------------
                # Supporting content
                # ------------------------------------------------

                with st.expander(
                    "🔎 View supporting content"
                ):

                    st.write(
                        result["text"]
                    )


                if index < len(results):

                    st.divider()