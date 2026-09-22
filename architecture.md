# SmartDocAI — Architecture Diagram

```mermaid
flowchart TD
    A[User] --> B[Streamlit UI]

    B --> C[PDF / TXT Upload]
    C --> D[Document Parser - PyPDF]
    D --> E[Text Chunking]
    E --> F[Sentence Transformer - all-MiniLM-L6-v2]
    F --> G[Semantic Search - Cosine Similarity]

    B --> H[User Question]
    H --> G

    G --> I[Top Relevant Chunks]
    I --> J{Relevance Check}

    J -->|Relevant| K[FLAN-T5-small]
    J -->|Not Relevant| L[Information Not Found]

    K --> M[Grounded Answer]
    I --> N[Source + Page Reference]

    M --> O[Answer + Evidence + Sources]
    N --> O

    E --> P[Document Summary]
    P --> O