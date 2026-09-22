# 📚 SmartDoc AI

SmartDoc AI is a GenAI-powered document question-answering application
that allows users to upload PDF or TXT documents and ask questions
about their contents.

The application uses a Retrieval-Augmented Generation (RAG) pipeline
to retrieve relevant document content before generating an answer.
This helps keep responses grounded in the uploaded documents.

---

## 🚀 Features

- Upload one or more PDF/TXT documents
- Extract text from uploaded documents
- Split extracted text into searchable chunks
- Generate semantic embeddings locally
- Perform semantic similarity search
- Retrieve the most relevant document sections
- Generate grounded answers using a local LLM
- Display source document and page references
- Show retrieved supporting evidence
- Prevent unsupported answers using relevance checks
- Generate an automatic document summary
- Simple and interactive Streamlit interface

---

## 🧠 How SmartDoc AI Works

The application follows a Retrieval-Augmented Generation (RAG)
architecture.

### Data Flow

User
↓
Streamlit Interface
↓
PDF/TXT Document Upload
↓
Text Extraction
↓
Text Chunking
↓
Local Embeddings
↓
Semantic Similarity Search
↓
Relevant Document Chunks
↓
Local LLM
↓
Grounded Answer + Sources

---

## 🏗️ Architecture

### Main Components

1. **Streamlit UI**
   - Handles document uploads
   - Accepts user questions
   - Displays answers and sources
   - Provides document summary functionality

2. **Document Processor**
   - Extracts text from PDF files using PyPDF
   - Reads TXT files
   - Preserves page information for PDF documents

3. **Chunking Module**
   - Splits extracted text into smaller searchable chunks
   - Uses overlapping chunks to preserve context between sections

4. **Embedding Model**
   - Uses `all-MiniLM-L6-v2`
   - Converts document chunks and user queries into numerical vectors

5. **Retrieval Layer**
   - Calculates cosine similarity between query and document embeddings
   - Retrieves the most relevant document chunks

6. **Local Language Model**
   - Uses `google/flan-t5-small`
   - Generates answers using retrieved document context

7. **Source Attribution**
   - Displays source document
   - Displays page number when available
   - Displays similarity score
   - Allows users to inspect supporting content

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web interface |
| PyPDF | PDF text extraction |
| Sentence Transformers | Semantic embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Transformers | Local LLM integration |
| FLAN-T5-small | Answer generation |
| NumPy | Vector and similarity calculations |
| python-dotenv | Environment variable management |

---

## 🤖 AI / GenAI Components

### Embedding Model

`all-MiniLM-L6-v2`

This model converts document chunks and user questions into
384-dimensional numerical embeddings.

These embeddings allow the application to compare the semantic
relationship between a question and document content.

### Language Model

`google/flan-t5-small`

The model generates a concise response using retrieved document
content as context.

The prompt instructs the model to:

- Use only the provided document context
- Avoid outside knowledge
- Avoid inventing information
- Provide a direct answer

---

## 🔍 Retrieval-Augmented Generation

SmartDoc AI uses the following RAG process:

### Step 1 — Document Upload

The user uploads one or more PDF/TXT documents.

### Step 2 — Text Extraction

Text is extracted from each document.

For PDF files, page numbers are preserved.

### Step 3 — Chunking

Large document text is divided into smaller overlapping chunks.

This makes individual sections easier to search.

### Step 4 — Embeddings

Each chunk is converted into a numerical embedding using
`all-MiniLM-L6-v2`.

### Step 5 — Query Embedding

When the user asks a question, the question is also converted
into an embedding.

### Step 6 — Similarity Search

Cosine similarity is used to compare the question embedding
with document chunk embeddings.

The highest-scoring chunks are retrieved.

### Step 7 — Grounded Generation

The retrieved chunks are passed to FLAN-T5-small as context.

The model generates an answer based on that context.

### Step 8 — Source Display

The application displays:

- Source document
- Page number
- Semantic similarity
- Supporting retrieved content

---

## 🛡️ Hallucination Handling

SmartDoc AI uses multiple safeguards to reduce hallucination.

### 1. Similarity Threshold

Retrieved content must pass a minimum similarity threshold.

If the retrieved information is not sufficiently relevant,
the application returns:

> I couldn't find this information in the uploaded documents.

### 2. Question-Content Relevance Check

The application also checks whether meaningful words from the
question occur in the retrieved document content.

If no meaningful overlap is found, the application does not
generate an unsupported answer.

### 3. Strict Prompting

The LLM is instructed to:

- Use only document context
- Avoid outside knowledge
- Avoid inventing facts
- Provide a direct answer

### 4. Evidence Display

Retrieved document sections are shown to the user so that the
answer can be verified against the source.

---

## 📝 Creative Feature — Document Summary

SmartDoc AI includes an automatic document summary feature.

The user can click:

`Generate Summary`

The application selects content from different parts of the
uploaded document and sends it to the local language model.

The model is instructed to generate a concise summary using only
the provided document content.

---

## 📂 Project Structure

```text
SmartDocAI/
│
├── app.py
├── rag_pipeline.py
├── document_processor.py
├── chunking.py
├── requirements.txt
├── README.md
├── .env
│
└── venv/