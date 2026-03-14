# MediBot: Hospital Desk Assistant

*This project is a Command-Line Interface (CLI) application that acts as an AI hospital receptionist. It allows patients to ask questions about doctor schedules, specialties, and operating hours.*

## Boot.dev Personal Project

This project was built as the first personal project for the [Boot.dev](https://boot.dev) curriculum. The primary goal of this assignment is not to build a production-ready application, but rather to practice the end-to-end process of building software from scratch. 

Key learning objectives for this project included:
* Breaking down a larger idea into manageable phases.
* Practicing version control by committing code often and pushing to GitHub.
* Writing custom data-handling logic.
* Writing clear documentation for end-users.
* Python type hints to make my code cleaner and reduce errors.
* How to handle a big project with many moving parts and constant room for improvement.

# 🚀 Features & Commands

MediBot is organized into several specialized CLI modules. Ensure you have your `.env` file configured with your `GEMINI_API_KEY` before running.

### 1. Augmented Generation & Chat
The primary interface for patient interaction. It supports one-off questions and interactive sessions with memory.

* **Interactive Conversation:** (Starts a persistent session with chat history)
    ```bash
    python augmented_generation_cli.py conversation
    ```
* **Direct Question:**
    ```bash
    python augmented_generation_cli.py question "Who is the cardiologist?"
    ```
* **RAG with Citations:**
    ```bash
    python augmented_generation_cli.py citations "When is Dr. Smith available?"
    ```

### 2. Advanced Search & Evaluation Tools
Beyond the main chat interface, MediBot includes specialized modules for refining and measuring search accuracy:

* **Hybrid Search (`hybrid_search_cli.py`):** Combines keyword (BM25) and semantic search using Reciprocal Rank Fusion (RRF) for better doctor matching.
* **Multimodal Search (`multimodal_search_cli.py` & `describe_image_cli.py`):** Allows searching for specialists using images and leverages LLMs to rewrite visual observations into clinical queries.
* **Core Utilities:** Tools to build inverted indices (`keyword_search_cli.py`) and verify high-dimensional vector embeddings (`semantic_search_cli.py`).
* **Evaluation (`evaluation_cli.py`):** A dedicated testing suite that calculates Precision, Recall, and F1 scores to quantify retrieval performance.

---

### 🛠️ Possible next improvements I would like to apply:

* **Metadata Filtering:** filter by specific days, times, or hospital wings.
* **Persistent Memory:** Save chat history to a database so the bot remembers users.
* **Evaluation Dataset:** Expand my database and test cases to cover more complex medical questions.
* **Web Interface:** Move the bot from the terminal to a clean web dashboard.
* **Speech-to-Text:** Allow patients to ask questions using their voice instead of typing.
