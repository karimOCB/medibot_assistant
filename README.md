# MediBot: Hospital Desk Assistant

*This project is a Command-Line Interface (CLI) application that acts as an AI hospital receptionist. It allows patients to ask questions about doctor schedules, specialties, and operating hours.*

*Instead of relying on the LLM's pre-trained knowledge, this tool uses a basic Retrieval-Augmented Generation (RAG) pipeline. It loads a local database of hospital staff from a JSON file, formats the schedule, and passes it to the LLM as system context. This ensures the AI provides accurate, grounded answers based strictly on the hospital's actual data rather than hallucinating.*

## Boot.dev Personal Project

This project was built as the first personal project for the [Boot.dev](https://boot.dev) curriculum. The primary goal of this assignment is not to build a production-ready application, but rather to practice the end-to-end process of building software from scratch. 

Key learning objectives for this project included:
* Breaking down a larger idea into manageable phases.
* Practicing version control by committing code often and pushing to GitHub.
* Writing custom data-handling logic.
* Writing clear documentation for end-users.