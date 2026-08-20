NYSC FAQ Chatbot

A Python-based FAQ chatbot that helps prospective and serving NYSC corps members find answers to common questions about registration, mobilization, camp, relocation, eligibility, clearance, and other NYSC-related topics.
 Project Overview

Corps members and prospective corps members often ask similar questions about the National Youth Service Corps (NYSC).

This project provides a simple question-and-answer system that searches a structured FAQ dataset and retrieves the most relevant answer.

How It Works

The system:

1. Accepts a user's question.
2. Cleans and preprocesses the question.
3. Converts questions into TF-IDF representations.
4. Compares the user's question with stored FAQ questions using cosine similarity.
5. Applies keyword and synonym matching to improve retrieval.
6. Uses conversation context for relevant follow-up questions.
7. Returns the most relevant answer.
8. Provides a fallback response when confidence is too low.

Features

- FAQ question matching
- TF-IDF text representation
- Cosine similarity search
- Keyword and synonym matching
- Intent/category detection
- Follow-up question handling
- Conversation history
- Confidence scoring
- Fallback responses
- Response category information
- Retrieval evaluation
- Automated tests

Dataset

The project currently contains:

- 304 FAQs
- 11 CSV files
- 10 FAQ categories

Categories include:

- Registration
- Relocation
- Camp
- Eligibility
- General
- Office
- Exclusion
- Exemption
- Foreign Graduates
- Mobilization

Technologies

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- TF-IDF
- Cosine similarity
- HTML/CSS/JavaScript
- Pytest

Evaluation

The retrieval system was evaluated against the FAQ dataset.

- FAQs evaluated: 304
- Correct retrievals: 283
- Retrieval consistency: 93.09%
- Automated tests: 28 passed

Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt