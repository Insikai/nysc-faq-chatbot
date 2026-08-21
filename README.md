 NYSC FAQ Chatbot

A Python-based FAQ chatbot that helps prospective and serving NYSC corps members find answers to common questions about registration, mobilization, camp, relocation, eligibility, clearance, and other NYSC-related topics.

 Author
Samuel Gabriel

 Project Overview

Corps members and prospective corps members frequently ask similar questions about the National Youth Service Corps (NYSC).

The NYSC FAQ Chatbot provides a simple question-and-answer system that searches a structured FAQ dataset and retrieves the most relevant answer to a user's question.

 How It Works

The system:

1. Accepts a user's question.
2. Cleans and preprocesses the question.
3. Converts FAQ questions into TF-IDF representations.
4. Compares the user's question with stored FAQ questions using cosine similarity.
5. Applies keyword and synonym matching to improve retrieval.
6. Detects question categories and intents.
7. Uses conversation context to handle relevant follow-up questions.
8. Calculates a confidence score for the retrieved result.
9. Returns the most relevant answer when confidence is sufficient.
10. Provides a fallback response when confidence is too low.

 Features

- FAQ question matching
- TF-IDF text representation
- Cosine similarity search
- Keyword and synonym matching
- Intent and category detection
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

 Categories

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
- HTML
- CSS
- JavaScript
- Pytest

 Evaluation

The retrieval system was evaluated against the FAQ dataset.

- FAQs evaluated: 304
- Correct retrievals: 283
- Retrieval consistency: 93.09%
- Automated tests: 28 passed

 Example

The chatbot can answer a direct question such as:

User:  
Can I relocate to another state?

Chatbot:  
Relocation to another state may be possible when you meet the applicable NYSC requirements and receive approval.

It can also handle a contextual follow-up:

User:  
What about for marriage?

Chatbot:  
A marriage certificate may be required when applying for relocation on the basis of marriage, together with other documents specified by NYSC.

 Running the Project

 1. Clone the repository

```bash
git clone https://github.com/Insikai/nysc-faq-chatbot.git
cd nysc-faq-chatbot
