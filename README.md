# AI Agent (together AI)

This project is a simple demo chatbot built with FastAPI that answers questions based on the content of PDF files using the powerful Llama 4 Maverick model (via Together API).

## Features

- Chatbot interface in your browser.
- Responds to any question—either using the PDFs as context or general knowledge.
- Uses [Llama 4 Maverick](https://together.ai/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8), one of the best large language models available.

## Limitations

- **Timeouts:** The chatbot may time out frequently, especially for long or complex queries. This is a known limitation due to model/API latency or resource constraints.
- **Responsiveness:** While the chatbot attempts to answer all questions, some queries may take longer or occasionally fail to return an answer.
- **API Key:** You need your own Together API key for this demo.

## Setup Instructions

1. **Unzip the project folder.**

2. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn together PyPDF2
   ```

3. **Set up your environment:**
   - Create a `.env` file in the project directory.
   - Add your Together API key:
     ```
     TOGETHER_API_KEY=your_api_key_here
     ```

4. **Add your PDF files:**
   - Place any PDFs you want the chatbot to use in the `pdfs/` folder.

5. **Run the chatbot:**
   ```bash
   python chatbot.py
   ```

6. **Open your browser:**
   - Go to [http://localhost:8000](http://localhost:8000) and start chatting!

## Notes

- This is a demo showcasing the capabilities of the Llama 4 Maverick model for PDF-based Q&A.
- For best results, keep your PDF content concise and relevant.
- Expect occasional delays or timeouts—this is normal for large model demos.

---

**Enjoy experimenting with Llama 4 Maverick!**
