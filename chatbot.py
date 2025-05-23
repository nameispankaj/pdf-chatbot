import os
import glob
import PyPDF2
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import markdown as md
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from together import Together
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

# === CONFIG ===
PDF_FOLDER = "pdfs"
TOGETHER_API_KEY = "tgp_v1_DJjgagYOdwXci5KGKCJe2I0jGtmvvMGlYt-OBk1JSVc"
EMBED_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
MAX_TOKENS = 2048
QA_SIM_THRESHOLD = 0.75

embedder = SentenceTransformer(EMBED_MODEL)
client = Together(api_key=TOGETHER_API_KEY)
executor = ThreadPoolExecutor(max_workers=4)  # Increase for more concurrency

# --- Extract QA pairs with source reference ---
def extract_qa_pairs_from_pdfs(pdf_folder):
    qa_pairs = []
    for pdf_file in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        pdf_filename = os.path.basename(pdf_file)
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    i = 0
                    while i < len(lines):
                        if lines[i].strip().lower().startswith("q"):
                            q = lines[i].split(":", 1)[-1].strip() if ':' in lines[i] else lines[i].strip()
                            j = i + 1
                            while j < len(lines) and not lines[j].strip().lower().startswith("a"):
                                j += 1
                            if j < len(lines):
                                a = lines[j].split(":", 1)[-1].strip() if ':' in lines[j] else lines[j].strip()
                                k = j + 1
                                answer_lines = [a]
                                while k < len(lines) and not lines[k].strip().lower().startswith("q"):
                                    answer_lines.append(lines[k].strip())
                                    k += 1
                                full_answer = "\n".join([l for l in answer_lines if l])
                                qa_pairs.append({
                                    "q": q,
                                    "a": full_answer,
                                    "pdf": pdf_filename,
                                    "page": page_num + 1
                                })
                                i = k
                            else:
                                i += 1
                        else:
                            i += 1
    return qa_pairs

def extract_text_from_pdfs(pdf_folder):
    all_text = []
    for pdf_file in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
    return "\n".join(all_text)

qa_pairs = extract_qa_pairs_from_pdfs(PDF_FOLDER)
pdf_questions = [qa["q"] for qa in qa_pairs]
pdf_answers = [qa["a"] for qa in qa_pairs]
pdf_sources = [(qa["pdf"], qa["page"]) for qa in qa_pairs]
pdf_q_embeddings = embedder.encode(pdf_questions) if pdf_questions else np.array([])
pdf_text = extract_text_from_pdfs(PDF_FOLDER)

def make_system_prompt(pdf_text):
    return (
        "You are a helpful assistant. If a user asks a question about the PDFs, use the following context:\n"
        f"{pdf_text[:100000]}\n"
        "Otherwise, answer general questions as a knowledgeable chatbot."
    )

def get_similar_pdf_answer(user_question, threshold=QA_SIM_THRESHOLD):
    if not pdf_questions:
        return None, 0, None
    user_emb = embedder.encode([user_question])
    sims = cosine_similarity(user_emb, pdf_q_embeddings)[0]
    max_idx = int(np.argmax(sims))
    max_score = sims[max_idx]
    if max_score > threshold:
        return pdf_answers[max_idx], max_score, max_idx
    return None, float(max_score), None

def paragraphify(text):
    """
    Converts text with excessive newlines into clean paragraphs.
    Paragraphs are separated by two or more newlines, everything else is joined into one line.
    """
    paras = re.split(r'\n{2,}', text)
    cleaned = []
    for para in paras:
        single_line = ' '.join(line.strip() for line in para.strip().split('\n') if line.strip())
        if single_line:
            cleaned.append(single_line)
    return '<p>' + '</p><p>'.join(cleaned) + '</p>'

app = FastAPI()

HTML_CHAT = """
<!DOCTYPE html>
<html>
<head>
  <title>PDF Chatbot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link href="https://fonts.googleapis.com/css?family=Inter:400,700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Inter', Arial, sans-serif;
      background: linear-gradient(120deg, #e7f0fd 0%%, #f5f7fa 100%%);
      color: #212c33;
      margin: 0;
      min-height: 100vh;
    }
    #chatbox {
      width: 100%%;
      max-width: 740px;
      margin: 42px auto 0 auto;
      background: #fff;
      padding: 28px 28px 15px 28px;
      border-radius: 18px;
      box-shadow: 0 8px 40px #b0b6c844;
    }
    #messages {
      min-height: 320px; max-height: 59vh;
      overflow-y: auto; margin-bottom: 25px;
      display: flex; flex-direction: column;
      gap: 8px;
    }
    .bubble {
      margin: 0;
      padding: 13px 22px;
      border-radius: 18px;
      max-width: 87%%;
      word-break: break-word;
      font-size: 1.12em;
      line-height: 1.68;
      box-shadow: 0 3px 14px #e3e8f367;
      transition: background .22s;
    }
    .user {
      background: #e5e9f8;
      align-self: flex-end; text-align: right;
      color: #315e8a; font-weight: 600;
      border-bottom-right-radius: 8px;
    }
    .bot {
      background: #f8fafc;
      border-left: 4px solid #7fc97f;
      align-self: flex-start;
      color: #184c20;
      border-bottom-left-radius: 8px;
    }
    .bot.loading {
      font-style: italic; color: #888; background: #f7f8fa;
      border-left: 4px solid #bbb;
    }
    .justify { text-align: justify; }
    #askform {
      display: flex; gap: 11px; margin-bottom: 0;
    }
    #question {
      flex: 1; padding: 15px; border-radius: 7px;
      border: 1.5px solid #bdd3e6;
      font-size: 1em; background: #f8fafc;
      transition: border-color .22s;
    }
    #question:focus { outline: none; border-color: #7fc97f; }
    button {
      padding: 14px 32px;
      border-radius: 7px; border: none;
      background: linear-gradient(90deg, #7fc97f, #4e7);
      color: #fff; font-weight: 700; font-size: 1em;
      cursor: pointer; box-shadow: 0 2px 7px #c7dfcd4d;
      transition: background .18s;
    }
    button:active { background: #4e7; }
    pre, code {
      background: #23272e; color: #eee;
      padding: 2px 7px; border-radius: 4px;
      font-size: 0.98em;
    }
    ul, ol { padding-left: 27px; }
    .spinner {
      display: inline-block; width: 18px; height: 18px;
      border: 3px solid #e6e6e6;
      border-top: 3px solid #7fc97f;
      border-radius: 50%%; animation: spin 1s linear infinite;
      margin-right: 8px; vertical-align: middle;
    }
    @keyframes spin {
      0%% { transform: rotate(0deg); }
      100%% { transform: rotate(360deg); }
    }
    @media (max-width: 600px) {
      #chatbox { padding: 8px; }
      #question { padding: 11px; }
      button { padding: 11px 14px; font-size: 0.97em;}
    }
    /* Enhance PDF answer appearance */
    .pdf-ref {
      margin-top:2em;
      font-size:0.97em;
      color:#888;
      text-align: left;
    }
    .justify p {
      margin-block: 0.65em 0.65em;
    }
  </style>
</head>
<body>
  <div id="chatbox">
    <h2 style="text-align:center; margin-bottom:20px; color:#3e6f44;">PDF Chatbot</h2>
    <div id="messages"></div>
    <form id="askform" autocomplete="off">
      <input type="text" id="question" placeholder="Type your message..." required autocomplete="off" />
      <button type="submit">Send</button>
    </form>
  </div>
  <script>
    const form = document.getElementById('askform');
    const questionInput = document.getElementById('question');
    const messagesDiv = document.getElementById('messages');
    let history = [];

    function renderMessages() {
      messagesDiv.innerHTML = '';
      for (const msg of history) {
        const div = document.createElement('div');
        div.className = 'bubble ' + (msg.role === 'user' ? 'user' : (msg.loading ? 'bot loading' : 'bot'));
        if (msg.role === 'bot') {
          div.innerHTML = msg.html || '<span class="spinner"></span> <i>Thinking...</i>';
        } else {
          div.textContent = msg.content;
        }
        messagesDiv.appendChild(div);
      }
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      const question = questionInput.value.trim();
      if (!question) return;
      history.push({ role: 'user', content: question });
      renderMessages();
      questionInput.value = '';
      history.push({ role: 'bot', loading: true, html: null });
      renderMessages();
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      history.pop(); // remove loading
      history.push({ role: 'bot', html: data.html_answer || "<b>Answer:</b><br>" + (data.answer || data.detail || "No response.") });
      renderMessages();
    };
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CHAT

def llm_query(system_prompt, user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            max_tokens=MAX_TOKENS
        )
        answer_pieces = []
        for token in response:
            if hasattr(token, 'choices') and token.choices:
                delta = getattr(token.choices[0], 'delta', None)
                if delta and hasattr(delta, 'content') and delta.content:
                    answer_pieces.append(delta.content)
        answer = ''.join(answer_pieces).strip()
        if not answer:
            answer = "No response."
        # Hide specific rate limit error
        if "rate limit" in answer.lower():
            return "Sorry, I'm currently handling a lot of requests. Please try again in a moment."
        return answer
    except Exception as e:
        if "rate limit" in str(e).lower():
            return "Sorry, I'm currently handling a lot of requests. Please try again in a moment."
        return "Sorry, there was an issue processing your request. Please try again later."

@app.post("/chat")
async def chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    user_input = data.get("question", "")
    pdf_answer, sim, pdf_idx = get_similar_pdf_answer(user_input)
    if pdf_answer:
        pdf_file, pdf_page = pdf_sources[pdf_idx]
        ref_html = f'<div class="pdf-ref">Reference: <b>{pdf_file}</b>, page {pdf_page}</div>'
        html_answer = f'<div class="justify">{paragraphify(pdf_answer)}<br><br>{ref_html}</div>'
        answer = pdf_answer + "\n\n\nReference: {} (page {})".format(pdf_file, pdf_page)
    else:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            executor,
            llm_query,
            make_system_prompt(pdf_text),
            user_input
        )
        html_answer = md.markdown(answer, extensions=['fenced_code', 'tables', 'nl2br'])
    return JSONResponse({"answer": answer, "html_answer": html_answer})

if __name__ == "__main__":
    # For production, use: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(app, host="0.0.0.0", port=8000)
