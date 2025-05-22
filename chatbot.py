import os
import glob
import PyPDF2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import markdown as md
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from together import Together

PDF_FOLDER = "pdfs"
TOGETHER_API_KEY = "tgp_v1_DJjgagYOdwXci5KGKCJe2I0jGtmvvMGlYt-OBk1JSVc"

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = Together(api_key=TOGETHER_API_KEY)

def extract_qa_pairs_from_pdfs(pdf_folder):
    qa_pairs = []
    for pdf_file in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    i = 0
                    while i < len(lines):
                        if lines[i].strip().lower().startswith("q"):
                            q = lines[i].split(":", 1)[-1].strip() if ':' in lines[i] else lines[i].strip()
                            # Find the 'A:' line
                            j = i + 1
                            while j < len(lines) and not lines[j].strip().lower().startswith("a"):
                                j += 1
                            if j < len(lines):
                                # Extract answer (multi-line)
                                a = lines[j].split(":", 1)[-1].strip() if ':' in lines[j] else lines[j].strip()
                                k = j + 1
                                answer_lines = [a]
                                while k < len(lines) and not lines[k].strip().lower().startswith("q"):
                                    answer_lines.append(lines[k].strip())
                                    k += 1
                                full_answer = "\n".join([l for l in answer_lines if l])
                                qa_pairs.append((q, full_answer))
                                i = k
                            else:
                                i += 1
                        else:
                            i += 1
    return qa_pairs

qa_pairs = extract_qa_pairs_from_pdfs(PDF_FOLDER)
pdf_questions = [q for q, a in qa_pairs]
pdf_answers = [a for q, a in qa_pairs]
if pdf_questions:
    pdf_q_embeddings = embedder.encode(pdf_questions)
else:
    pdf_q_embeddings = np.array([])

def get_similar_pdf_answer(user_question, threshold=0.75):
    if not pdf_questions:
        return None, 0
    user_emb = embedder.encode([user_question])
    sims = cosine_similarity(user_emb, pdf_q_embeddings)[0]
    max_idx = int(np.argmax(sims))
    max_score = sims[max_idx]
    if max_score > threshold:
        return pdf_answers[max_idx], max_score
    return None, float(max_score)

def extract_text_from_pdfs(pdf_folder):
    all_text = ""
    for pdf_file in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
    return all_text

def make_system_prompt(pdf_text):
    return (
        "You are a helpful assistant. If a user asks a question about the PDFs, use the following context:\n"
        f"{pdf_text[:100000]}\n"
        "Otherwise, answer general questions as a knowledgeable chatbot."
    )

pdf_text = extract_text_from_pdfs(PDF_FOLDER)

app = FastAPI()

HTML_CHAT = """
<!DOCTYPE html>
<html>
<head>
  <title>PDF Chatbot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f4f4; }
    #chatbox { width: 100%%; max-width: 700px; margin: 40px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow:0 0 10px #bbb; }
    #messages { min-height: 300px; max-height: 55vh; overflow-y: auto; margin-bottom: 16px; }
    .bubble { margin: 12px 0; padding: 12px 18px; border-radius: 18px; max-width: 80%%; word-break: break-word; }
    .user { background: #e0e6f7; align-self: flex-end; text-align: right; }
    .bot { background: #eef; border-left: 4px solid #4e7; align-self: flex-start; }
    #askform { display: flex; gap: 10px; }
    #question { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #bbb; }
    button { padding: 10px 24px; border-radius: 6px; border: none; background: #4e7; color: #fff; font-weight: bold; }
    pre, code { background: #222; color: #eee; padding: 2px 4px; border-radius: 3px; }
    ul, ol { padding-left: 20px; }
    @media (max-width: 600px) {
      #chatbox { padding: 8px; }
      #question { padding: 8px; }
      button { padding: 8px 16px; }
    }
  </style>
</head>
<body>
  <div id="chatbox">
    <h2 style="text-align:center;">PDF Chatbot</h2>
    <div id="messages" style="display: flex; flex-direction: column;"></div>
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
        div.className = 'bubble ' + (msg.role === 'user' ? 'user' : 'bot');
        if (msg.role === 'bot') {
          div.innerHTML = msg.html || '<i>...</i>';
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
      history.push({ role: 'bot', html: '<i>Thinking...</i>' });
      renderMessages();
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      history.pop(); // remove 'Thinking...'
      history.push({ role: 'bot', html: data.html_answer || "<b>Answer:</b><br>" + (data.answer || data.detail || "No response.") });
      renderMessages();
    };
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_CHAT

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("question")
    pdf_answer, sim = get_similar_pdf_answer(user_input)
    if pdf_answer:
        answer = pdf_answer
    else:
        # Use LLM for generic/basic questions not covered in PDF
        messages = [
            {"role": "system", "content": make_system_prompt(pdf_text)},
            {"role": "user", "content": user_input}
        ]
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=messages,
                stream=True,
                max_tokens=2048
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
        except Exception as e:
            answer = f"Error: {str(e)}"
    html_answer = md.markdown(answer, extensions=['fenced_code', 'tables', 'nl2br'])
    return JSONResponse({"answer": answer, "html_answer": html_answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=8000)