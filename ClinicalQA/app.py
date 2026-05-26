import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from retriever.hybrid_retriever import HybridRetriever
from retriever.prompts import build_prompt, SYSTEM_QA
from scripts.embedding_utils import embed_texts   

load_dotenv()

client = OpenAI()

retriever = HybridRetriever(
    chroma_path="./chroma_db",
    medline_collection_name="clinical_knowledge",
    medline_pt="./embeddings/medlineplus_embeddings.pt",
    embed_fn=embed_texts
)


def format_context(results):
    blocks = []

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        topic = meta.get("topic", "Unknown topic")
        url = meta.get("url", "")
        citation = f"MedlinePlus: {topic} ({url})"

        blocks.append(f"""
[Source {i}]
Source type: {r["source_type"]}
Citation: {citation}

{r["text"]}
""".strip())

    return "\n\n".join(blocks)


def llm_generate(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_QA},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def answer_question(question: str) -> dict:
    results = retriever.retrieve(question, final_k=5)
    context = format_context(results)
    prompt = build_prompt(question=question, context=context)
    answer = llm_generate(prompt)

    return {
        "answer": answer,
        "sources": results
    }


def format_sources_for_ui(sources):
    if not sources:
        return "No sources retrieved."

    blocks = []

    for i, s in enumerate(sources, 1):
        meta = s["metadata"]
        snippet = s["text"][:700]

        blocks.append(f"""
### Source {i}

**Source type:** {s.get("source_type", "")}  
**Topic:** {meta.get("topic", "")}  
**Group:** {meta.get("group", "")}  
**URL:** {meta.get("url", "")}  

{snippet}
""")

    return "\n\n---\n\n".join(blocks)


def gradio_qa(question):
    if not question or not question.strip():
        return "Please enter a clinical question.", ""

    try:
        result = answer_question(question)
        return result["answer"], format_sources_for_ui(result["sources"])
    except Exception as e:
        return f"Error: {str(e)}", ""


CSS = """
.gradio-container { background-color: #d9f5fb }
html, body { background: transparent !important; }
#shell { max-width: 1100px; margin: 0 auto; padding: 32px 20px 56px; }
#app-title h1 { text-align: center; margin: 0; }
#subtitle { text-align: center; }
.card {
  background: #ffffffcc;
  backdrop-filter: blur(2px);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 6px 20px rgba(0,0,0,.06);
}
#btn-qa button {
  background-color: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}
#btn-qa button:hover {
  background-color: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
}
"""

theme = gr.themes.Ocean(primary_hue="slate")

with gr.Blocks(css=CSS, theme=theme, title="NoteRx Clinical QA") as demo:
    with gr.Column(elem_id="shell"):
        gr.Markdown("# NoteRx Clinical QA", elem_id="app-title")
        gr.Markdown(
            "Ask a clinical question. This public demo retrieves evidence from "
            "**MedlinePlus medical knowledge** and answers with citations.",
            elem_id="subtitle"
        )

        question_in = gr.Textbox(
            label="Clinical Question",
            placeholder="e.g., What is Lasix used for in heart failure?",
            lines=3
        )

        btn_qa = gr.Button("Answer Question", variant="primary", elem_id="btn-qa")

        with gr.Row():
            with gr.Column(scale=5):
                answer_out = gr.Markdown(label="Answer", elem_classes=["card"])
            with gr.Column(scale=5):
                sources_out = gr.Markdown(label="Retrieved Sources", elem_classes=["card"])

        btn_qa.click(
            gradio_qa,
            inputs=question_in,
            outputs=[answer_out, sources_out]
        )

demo.launch()