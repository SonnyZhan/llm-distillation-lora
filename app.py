import re
import time
from threading import Thread
from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
# Your adapter in Google Drive (Google Drive for desktop shows up as the G: drive)
ADAPTER = str(Path(__file__).parent / "lora_analysis")
INSTRUCTION = ("Classify the sentiment of this financial news sentence for an investor. "
               "Answer with positive, negative, or neutral.")
EXAMPLES = [
    "Net sales increased by 18% compared to last year.",
    "The company will lay off 200 employees as demand falls.",
    "The annual general meeting will be held on 15 March.",
]
COLORS = {"positive": "green", "negative": "red", "neutral": "gray"}


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    model.to(device).eval()
    return tokenizer, model


def parse_label(text):
    text = text.lower()
    match = re.search(r"label:\s*(positive|negative|neutral)", text)
    if match:
        return match.group(1)
    found = [(text.find(l), l) for l in ["positive", "negative", "neutral"] if l in text]
    return min(found)[1] if found else "neutral"


def stream_answer(tokenizer, model, sentence, max_new_tokens=200):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\nSentence: {sentence}"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    settings = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
    Thread(target=model.generate, kwargs=settings).start()
    for piece in streamer:
        yield piece.replace("\n", "\n\n")


st.set_page_config(page_title="Tiny Finance Analyst")
st.title("Tiny Finance Analyst")
st.caption("A 0.5B open model, fine-tuned with LoRA on labels and analysis written by Claude. "
           "It runs entirely on this machine.")

with st.spinner("Loading the model (the first run downloads about 1 GB)..."):
    tokenizer, model = load_model()

example = st.selectbox("Pick an example, or type your own below", [""] + EXAMPLES)
sentence = st.text_area("Financial news sentence", value=example, height=100)

if st.button("Analyze", type="primary") and sentence.strip():
    start = time.perf_counter()
    with st.container(border=True):
        full_text = st.write_stream(stream_answer(tokenizer, model, sentence))
    seconds = time.perf_counter() - start
    label = parse_label(full_text)
    st.markdown(f"### Sentiment: :{COLORS[label]}[{label.upper()}]")
    st.caption(f"Written in {seconds:.1f} seconds on {model.device.type.upper()}. "
               "Small models can be wrong. Not investment advice.")