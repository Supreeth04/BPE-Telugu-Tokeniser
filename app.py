import gradio as gr
from telugu_tokeniser import encode, decode, train_bpe, read_text

# Load the pre-trained merges
text = read_text('telugu_data.txt')
merges, _ = train_bpe(text)

def tokenize_text(input_text):
    encoded = encode(input_text, merges)
    decoded = decode(encoded, merges)
    return {
        "Encoded tokens": str(encoded),
        "Token count": len(encoded),
        "Decoded text": decoded,
        "Successful roundtrip": input_text == decoded
    }

# Create the interface
iface = gr.Interface(
    fn=tokenize_text,
    inputs=gr.Textbox(lines=4, placeholder="Enter Telugu text here..."),
    outputs=gr.JSON(),
    title="Telugu BPE Tokenizer",
    description="A byte-pair encoding tokenizer for Telugu text"
)

iface.launch() 