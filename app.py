from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)
path = 'C:\\Users\\lenovo\\Desktop\\chatbot\\gpt-small'
model = GPT2LMHeadModel.from_pretrained(path, local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained(path, local_files_only=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    sequence = f"Orang 1: {request.form['question']}  Orang 2:"
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=100,
        pad_token_id=model.config.eos_token_id,
        top_p=0.5,
        temperature=0.5
    )

    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    generated_text = generated_text.split("Orang 2:")[1].split("Orang 1:")[0]
    return render_template("index.html", answer=str(generated_text.strip()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")