from flask import Flask, render_template, request, jsonify


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    # Lowercase for simple matching
    user_input = text.lower().strip()

    # Custom responses for key phrases
    if "hello" in user_input:
        return "Hello! How can I support you today?"
    elif "i am so depressed" in user_input:
        return "I'm really sorry you're feeling this way. You're not alone—I'm here for you."
    elif "what are the techniques to get relief from sress" in user_input or "how to overcome from it" in user_input or "help with depression please!" in user_input:
        return "Some helpful techniques include regular exercise, meditation, talking to a friend, and seeking therapy. Would you like resources for any of these?"
    elif "therapy?" in user_input:
        return "I can help you find resources for therapy or support groups. Would you like me to do that?"
    elif "yes" in user_input or "sure" in user_input:
        return "Great! Here are some resources: [Types of Therapies for Stress Relief: Cognitive Behavioral Therapy (CBT), mindfulness practices, relaxation techniques like deep breathing and progressive muscle relaxation, and engaging in physical activities like yoga or exercise]."
    elif "thank you" in user_input or "thanks" in user_input:
        return "You're welcome! I'm here to help. If you have any more questions, feel free to ask."
    # Otherwise, use the model for general input (stateless)
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)




if __name__ == '__main__':
    app.run(debug=True)
