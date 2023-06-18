from tokenize import String

from flask import Flask, request, render_template
from chatbot import Chatbot
from infer import emotion_detection_function
import re
from flask import Flask
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return "我是智能机器人小莫，欢迎您的到来。ps：聊天请转到/chat路由!"

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # 创建 Chatbot 实例
    chatbot = Chatbot()
    if request.method == 'POST':
        user_input = request.json['message']
        chatbot_response = chatbot.chat_response(user_input)

        last_sentence = get_last_sentence(user_input)

        emotion = emotion_detection_function(last_sentence)
        # 进一步处理聊天机器人的响应和情绪判断结果
        response_data = {
            'response': chatbot_response,
            'emotion': emotion
        }
        s = response_data.get("response")
        # s.replace("EOS","");
        e=response_data.get("emotion")

        print(s+"\n情绪检测结果："+e)
        # 返回带有聊天机器人回复和情绪结果的字典
        return response_data

    else:
        # 处理GET请求的逻辑
        return render_template('newchat.html')  # 返回chat.html页面

def get_last_sentence(user_input):
    sentences = re.split(r'[.!?]', user_input)
    last_sentence = sentences[-1].strip()
    return last_sentence

if __name__ == '__main__':
    app.run(port=8000)
