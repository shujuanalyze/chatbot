from chatbot import Chatbot
from infer import emotion_detection_function
import re
def chat(user_input):
    # 创建 Chatbot 实例
    chatbot = Chatbot()
    chatbot_response = chatbot.chat_response(user_input)
    last_sentence = get_last_sentence(user_input)
    emotion = emotion_detection_function(last_sentence)
    return {'response': chatbot_response, 'emotion': emotion}

def get_last_sentence(user_input):
    sentences = re.split(r'[.!?]', user_input)
    last_sentence = sentences[-1].strip() if sentences else ''
    return last_sentence

while True:
    seq = input()
    if seq == 'x':
        break

    ans = chat(seq)
    print(ans)





