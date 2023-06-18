# chatbot
### 毕业设计项目，聊天机器人+情绪检测，可以初步实现测试与聊天机器人聊天用户的情绪状况.技术框架 ：Seq2seq框架，LSTM，Attation机制，Tensorflow2.0+Keras，Html+Vue，Ajax  项目介绍 ：重点研究了文本预处理、模型构建和训练，以及网页设计与实现。通过改进的Seq2seq模型，实现了实时对话，通过带标签数据和分类模型训练抑郁模型，实现文本抑郁检测。用户通过网页访问聊天机器人，对话的过程中调用抑郁检测模型，实现初步判断用户是否患有抑郁。  

    
## 聊天机器人模型
## 关键点
* LSTM
* seq2seq
* attention 实验表明加入attention机制后训练速度快，收敛快，效果更好。
## 语料及训练环境
  青云语料库10万组对话，在pycharm训练（不需要改路径）,最好租个gpu训练（模型训练部分稍微改一下路径）。
## 运行
### 方式一：完整过程
- **数据预处理**<br>
  `get_data`<br>
- **模型训练**<br>
  `chatbot_train`(此为挂载到google colab版本，本地跑对路径等需略加修改)<br>
- **模型预测**<br>
  `chatbot_inference_Attention`<br>
### 方式二：加载现有模型
- 运行`chatbot_inference_Attention`<br>
- 加载`models/W--184-0.5949-.h5` 
## 界面(Tkinter/前后端)
- ![](https://github.com/jiayiwang5/Chinese-ChatBot/blob/master/image/image.png)
- pycharm启动server.py,通过终端给出的网址进入，然后在网址后面加上“/chat”，以此来访问该项目。

## 抑郁检测模型
 模型放在文件夹model里，数据来自推特，数据处理部分代码放在code文件夹，模型实现和调用看infer.py文件。
