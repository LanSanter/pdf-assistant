import tkinter as tk
from tkinter import filedialog
import readpdf
import discord
import openAI
import os
import numpy as np
import tracemalloc
from dotenv import load_dotenv
import os

load_dotenv()

tracemalloc.start()

pdf_file = ''
conversation = [{'role': "assistant", "content": ""}]
pretalk = [{'role': "system", "content": ""}]


saved_embeddings_file =""

def nameChk(s):
    name2ID = ['助手']
    for name in name2ID:
        if name in s: return 1
    return 0

def open_file_dialog():
    global pdf_file, saved_embeddings_file
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    pdf_file = file_path
    embed_file = file_path.split("/")
    saved_embeddings_file = embed_file[5].replace(".pdf","")
    if file_path:
        # 执行后续操作，例如读取PDF文件并进行处理
        print("已選擇的PDF文件路徑:", file_path)

root = tk.Tk()

button = tk.Button(root, text="選擇PDF文件", command=open_file_dialog)
button.pack(pady=10)

root.mainloop()



intents = discord.Intents.default()
intents.message_content = True
intents.guild_messages = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user.name}')
    await readpdf.embedding(pdf_file, saved_embeddings_file)


@client.event
async def on_message(message):
    global pretalk, conversation
    if message.content =='test':
        await message.channel.send('你的BOT已上線')
    if message.author.bot: 
        return
    n = min(len(message.content), 10)
    if nameChk(message.content[:n]):       
        total_characters = sum(len(d['content']) for d in conversation)
        while total_characters > 2000 and len(conversation) > 1:
            conversation.pop(1)
        conversation.append({"role": "user", "content": message.content})
        relative_text = list(await readpdf.searchtext(message.content, pdf_file, saved_embeddings_file))
        material = ' '.join(relative_text)
        pretalk[0]['content'] = "你是一名問答助手，請參考下列訊息後針對使用者提出的疑問進行最精簡回答:" + material
        result = pretalk + conversation
        respone = await openAI.openai_answer(result)
        conversation.append({"role": "assistant", "content": respone})
        await message.channel.send(respone)
        

client.run(os.getenv("BOT_TOKEN"))
