import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import openAI
import PyPDF2
import time

pdf_text = ''
segments = []

def convert_pdf_to_text(path):
    text = ""
    with open(path, 'rb') as fp:
        reader = PyPDF2.PdfFileReader(fp)
        num_pages = reader.numPages
        for page_num in range(num_pages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text


def split_text_into_segments(text, segment_length):
    segments = []
    sentences = text.split("。")
    current_segment = ''
    for sentence in sentences:
        current_segment += sentence
        if len(current_segment) >= segment_length:
            segments.append(current_segment)
            current_segment = ''
    if current_segment:
        segments.append(current_segment)
    return segments


async def embed_text(segments):
    for segment in segments:
        new_embedding = await openAI.embedding_text(segment)
        yield new_embedding
async def embedd_text(segment):
    embedding = await openAI.embedding_text(segment)
    return embedding
    
def compare_similarity(new_embedding, saved_embeddings):
    new_embedding = np.array(new_embedding).reshape(1, -1)
    saved_embeddings = np.array(saved_embeddings)
    similarities = cosine_similarity(new_embedding, saved_embeddings)
    return similarities


def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)


def load_embeddings(filename):
    return np.load(filename + '.npy')


def find_most_similar_texts(new_embedding, saved_embeddings, segments):
    similarities = compare_similarity(new_embedding, saved_embeddings)
    sorted_indices = np.argsort(similarities)[0][::-1]
    top_indices = sorted_indices[:3]  # 提取相似度最高的前两个索引
    most_similar_texts = [segments[i] for i in top_indices]
    return most_similar_texts


async def embedding(pdf_file, saved_embeddings_file):
    global pdf_text, segments

    if not os.path.isfile(saved_embeddings_file + '.npy'):
        saved_embeddings = np.empty((0, 0))
        pdf_text = convert_pdf_to_text(pdf_file)
        segments = split_text_into_segments(pdf_text, 400)
        embeddings = []
        async for batch_embeddings in embed_text(segments):
            embeddings.append(batch_embeddings)
        print("embedding finish")
        saved_embeddings = np.array(embeddings)
        save_embeddings(saved_embeddings, saved_embeddings_file)

    else:
        pdf_text = convert_pdf_to_text(pdf_file)
        segments = split_text_into_segments(pdf_text, 400)
        print("load done")

    
async def searchtext(text, pdf_file, saved_embeddings_file):
    global segments
    texts = segments
    save_embeddings = load_embeddings(saved_embeddings_file)
    new_embedding = await embedd_text(text)

    return find_most_similar_texts(new_embedding, save_embeddings, texts)




