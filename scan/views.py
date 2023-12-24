from django.shortcuts import render
from django.http import HttpResponse
import spacy
from PyPDF2 import PdfReader
from transformers import pipeline




# Create your views here.
def index(request):
    if request.method == 'POST':
        
        if 'pdf_file' in request.FILES:
            pdf_file = request.FILES['pdf_file']
            pdf_text = extract_text_from_pdf(pdf_file)
            solution = summarize_text(pdf_text)
            return render(request, 'summary.html', {'summary': solution})

    
    return render(request, 'index.html')


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def summarize_text(text):
    summarizer = pipeline('summarization')
    summary = summarizer(text, max_length=500, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0][
        'summary_text']

    return summary

