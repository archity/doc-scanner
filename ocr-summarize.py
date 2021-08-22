import pytesseract
import spacy
import pytextrank
import networkx as nx
import altair as alt
from icecream import ic


def image_to_text():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    text = pytesseract.image_to_string("./img/scanned_doc.jpg")
    text = text.replace("\n", " ")
    with open("ocr.txt", "w") as text_file:
        text_file.write(text)
    return text


def text_summarizer(text):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    doc = nlp(text)

    # To store all the important keywords extracted from the doc
    keywords = []

    # Examine the top-ranked phrases in the document
    for p in doc._.phrases:
        print('{:.4f} {:5d}  {}'.format(p.rank, p.count, p.text))
        if p.rank != 0:
            keywords.append(p.text)

    # Store the keyphrases and rank in an interactive html-based chart
    tr = doc._.textrank
    altair_chart = tr.plot_keyphrases()
    altair_chart.save('./img/altair_chart.html', embed_options={'renderer': 'svg'})

    summary = []

    for sent in tr.summary(preserve_order=True):
        summary.append(sent)
        ic(sent)

    with open("summary.txt", "w") as text_file:
        text_file.write(f"Summary: \n {summary}\n\n")
        text_file.write(f"Keywords: \n {keywords}")


if __name__ == "__main__":
    ocr_text = image_to_text()
    text_summarizer(text=ocr_text)
