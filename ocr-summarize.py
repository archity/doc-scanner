import pytesseract
from gensim.summarization import summarize


def image_to_text():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    text = pytesseract.image_to_string("./img/scanned_doc.jpg")
    return text


def text_summarizer(text):
    print(summarize(text))


if __name__ == "__main__":
    ocr_text = image_to_text()
    text_summarizer(text=ocr_text)
