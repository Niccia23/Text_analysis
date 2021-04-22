import streamlit as st

#NLP packages
import spacy
from textblob import TextBlob
from gensim.summarization import summarize

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#Summary Function
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)

    tokens = [token.text for token in docx]
    allData = [('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma )) for token in docx]
    return allData


def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text,entity.label_)for entity in docx.ents ]
    allData = ['"Tokens":{},\n"Entities":{}'.format(tokens,entities)]
    return allData

def main():

    st.title("TA App")
    st.subheader("A simple NLP App for text analysis")


    #Tokenizer
    if st.checkbox("Show Tokens"):
        st.subheader("Tokenize the text")
        message = st.text_area("Type your text here")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Named Entity
    if st.checkbox("Show Named Entities"):
        st.subheader("Exctract Entities From Your Text")
        message = st.text_area("Type your text here")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)


    # Sentiment Analysis
    if st.checkbox( "Show Sentiment Analysis"):
        st.subheader("Sentiment of Your Text")
        message = st.text_area("Type your text here")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Text Summarization
    if st.checkbox( "Show Text Summarization"):
        st.subheader("Summarize Your Text")
        message = st.text_area("Type your text here")
        summary_options = st.selectbox("Choice Your Summarizer",("gensim", "sumy"))
        if st.button("Summarize"):
            if summary_options == 'gensim':
                st.text("Using Gensim..")
                summary_result = summarize(message)
            elif summary_options == 'sumy':
                st.text("Using Sumy..")
                summary_result = sumy_summarizer(message)
            else:
                st.warning("Using Default Summarizer")
                st.text("Using Gensim")
                summary_result = summarize(message)

            st.success(summary_result)

    st.sidebar.subheader("About The App")
    st.sidebar.text("Text Analysis App")


if __name__ == '__main__':
    main()
