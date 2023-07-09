from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers
from langdetect import detect
from deep_translator import GoogleTranslator


class RAG:
    def __init__(self, input_df, ret_query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                 ret_passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                 gen_model_name_or_path="facebook/rag-token-nq"):
        self.document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)
        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model=ret_query_embedding_model,
            passage_embedding_model=ret_passage_embedding_model,
            use_gpu=True,
            embed_title=True,
        )
        self.generator = RAGenerator(
            model_name_or_path=gen_model_name_or_path,
            use_gpu=True,
            top_k=3,
            max_length=500,
            min_length=2,
            embed_title=True,
            num_beams=5,
        )
        self.init_document_store(input_df)
        self.qa_pipe = self.build_qa_pipe(self.generator, self.retriever)

    def init_document_store(self, input_data):
        # Use data to initialize Document objects
        titles = list(input_data["title"].values)
        texts = list(input_data
                     ["text"].values)
        documents = []
        for title, text in zip(titles, texts):
            documents.append(Document(content=text, meta={"name": title or ""}))

        # Delete existing documents in documents store
        self.document_store.delete_documents()

        # Write documents to document store
        self.document_store.write_documents(documents)

        # Add documents embeddings to index
        self.document_store.update_embeddings(retriever=self.retriever)

    def build_qa_pipe(self, generator, retriever):
        return GenerativeQAPipeline(generator=generator, retriever=retriever)

    def ask_question(self, question: str, print_answer=True,
                     params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}}):
        answer = self.qa_pipe.run(query=question, params=params)
        if print_answer:
            print_answers(answer, details="minimum")
        return answer

    def multilingual_questioning(self, question):
        lang = detect(question)
        if lang != 'en':
            translated_question = GoogleTranslator(source='auto', target='en').translate(text=question)
        else:
            translated_question = question
        answer = self.ask_question(translated_question, False).get('answers')[0].answer
        if lang != 'en':
            translated_answer = GoogleTranslator(source='auto', target=lang).translate(text=answer)
        return {"question": question, "translated_question": translated_question,
                "answer": answer,
                "translated_answer": answer if lang == 'en' else translated_answer}
