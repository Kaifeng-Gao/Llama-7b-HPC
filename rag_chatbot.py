from chatbot import ChatBot
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class RagChatbot(ChatBot):
    def __init__(self, model_path, new_model_path = None, document_list = []):
        super().__init__(model_path, new_model_path)
        self.retriever = self.load_docs(document_list)
        self.rag_chain = self.init_rag_chain()
    
    def load_docs(self, document_list):
        loader = WebBaseLoader(document_list)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        vector = FAISS.from_documents(documents, embeddings)
        retriever = vector.as_retriever(search_kwargs={"k": 1})
        return retriever

    def init_rag_chain(self):
        prompt_template = \
        """
        [INST] <<SYS>> Use context provided to answer the question below, explicitly state what part of your response is retrieved from context and what part is based on general knowledge <</SYS>>
        CONTEXT:
        {context}
        QUESTION:
        {question} [/INST]
        """

        # Create prompt from prompt template 
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        text_generation_pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )

        llama_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        llm_chain = LLMChain(llm=llama_llm, prompt=prompt)
        rag_chain = ( 
        {"context": self.retriever, "question": RunnablePassthrough()}
            | llm_chain
        )
        return rag_chain
        
    def generate_response(self, conversation_history):
        history_prompt, length = super().generate_prompt_from_history(conversation_history)
        print(history_prompt)
        output = self.rag_chain.invoke(history_prompt)
        print(output)
        response = output['text']
        return response
