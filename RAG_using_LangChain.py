!pip install openai langchain langchain_community docx2txt tiktoken chromadb unstructured yt_dlp --upgrade --quiet

import sys
import os
import openai
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())               # find and load the local env file

os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.environ['OPENAI_API_KEY']


######################## DOCUMENT PARSING ############################

# WORKING WITH PDF

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader('/kaggle/input/llmtext/LLMs.docx')
pages = loader.load()
print(len(pages))

page = pages[0]
dash = "-"*100
print(f"First 500 chars \n{dash} \n",page.page_content[0: 500], f"\n{dash}")

print("\nMetadata \n\n", page.metadata)



# AUDIO LOADERS

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url = 'https://www.youtube.com/watch?v=zBjJUV-lzHo'
save_dir = '/kaggle/working/youtube/'

loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser())
docs = loader.load()



# WEB LOADERS

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader('https://towardsdatascience.com/recommender-systems-a-complete-guide-to-machine-learning-models-96d3f94ea748')
docs = loader.load()
print(len(docs))

print(docs[0].page_content[:500])








######################## DOCUMENT SPLITTING ############################

'''
List of splitters -
- CharacterTextSplitter
- MarkdownHeaderTextSplitter
- TokenTextSplitter
- SentenceTransformersTokenTextSplitter
- RecursiveCharacterTextSplitter
- NLTKTextSplitter
- SpacyTextSplitter
- Language - (for programming languages)
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter

chunk_size = 26
chunk_overlap = 4

r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
c_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=' ')

text1 = 'abcdefghijklmnopqrstuvvdkfbfdsKbfdhBF'
print(r_splitter.split_text(text1))

text2 = 'abcdefghijklmnopqrs tuvwxyzvdkfbf dsKbfdhBF'
print(c_splitter.split_text(text2))

text3 = 'a b c d e f g h i j k l m n o p q r s t u v w x y z v d k f b f d s K b f d h B F'
print(c_splitter.split_text(text3))


paragraph = 'Recommender systems are algorithms providing personalized suggestions for items that are most relevant to each user. With the massive growth of available online contents, users have been inundated with choices. It is therefore crucial for web platforms to offer recommendations of items to each user, in order to increase user satisfaction and engagement. \
The following list shows examples of well-known web platforms with a huge number of available contents, which need efficient recommender systems to keep users interested. All these platforms use powerful machine learning models in order to generate relevant recommendations for each user. \
\
In recommender systems, machine learning models are used to predict the rating rᵤᵢ of a user u on an item i. At inference time, we recommend to each user u the items l having highest predicted rating rᵤᵢ. \
We therefore need to collect user feedback, so that we can have a ground truth for training and evaluating our models. An important distinction has to be made here between explicit feedback and implicit feedback.'

print(len(paragraph))

c_splitter = CharacterTextSplitter(
            chunk_size = 150,
            chunk_overlap = 25,
            separator = ' ')
c_splitter.split_text(paragraph)

r_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 150,
            chunk_overlap = 25,
            separators = ["\n\n", "\n", " ", "", "."])
r_splitter.split_text(paragraph)



# ADDING METADATA TO CHUNKS

print(pages[0].metadata)

''' MarkdownHeaderTextSplitter adds header split metadata to each split. '''

markdown_doc = """ # Title\n\n
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n\\
### Section \n\n\ \
Hi this is Lance \n\n
## Chapter  2\n\n \
Hi this is Molly"""

headers_to_split_on = [
    ("#", "Header1"),
    ("##","Header2"),
    ("###","Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_doc)

print(md_header_splits[0],"\n")
print(md_header_splits[1])







######################## VECTOR STORES AND EMBEDDINGS ############################

''' A vector store is a database that stores similar embeddings 
so that it is easier to retrieve similar data quickly.'''

loaders = [
    PyPDFLoader('/kaggle/input/infodata/Soham_Deshmukh_B21EE067_Exp_5.pdf'),
    PyPDFLoader('/kaggle/input/infodata/Soham_Deshmukh_B21EE067_Expt10.pdf'),
    PyPDFLoader('/kaggle/input/infodata/Soham_Deshmukh_B21EE067_Expt11.pdf'),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

for i in range(5):
    print(docs[i])
    print("\n")


text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap = 150)

splits = text_splitter.split_documents(docs)

from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

s1 = "i like dogs"
s2 = "i like canines"
s3 = "the weather is ugly outside"

e1 = embedding.embed_query(s1)
e2 = embedding.embed_query(s2)
e3 = embedding.embed_query(s3)

'''To find similarity'''
similarity = np.dot(e1, e2)



################ VECTOR STORE ############

from langchain.vectorstores import Chroma

persist_dir = '/kaggle/working/chroma/'

!rm -rf ./kaggle/working/chroma

vectordb = Chroma.from_documents(
            documents = splits,
            embedding = embedding,
            persist_directory = persist_dir
)

question = 'Ask some question realted to the text.'
docs = vectordb.similarity_search(question, k=3)    # k= means return 3 documents as output

print(docs[0].page_content)

vectordb.persist()            # makes db fixed for further use -- no loss





######################## RETRIEVAL ############################


''' Important for query time - retrieve most relevant split 

1) Maximum Marginal Relevance (MMR)
    - not select the most similar responses 
    - select more important info not relevant
    - create more diverse responses
    - 'fetch_k' parameter - k responses
    
2) LLM Aided Retrieval
    - Query is more than just the Question
    - LLM is used to convert question to query
    - Query contains - Filter (metadata) + Search Term
    
3) Compression
    - Shrink the responses to only the relevant info
    - Use Compression LLM to get most relevant parts
    
'''

texts = [
    """The death cap has a large and imposing epigeous (aboveground) fruiting body (basidiocarp), usually with a pileus (cap) from 5 to 15 centimetres (2 to 5+7⁄8 inches) across, initially rounded and hemispherical, but flattening with age. The color of the cap can be pale-green, yellowish-green, olive-green, bronze, or (in one form) white; it is often paler toward the margins, which can have darker streaks; it is also often paler after rain. The cap surface is sticky when wet and easily peeled—a troublesome feature, as that is allegedly a feature of edible fungi.""", 
    """The remains of the partial veil are seen as a skirtlike, floppy annulus usually about 1 to 1.5 cm (3⁄8 to 5⁄8 in) below the cap. The crowded white lamellae (gills) are free. The stipe is white with a scattering of grayish-olive scales and is 8 to 15 cm (3+1⁄8 to 5+7⁄8 in) long and 1 to 2 cm (3⁄8 to 3⁄4 in) thick, with a swollen, ragged, sac-like white volva (base). As the volva, which may be hidden by leaf litter, is a distinctive and diagnostic feature, it is important to remove some debris to check for it. Spores: 7-12 x 6-9 μm. Smooth, ellipsoid, amyloid.""",
    """The smell has been described as initially faint and honey-sweet, but strengthening over time to become overpowering, sickly-sweet and objectionable.[32] Young specimens first emerge from the ground resembling a white egg covered by a universal veil, which then breaks, leaving the volva as a remnant. The spore print is white, a common feature of Amanita. The transparent spores are globular to egg-shaped, measure 8–10 μm (0.3–0.4 mil) long, and stain blue with iodine. The gills, in contrast, stain pallid lilac or pink with concentrated sulfuric acid."""
]

smalldb = Chroma.from_texts(texts, embedding = embedding)

question = "Tell me abput all-white mushrooms with large fruiting bodies"
smalldb.similarity_search(question, k=2)

#### MMR ####
print(smalldb.max_marginal_relevance_search(question, k=2, fetch_k=3))


#########  GETTING RELEVANT DOCUMENTS FIRST #########

docs = vectordb.similarity_search(question, k=3, filter={"source":"docs/..."})

for d in docs:
    print(d.metadata)



from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AtrributeInfo{
        name="source",
        description="The lecture the chunk is from, should be from one of 'docs/...'",
        type="string",
    },
    AttributeInfo{
        name="page",
        description="The oage from the lecture",
        type="string"
    }
]

document_content_description = "Notes"
llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "What do they say about regression in the third lecture?"

docs = retriever.get_relevant_documents(question)

for doc in docs:
    print(doc.metadata)



########### CONTEXTUAL CHAIN RETRIEVAL ##########

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = OpenAI(temperature = 0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retirever = vectordb.as_retriever(search_type='mmr')          # MMR - to remove repeating info
)

question = "what did they say about matlab ?"
compressed_docs = compression_retriever.get_relevant_documents(question)

# print(compressed_docs)



from langchain.retrievers import SVMRetriever, TFIDFRetriever
from langchain.langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("../address/..")
pages = loader.load()

all_pages_text = [p.page_content for p in pages]
joined_page_text = " ".join(all_page_text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=50)
splits = text_splitter.split_text(joined_page_text)

svm_retriever = SVMRetriever.from_texts(splits, embeddings)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

question = "what do they say about matlab ?"

docs_svm = svm_retriever.get_relevant_documents(question)
docs_tfidf = tfidf_retriever.get_relevant_documents(question)








############################ QUESTION ANSWERING ########################### 

import os
import openai
import sys

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

persist_directory = 'docs/chroma/'
embedding  = OpenAIEmbeddings()

vectordb = Chroma(persist_directory = persist_directory , embedding_function = embedding)
print(vectordb._collection.count())

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAqa

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature=0)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

result = qa_chain({"query":question})
result["result"]

from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question
{context}
Question: {question}
Helpful answer:
"""

QA_chain_prompt = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever= vectordb.as_retriever(),
    return_source_documents = True,
    chain_type_kwargs = {"prompt":QA_chain_prompt}
)

question = "Is probability a class topic?"
result = qa_chain({"prompt":question})

print(result["Result"])
print(result["source_documents"])



''' What happens inside a Lang QA chain ??'''
# Document chain - multiple calls to Lang model
# Input output for each doc
# All responses stuffed together
# Combination of all responses -- given as output






######################## MEMORY ELEMENT IN RETRIEVAL CHAINS ############################

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key = "chat_history",
    return_messages = True
)
# returns a list of chats


from langchain.chains import ConversationalRetrievalChain

retriever = vectordb.as_retriever()
qa = ConversationRetrievalChain.from_llm(
    llm,
    retriever = retriever,
    memory = memory
)

question = "Is probability a class topic?"
result =  qa({"question":question})
print(result["Answer"])

question = "Why are those prerequisites necessary?"
result =  qa({"question":question})
print(result["Answer"])