import os
import boto3
import json
import markdown
import streamlit as st
import elasticapm
import time
import base64
from string import Template
from elasticsearch import Elasticsearch
from elasticsearch_llm_cache.elasticsearch_llm_cache import ElasticsearchLLMCache


'''
Setup

- AWS CLI
Need to install AWS CLI and configure it with your AWS account credentials

- ENV Variables
export ELASTIC_CLOUD_ID=
export ELASTIC_USER=
export ELASTIC_PASSWORD=
export ELASTIC_APM_SERVER_URL=
export ELASTIC_APM_SECRET_TOKEN=
export ELASTIC_APM_ENVIRONMENT=
export ELASTIC_APM_SERVICE_NAME=
export ELASTIC_INDEX_DOCS=
'''


# https://ela.st/aws-vestal-preso
st.set_page_config(layout="wide")

#move to env if time
cache_index="movie_reviews-cache"

@st.cache_data()
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_background('images/smoke_blackyellow1.png')


def sidebar_bg(side_bg):

    side_bg_ext = 'png'
    st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
    )


side_bg = './images/sidebar_chairs3.jpg'
sidebar_bg(side_bg)

# sidebar logo
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image("images/elastic_logo_transp_100.png")


# Configure APM and Elasticsearch clients
@st.cache_resource
def initElastic():
    """
    Initialize Elastic APM and Elasticsearch clients.
    """
    apmclient = elasticapm.Client()
    elasticapm.instrument()

    es = Elasticsearch(
        cloud_id=os.environ['ELASTIC_CLOUD_ID'],
        basic_auth=(os.environ['ELASTIC_USER'], os.environ['ELASTIC_PASSWORD']),
        request_timeout=30
    )

    return apmclient, es


apmclient, es = initElastic()

# Set our data index
index = os.environ['ELASTIC_INDEX']

# Run an Elasticsearch query using hybrid RRF scoring of KNN and BM25
@elasticapm.capture_span("knn_search")
def search_knn(query_text, es):
    """
    Perform a KNN search with Elasticsearch.
    Args:
    - query_text: The text query for the search.
    - es: Elasticsearch client instance.
    Returns:
    - The body content and URL of the first hit.
    """
    # query = {
    #     "bool": {
    #         "must": [
    #             {
    #                 "match": {
    #                     "review_detail": {
    #                         "query": query_text
    #                     }
    #                 }
    #             }
    #         ]
    #     }
    # }

    knn = {
        "inner_hits": {
            "_source": False,
            "fields": [
                "passages.text"
            ]
        },
        "field": "passages.vector.predicted_value",
        "k": 5,
        "num_candidates": 100,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        }
    }

    # rank = {
    #     "rrf": {}
    # }

    fields = [
        "movie",
        "review_date",
        "review_detail",
        "rating",
        "review_summary",
        "reviewer",
        "helpful",
        "spoiler_tag",
        "review_id",
        "_score"
    ]


    resp = es.search(index=index,
                     #query=query,
                     knn=knn,
                     #rank=rank,
                     fields=fields,
                     size=10,
                     source=False)

#    st.json(resp['hits']['hits'][0]['fields'])
    results = [hit['fields'] for hit in resp['hits']['hits']]

    results = []
    for hit in resp['hits']['hits']:
        hit['fields']['_score'] = hit['_score']
        results.append((hit['fields']))

    return results


def truncate_text(text, max_tokens):
    """
    Truncate text to a specified maximum number of tokens.
    Args:
    - text: The text to truncate.
    - max_tokens: The maximum number of tokens to keep.
    Returns:
    - Truncated text.
    """
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

def clear_es_cache(es):
    match_all_query = {"query": {"match_all": {}}}
    clear_response = es.delete_by_query(index=cache_index, body=match_all_query)
    return clear_response


# Generate a response from ChatGPT based on the given prompt
def genAI(prompt, max_tokens=5000, temperature=0.5, top_p=1, top_k=250):
    """
    Generate a response from Claude model on Amazon Bedrock LLM service.
    Args:
    - prompt: The prompt to send to the model.
    - max_tokens, temperature, top_p, top_k: Parameters for the model.
    Returns:
    - The generated response.
    """
    # Initialize the Bedrock client
    bedrock = boto3.client(service_name="bedrock-runtime")

    # Prepare the payload
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    })

    # Invoke the Bedrock model
    response = bedrock.invoke_model(body=body, modelId="anthropic.claude-v2")

    # Process the response
    response_body = json.loads(response.get("body").read())
    completion = response_body.get("completion")

    return completion


def render_markdown_with_background(text, background_color="#101012", padding="10px", border_radius="10px"):
    """
    Render markdown text with a background color.
    """
    html_content = markdown.markdown(text)

    # Create the HTML string with inline CSS for the background
    html = f"""
    <div style="
        background-color: {background_color};
        padding: {padding};
        border-radius: {border_radius};
    ">
        <span style="color: white;">{html_content}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def calc_similarity(score, func_type='dot_product'):
    if func_type == 'dot_product':
        return (score + 1) / 2
    elif func_type in ['cosine', 'l2_norm']:
        #TODO
        return score
    else:
        return score


def toLLM(resp, usr_prompt, sys_prompt, neg_resp, show_prompt):
    """
    Generate a response using the LLM based on Elasticsearch results.
    Args:
    - resp: The response text from Elasticsearch.
    - url: The URL of the Elasticsearch result.
    - usr_prompt: The user-defined prompt template.
    - sys_prompt: System-defined prompt information.
    - neg_resp: Negative response text.
    - show_prompt: Boolean to show the full prompt.
    Returns:
    - The answer from the LLM.
    """

    col1, col2 = st.columns(2)

    with col2:
        st.subheader(':green[Elasticsearch Response]')
        st.json(resp)

    prompt_template = Template(usr_prompt)
    prompt_formatted = prompt_template.substitute(query=query, resp=resp, negResponse=negResponse)
    prompt_formatted = f"Human: {prompt_formatted}\n\nAssistant:"
    answer = genAI(prompt_formatted, max_tokens=5000, temperature=0.5, top_p=1, top_k=250)


    with col1:
        st.subheader(':yellow[GenAI Response]')
        try:
            answer_dict = json.loads(answer)

            # Display response from LLM
            st.header(':orange[Movie Recommendation]')
            st.subheader(f":violet[{answer_dict['recommended_movie']}]")
            render_markdown_with_background(answer_dict['llm_answer'])

            if answer_dict['review_id']:

                for review in resp:
                    if review['review_id'][0] == answer_dict['review_id']:
                        st.subheader(':yellow[Review used for recommendation]')
                        render_markdown_with_background(review['review_detail'][0])
                        s_score = calc_similarity(review['_score'], func_type='dot_product')
                        st.code(f"Review ID used: {answer_dict['review_id']} | Similarity Value: {s_score:.5f}")



            st.subheader(':blue[Other Suggestions]')
            render_markdown_with_background(f"<div class='markdown-container'>{answer_dict['other_suggestions']}")


        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(answer)
            st.text(answer)

    # Display full prompt if checkbox was selected
    if show_prompt:
        st.divider()
        st.subheader(':red[Full prompt sent to LLM]')
        st.code(prompt_formatted)

    return answer


@elasticapm.capture_span("cache_search")
def cache_query(cache, prompt_text, similarity_threshold=0.5):
    return cache.query(prompt_text=prompt_text, similarity_threshold=similarity_threshold)


@elasticapm.capture_span("add_to_cache")
def add_to_cache(cache, prompt, response):
    return cache.add(prompt=prompt, response=response)


# sidebar setup
st.sidebar.header("Elasticsearch :red[LLM Cache] Info")

### MAIN

# Init Elasticsearch Cache
cache = ElasticsearchLLMCache(es_client=es,
                              index_name=cache_index,
                              create_index=False # setting only because of Streamlit behavor
                             )
st.sidebar.markdown('`creating Elasticsearch Cache`')


# Only want to attempt to create the index on first run
if "index_created" not in st.session_state:
    st.sidebar.markdown('`running create_index`')
    cache.create_index(768)

    # Set the flag so it doesn't run every time
    st.session_state.index_created = True
else:
    st.sidebar.markdown('`index already created, skipping`')

# Prompt Defaults
prompt_default = """Answer this question: $query
Using only the information from these IMDB Movie Reviews. There are multiple reviews in chunks
the reviews are in the field `review_detail` The other fields provide additional context.
Return a sing top recommendation based on the reviews. Then, for all the other review chunks provided,
if they could potentially be a fit for the question, suggest them alternative movies that they might like. 
\n\n$resp\n\n
Format the answer in standard JSON  in the following format:
 {"recommended_movie" : ONLY the movie title here,
 "llm_answer" : Only the reason for recommending `recommended_movie` here in standard markdown code format,
 "other_suggestions": A bullet list of alternative suggestions and the rest of your response here, in standard markdown code format,
 "review_id" : The ID of the review that was used to answer the question or None if the answer was not found in the reviews
 }
 ALL of your response should fit in one of the 4 fields above. 
 
If the answer is NOT contained in the supplied reviews:
{"llm_answer" : reply '$negResponse,
"other_suggestions": "1. From the list of movies that were passed in review chunks select the closest option. 2. Provide the names of the movies that were passed in review chunks - field name `movie`,
"review_id" : None}
 ALL of your response should fit in one of the 3 fields above. 


DO NOT REPLY WITH ANYTHING OUTSIDE THE JSON FORMAT ABOVE. This is extremely important!!
For multi-line responses, use the `\\n` character to indicate a new line.
"""

system_default = 'You are a helpful assistant who provides movie recommendations based on user reviews. Your recomendations should be insiteful and witty.'
neg_default = "I'm unable to directly answer the question based on the information I have from IMDB movie reviews."

st.title(":green[Elasticsearch] Movie Recomender:orange[Bot]")

with st.form("chat_form"):
    query = st.text_input("Ask movie reviews a question: ",
                          placeholder='I want to see a funny space movie')

    with st.expander("Show Prompt Override Inputs"):
        # Inputs for system and User prompt override
        sys_prompt = st.text_area("create an alternative system prompt", placeholder=system_default,
                                  value=system_default)
        usr_prompt = st.text_area("create an alternative user prompt required -> \$query, \$resp, \$negResponse",
                                  placeholder=prompt_default, value=prompt_default)

        # Default Response when criteria are not met
        negResponse = st.text_area("Create an alternative negative response", placeholder=neg_default,
                                   value=neg_default)

    show_full_prompt = st.checkbox('Show Full Prompt Sent to AWS Bedrock')

    butt_col1, butt_col2, butt_col3 = st.columns(3)
    with butt_col1:
        query_button = st.form_submit_button("Run With Cache Check")
        # Slider for adjusting similarity threshold
        similarity_threshold_selection = st.slider("Select Similarity Threshold (dot_product - Higher Similarity means closer)",
                                         min_value=0.0, max_value=2.0,
                                         value=0.5, step=0.01)
    with butt_col2:
        refresh_button = st.form_submit_button("Refresh Cache with new call to AWS Bedrock")
    with butt_col3:
        clear_cache_butt = st.form_submit_button(':red[Clear LLM Cache]')



# Clear Cache Button
if clear_cache_butt:
    st.session_state.clear_cache_clicked = True

# Confirmation step
if st.session_state.get("clear_cache_clicked", False):
    apmclient.begin_transaction("clear_cache")
    elasticapm.label(action="clear_cache")

    # Start timing
    start_time = time.time()

    if st.button(":red[Confirm Clear Cache]"):
        response = clear_es_cache(es)
        st.success("Cache cleared successfully!", icon="🤯")
        st.session_state.clear_cache_clicked = False  # Reset the state

    apmclient.end_transaction("clear_cache", "success")

if query_button:
    apmclient.begin_transaction("query")
    elasticapm.label(search_method="knn")
    elasticapm.label(query=query)

    # Start timing
    start_time = time.time()

    # check the llm cache first
    query_check = cache_query(cache, prompt_text=query, similarity_threshold=similarity_threshold_selection)

    if query_check:
        st.sidebar.markdown('`cache match, using cached results`')
        st.subheader('Response from Cache')
        answer_dict = json.loads(query_check['response'][0])

        # Display response from LLM
        st.header('Movie Recommendation')
        st.subheader(f":violet[{answer_dict['recommended_movie']}]")
        render_markdown_with_background(answer_dict['llm_answer'])

        if answer_dict['review_id']:
            s_score = calc_similarity(query_check['_score'], func_type='dot_product')
            st.code(f"Review ID used: {answer_dict['review_id']} | Similarity Value: {s_score:.5f}")

        st.subheader('Other Suggestions')
        render_markdown_with_background(answer_dict['other_suggestions'])
        es_time = time.time()

    else:
        st.sidebar.markdown('`no cache match, querying es and sending to LLM`')
        resp = search_knn(query, es)  # run kNN hybrid query
        es_time = time.time()

        llmAnswer = toLLM(resp,
                          usr_prompt,
                          sys_prompt,
                          negResponse,
                          show_full_prompt,
                          )

        st.sidebar.markdown('`adding prompt and response to cache`')
        add_to_cache(cache, query, llmAnswer)

    # End timing and print the elapsed time
    elapsed_time = time.time() - start_time
    es_elapsed_time = es_time - start_time

    ct1, ct2 = st.columns(2)
    with ct1:
        st.markdown(f"`GenAI Time taken: {elapsed_time:.2f} seconds`")
    with ct2:
        st.markdown(f"`ES Query Time taken: {es_elapsed_time:.2f} seconds`")


    apmclient.end_transaction("query", "success")

if refresh_button:
    apmclient.begin_transaction("refresh_cache")
    st.sidebar.markdown('`refreshing cache`')

#    '''
#    Cache Refresh idea: set an 'invalidated' flag in the
#    already cached document and then call the LLM
#    '''

    elasticapm.label(search_method="knn")
    elasticapm.label(query=query)

    # Start timing
    start_time = time.time()

    st.sidebar.markdown('`skipping cache check - sending to LLM`')

    resp = search_knn(query, es)  # run kNN hybrid query
    es_time = time.time()

    llmAnswer = toLLM(resp,
                      usr_prompt,
                      sys_prompt,
                      negResponse,
                      show_full_prompt,
                      )

    st.sidebar.markdown('`adding prompt and response to cache`')
    add_to_cache(cache, query, llmAnswer)

    # End timing and print the elapsed time
    elapsed_time = time.time() - start_time
    es_elapsed_time = es_time - start_time

    ct1, ct2 = st.columns(2)
    with ct1:
        st.markdown(f"`GenAI Time taken: {elapsed_time:.2f} seconds`")
    with ct2:
        st.markdown(f"`ES Query Time taken: {es_elapsed_time:.2f} seconds`")

    st.sidebar.markdown(f"`ES Query Time taken: {es_elapsed_time:.2f} seconds`")
    st.sidebar.markdown(f"`GenAI Time taken: {elapsed_time:.2f} seconds`")

    apmclient.end_transaction("query", "success")

    st.sidebar.markdown('`cache refreshed`')
    apmclient.end_transaction("refresh_cache", "success")