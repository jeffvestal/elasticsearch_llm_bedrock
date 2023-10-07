# Elasticsearch RAG with Amazon Bedrock



# elasticsearch_llm_cache
Allowing Elasticsearch to be used as a caching layer for GenerativeAI Applications

Key benefits
- reduce costs for LLM services
- improve response speed as seen by end user

## class elasticsearch_llm_cache()
### create_index
used to create new indices for a cache
- check if index exists, if it does return a key like `created_new : false`
- create new index with correct mappings / settings
- Allow optional user created index cache name, either way return index name

#### mapping
```
prompt : text response : text
create_date : date
last_hit_date : date
prompt_vector : dense_vector
```

### query
Used at search time,<br>
most often before querying ES for doc to augment for RAG<br>
Can also work without RAG, before sending prompt to LLM

- run knn search against previously vectorized "user" inputs / prompts
- value to tune -> `similarity` max threshold to return only answers closest to query vector
- if close match ^ return top K inputs and responses
- also update 'last_hit_date' field

### add
used when there was not a cache hit and had to pull from LLM
- insert new doc with user prompt and llm response
- include other metadata like create_date
- part of this is vectorizing the prompt with ingest pipeline
- optional include the LLM source


