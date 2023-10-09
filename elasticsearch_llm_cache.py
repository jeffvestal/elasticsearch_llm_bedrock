from datetime import datetime
from typing import Dict, List, Optional

from elasticsearch import Elasticsearch


class ElasticsearchLLMCache:
    def __init__(self, 
                 es_client: Elasticsearch,
                 index_name: Optional[str] = None,
                 es_model_id: Optional[str] = 'sentence-transformers__all-distilroberta-v1',
                 create_index = True
                ):
        """
        Initialize the ElasticsearchLLMCache instance.

        :param es_client: Elasticsearch client object
        :param index_name: Optional name for the index; defaults to 'llm_cache'
        """
        self.es = es_client
        self.index_name = index_name or 'llm_cache'
        self.es_model_id = es_model_id
        if create_index:
            self.create_index()

    def create_index(self) -> Dict:
        """
        Create the index if it does not already exist.
        """
        if not self.es.indices.exists(index=self.index_name):
            mappings = {
                "mappings": {
                    "properties": {
                        "prompt": {"type": "text"},
                        "response": {"type": "text"},
                        "create_date": {"type": "date"},
                        "last_hit_date": {"type": "date"},
                        "prompt_vector": {"type": "dense_vector",
                                          "dims": 768,
                                          "index": True,
                                          "similarity": "dot_product"
                                         } 
                    }
                }
            }
            
            self.es.indices.create(index=self.index_name, body=mappings)
            print(f"Index {self.index_name} created.")
            
            return {'cache_index': self.index_name, 'created_new' : True}
        else:
            print(f"Index {self.index_name} already exists.")
            return {'cache_index': self.index_name, 'created_new' : False}

    def query(self, 
              prompt_text: str, 
              similarity_threshold: Optional[int] = 0.5
             ) -> dict:
        """
        Query the index to find similar prompts.
        If a hit is found and returned, also update the `last_hit_date` for that doc
        to the current datetime

        :param prompt_text: The text of the prompt to find similar entries for
        :param similarity_threshold: The similarity threshold for filtering results
        :return: A dictionary containing the hits or an empty dictionary if no hits

        """
        knn = [
            {
                "field": "prompt_vector",
                "k": 1,
                "num_candidates": 1000,
                "similarity": similarity_threshold,
                "query_vector_builder": {
                    "text_embedding": {
                        "model_id": self.es_model_id,
                        "model_text": prompt_text
                 }
              }
            }
          ]

        #print(knn)
        
        fields= [
            "prompt",
            "response"
          ]
    
        resp = self.es.search(index=self.index_name,
                         knn=knn,
                         fields=fields,
                         size=1,
                         source=False
                             )

        # Check the size of the results
        if resp['hits']['total']['value'] == 0:
            # if 0 return an empty dictionary
            return {}
        else:
            # Update the 'last_hit_date' for the returned document
            doc_id = resp['hits']['hits'][0]['_id']
            update_body = {
                "doc": {
                    "last_hit_date": datetime.now()
                }
            }
            self.es.update(index=self.index_name, id=doc_id, body=update_body)
            
            return resp['hits']['hits'][0]['fields']['response'][0]


    def _generate_vector(self,
                        prompt: str
                       ) -> List[float]:

        docs =  [
                {
                  "text_field": prompt
                }
            ]
            
        embedding = self.es.ml.infer_trained_model(model_id=self.es_model_id, 
                                              docs=docs
                                             )

        return embedding['inference_results'][0]['predicted_value']

    def add(self, prompt: str, 
            response: str, 
            source: Optional[str] = None
           ) -> dict:
        """
        Add a new document to the index.

        :param prompt: The user prompt
        :param response: The LLM response
        :param prompt_vector: The prompt vector
        :param source: Optional source identifier for the LLM
        """
        prompt_vector = self._generate_vector(prompt=prompt)
       # print(prompt_vector)
        
        doc = {
            "prompt": prompt,
            "response": response,
            "create_date": datetime.now(),
            "last_hit_date": datetime.now(),
            "prompt_vector": prompt_vector,
            "source": source  # Optional
        }
        self.es.index(index=self.index_name, document=doc)
        return {'success caching new prompt & response' : True}

