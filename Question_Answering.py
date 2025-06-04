import os
from google import genai
from google.genai import types
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import json
import numpy as np
import google.generativeai


class QuestionAnswer:
    """
    A class to digest Excel files generated from AlloyDataExtractor for question and answering
    using the Google Gemini API
    """

    MODEL_ID = 'gemini-2.0-flash'
    EMBEDDING_MODEL_ID = 'models/embedding-001'
    n_embeddings = 2

    system_instruction = """
    You are a materials scientist who is highly skilled at interpreting alloy-related data from tables. When asked a question,
    say about an alloy's composition, you are very careful in to only reference the table that is uploaded when providing an answer.
    If the question does not refer to data directly in the table, you say "the question does not refer to a datapoint in the table";
    you are careful not to go to outside sources in crafting your answer. You are equally careful in only referencing the data that
    is available in the uploaded table. If the answer is not available in the uploaded table, you tell the user "the question does not 
    refer to a datapoint in the table". However, if you receive a question about an alloy that is close to a datapoint in the 
    reference data, you do your best as a materials scientist to estimate the answer. However, you make it clear that your answer is 
    an estimate if your answer is about an alloy that does not exist in the reference data. You will also be receiving extra information
    before the question in the form of json-like strings. You can choose to ignore this additional information if it conflicts with what
    you carefully acertain from the uploaded table.
    """

    def __init__(self, api_key: str, drive_root: str):
        """
        Initializing the QuestionAnswer class with the API key and drive root

        Args:
            api_key (str): Your Gemini API key
            drive_root (str): Root directory of your .txt files
        """
        self.api_key = api_key
        self.drive_root = drive_root

        self.client = genai.Client(api_key=self.api_key)
        self.files = self._get_txt_files()
        self.text, self.embeddings = self._create_file_embeddings()

    def _get_txt_files(self) -> list:
        """
        Retriving the Excel files from the root directory

        Returns:
            A list of all excel files in the root directory
        """
        return [os.path.join(self.drive_root, f) for f in os.listdir(self.drive_root) if f.endswith('txt')]
    
    def _create_file_embeddings(self) -> list:
        """
        Creates embeddings of queried file. Each embedding is a single line. 

        Returns:
            A two lists (for now) of embeddings of the file to be queried and corresponding text
        """       

        text = []
        for file in self.files:
            df = pd.read_csv(file, sep=' ')
            df_json = df.to_json(orient="records")
            df_json_list = json.loads(df_json)
            for i in df_json_list:
                text.append(str(i))
        
        embeddings = []
        google.generativeai.configure(api_key=self.api_key)
        for i in text:
            embedding = google.generativeai.embed_content(model=self.EMBEDDING_MODEL_ID,
                                content=i,
                                task_type="retrieval_document"
            )
            embeddings.append(embedding['embedding'])
        
        return text, embeddings
    
    def _similarity_search(self, prompt: str) -> str:
        """
        Determines similarity between embedded prompt and list of embeddings

        Args:
            n (int): The number of embeddings to return and feed to Gemini API

        Returns:
            A str of most similar db embeddings to prompt embedding
        """
        prompt_embedding = google.generativeai.embed_content(model=self.EMBEDDING_MODEL_ID,
                                content=prompt,
                                task_type="retrieval_document"
            )
        prompt_embedding = prompt_embedding['embedding']

        cosine_similarity = []
        for i in self.embeddings:
            similarity = np.dot(prompt_embedding, i) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(i))
            cosine_similarity.append(similarity)
        
        cosine_similarity_sorted = sorted(cosine_similarity)
        n_largest = cosine_similarity_sorted[-self.n_embeddings:]
        n_most_similar = []
        for i in n_largest:
            index = cosine_similarity.index(i)
            n_most_similar.append(self.text[index])
        
        n_most_similar_joined = " ".join(n_most_similar)

        return n_most_similar_joined
    
    
    def generate_content(self, prompt: str) -> str:
        """
        Method to generate content using the Gemini API

        Args:
            system_instruction (str): The system instruction for the model -- acts as a guiding prompt
            prompt (str): The prompt
            file_upload (types.Blob, optional): The .txt file generated from the AlloyDataExtraction class for Q&A

        Returns:
            str: The generated response
        """
        similar_records = self._similarity_search(prompt=prompt)

        df = []
        for i in self.files:
            data = pd.read_csv(i, sep=' ')
            data = data.to_string()
            df.append(data)
        file_upload = df
        chat_config = types.GenerateContentConfig(system_instruction=self.system_instruction)
        response = self.client.models.generate_content(
            model=self.MODEL_ID,
            config=chat_config,
            contents=[file_upload, similar_records, prompt]
        )
        return response.text
    

if __name__ == "__main__":
    api_key_input = input("Please enter your GCP Gemini API key: ")
    drive_root_input = input("Please enter the root directory of your extracted data files: ")

    qa = QuestionAnswer(api_key=api_key_input, drive_root=drive_root_input)
    while True:
        prompt = input("Ask a questions about the data (type 'exit' to close the program)")
        if prompt.lower() == 'exit':
            break

        print(qa.generate_content(prompt=prompt))

    print("Closing Program...")


        
    

        
        
        

        






