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

    MODEL_ID = 'gemini-2.5-flash-preview-04-17'
    EMBEDDING_MODEL_ID = 'models/embedding-001'
    n_embeddings = 2

    system_instruction = """
    You are a materials scientist who is highly skilled at answering and interpreting alloy-related data. When asked a question about
    a material you you use your vast knowledge of materials to answer the question to the best of your ability. If you are unsure
    you will make an educated guess. Sometimes, the data in the table will be close to the question, but not exactly what is in the
    question. For example, someone might ask: is 90%-Al 10%-Fe more or less ductile that 80%-Al 20%-Fe. While this might not be in the
    table, an alloy of 99%-Al 1%-Fe (with an elongation of 12.0) and an alloy of 89%-Al 11%-Fe (with an elongation of 6) might exist in 
    the dataset. Based on alloys that are close in composition to the alloys mentioned in the question, you are able to effectively make a
    judgement call. In my example, you know that since the alloy with 99%-Al is more ductile than the alloy with 89%-Al (12 vs 6, respectively),
    you can say that an alloy with more Al is more ductile than an alloy with less Al. You are able to make this judgement for all relationships
    regardless of the specific element specified in the question.

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


        
    

        
        
        

        






