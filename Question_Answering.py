import os
from google import genai
from google.genai import types
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import json
import numpy as np


class QuestionAnswer:
    """
    A class to digest Excel files generated from AlloyDataExtractor for question and answering
    using the Google Gemini API
    """

    MODEL_ID = 'gemini-2.0-flash'

    system_instruction = """
    You are a materials scientist who is highly skilled at interpreting alloy-related data from tables. When asked a question,
    say about an alloy's composition, you are very careful in to only reference the table that is uploaded when providing an answer.
    If the question does not refer to data directly in the table, you say "the question does not refer to a datapoint in the table";
    you are careful not to go to outside sources in crafting your answer. You are equally careful in only referencing the data that
    is available in the uploaded table. If the answer is not available in the uploaded table, you tell the user "the question does not 
    refer to a datapoint in the table". You do not go to outside sources when providing an answer. 
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

    def _get_txt_files(self) -> list:
        """
        Retriving the Excel files from the root directory

        Returns:
            A list of all excel files in the root directory
        """
        return [os.path.join(self.drive_root, f) for f in os.listdir(self.drive_root) if f.endswith('txt')]
    
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
            contents=[file_upload, prompt]
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


        
    

        
        
        

        






