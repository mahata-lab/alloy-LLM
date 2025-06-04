import os
from google import genai
from google.genai import types
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import json
import numpy as np

class AlloyDataExtractor:

    """
    A class to extract and process alloy data from PDF research papers
    using the Google Gemini API.
    """

    MODEL_ID = "gemini-2.0-flash"
    ATOMIC_WEIGHTS = {
        "Al": 26.98, "Si": 28.09, "Cu": 63.55, "Mg": 24.31, "Zn": 65.38,
        "Mn": 54.94, "Cr": 51.996, "Zr": 91.22, "Sc": 44.96, "Ti": 47.87,
        "Fe": 55.85, "Li": 6.94, "Ni": 58.69, "V": 50.94, "Ag": 107.87,
        "Ce": 140.12, "Co": 58.93, "Nb": 92.91, "Er": 167.26, "Mo": 95.95,
        "Titanium": 47.87, "Aluminum": 26.98, "Vanadium": 50.94, "Molybdenum": 95.95,
        "Chromium": 51.996, "Iron": 55.85, "Zinc": 65.38, "Magnesium": 24.31, "Copper": 63.55,
        "Cerium": 140.12, "Silicon": 28.09, "H": 1.008, "C": 12.011, "O": 15.999,
        "Cobalt": 58.93, "Sn": 118.71, "Pb": 207.2, "Ca": 40.078, "B": 10.811, "Cd": 112.414,
        "Be": 9.012, "Ta": 180.945, "Aluminium": 26.98
    }
    AM_KEYWORDS = {
        "electron beam melting": "EBM", "laser additive manufacturing": "LAM",
        'laser melting': 'LM', 'laser powder bed fusion': "LPBF",
        'selective laser melting': "SLM", 'powder bed fusion': "PBF",
        'wire arc additive manufacturing': "WAAM", 'wire-arc additive manufacturing': "WAAM",
        'directed energy deposition': "DED", 'electron powder bed fusion': "EPBF",
        'l-pbf': "LPBF", 'e-pbf': "EPBF", 'laser powder-bed fusion': "LPBF",
        'lb-pbf': "LPPBF", 'laser-powder bed fusion': "LPBF",
        'direct metal laser sintering': "DMLS", 'binder jetting': "BJT",
        'laser metal deposition': "LMD", 'Friction Stir Additive Manufacturing': "FSAM",
        'Additive Friction Stir Deposition': "AFSD", 'powder bed fusion direct metal laser sintering': "PBF-DMLS",
        'electron beam freeform fabrication': "EBFFF"
    }

    def __init__(self, api_key: str, drive_root: str, alloy: str):
        """
        Initializes the AlloyDataExtractor with API key and drive root.

        Args:
            api_key (str): Your GCP Gemini API key.
            drive_root (str): The root directory containing the PDF files.
        """
        self.client = genai.Client(api_key=api_key)
        self.drive_root = drive_root
        self.alloy = alloy
        self.files = self._get_pdf_files()
        self.extracted_responses = []
        self.df = pd.DataFrame()

    def _get_pdf_files(self) -> list:
        """
        Lists all PDF files in the specified drive root directory.

        Returns:
            list: A list of PDF file names.
        """
        return [f for f in os.listdir(self.drive_root) if f.endswith('pdf')]

    def _generate_content(self, system_instruction: str, prompt: str, file_upload: types.Blob = None) -> str:
        """
        Helper method to generate content using the Gemini model.

        Args:
            system_instruction (str): The system instruction for the model.
            prompt (str): The user prompt.
            file_upload (types.Blob, optional): The uploaded file for content generation. Defaults to None.

        Returns:
            str: The generated text response.
        """
        chat_config = types.GenerateContentConfig(system_instruction=system_instruction)
        contents = [file_upload, prompt] if file_upload else [prompt]
        response = self.client.models.generate_content(
            model=self.MODEL_ID,
            config=chat_config,
            contents=contents
        )
        return response.text

    def _clean_json_response(self, json_string: str) -> dict:
        """
        Cleans and decodes a JSON string from the model's response.

        Args:
            json_string (str): The raw JSON string from the model.

        Returns:
            dict: The decoded JSON as a dictionary, or an empty dictionary if decoding fails.
        """
        json_start_index = json_string.find('{')
        json_end_index = json_string.rfind('}') # Use rfind to get the last occurrence
        if json_start_index != -1 and json_end_index != -1:
            cleaned_json_string = json_string[json_start_index : json_end_index + 1]
            try:
                return json.loads(cleaned_json_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Offending string: {cleaned_json_string}")
        return {}

    def extract_alloy_data(self):
        """
        Extracts alloy data from PDF files using the Gemini API.
        """
        system_instruction = """
        You are an expert material scientist. Your area of expertise resides in being able to identify the aluminium alloy of focus from research papers
        and also identifying the alloy's following properties: yield strength (ys), ultimate tensile strength (uts), elongation (% strain), and the AM (additive manufacturing) process used.
        You will also provide the alloy's composition, carefully specifying which elements make up the alloy. You always return your response in the following
        json format: {alloy_name: 'example alloy', alloy_composition: 'example composition', ys: 'example ys', uts: 'example uts', elongation: 'example elongation', AM: 'example AM'}.
        If there is no alloy specified in the paper, you will inform us that there is no alloy mentioned. You will provide the value if a theoretical value is calculated or specified.
        You are very careful not to make mistakes in your work."""

        for file in self.files:
            print(f"Processing {file}...")
            pdf_path = os.path.join(self.drive_root, file)
            try:
                file_upload = self.client.files.upload(file=pdf_path)
                response_text = self._generate_content(
                    system_instruction=system_instruction,
                    prompt="Please identify the alloy in the research paper and include its properties in your response.",
                    file_upload=file_upload
                )
                self.extracted_responses.append((response_text, file))
            except InvalidArgument as e:
                print(f"Skipping file due to InvalidArgument error for {file}: {e}")
                continue
            except Exception as e:
                print(f"Skipping file due to an unexpected error for {file}: {e}")
                continue

    def process_extracted_data(self):
        """
        Cleans up the initial extracted data and converts it into a DataFrame.
        """
        data_dicts = []
        valid_responses = []
        for i, (json_string, file_name) in enumerate(self.extracted_responses):
            cleaned_data = self._clean_json_response(json_string)
            if cleaned_data:
                data_dicts.append(cleaned_data)
                valid_responses.append((json_string, file_name))
            else:
                print(f"Skipping response from file {file_name} due to JSON decoding error.")

        self.df = pd.DataFrame(data_dicts)
        self.extracted_responses = valid_responses # Update to only include valid responses

    def get_alloy_composition(self):
        """
        Uses the Gemini API to get the detailed percent composition of alloys.
        """
        if self.df.empty:
            print("No data to process. Run extract_alloy_data() and process_extracted_data() first.")
            return

        system_instruction = """
        You are an expert material scientist, focused on alloys. Your area of expertise resides in being able to stratify the percent composition by weight of an
        alloy from its name and composition. For example, you expertly determine that the alloy with name Al-Si2Mg2 is made up of 2% silicon, 2% magnesium, and 96% aluminium,
        because you know that when a number does not accompany aluminium (or any other element for that matter), it is implied that the remainder of the alloy is composed of that element.
        Similarly, you can tell that an alloy with composition 'Titanium, 6% Aluminium, 4% Vanadium' is made up of 6% aluminium, 4% vanadium, and 90% titanium, because you
        know that when a number does not accompany titanium (or any other element for that matter), it is implied that the remainder of the alloy is composed of that element.
        You can expertly use both the alloy name and alloy composition together to strategy the percent composition by weight of that alloy using the rules described above.
        In the event that the weight for a given element is a range, for example "Cu": "3%-5%", you will simply take the midpoint of the range. You are also careful to
        present the elements in their abbreviated form as they appear in the periodic table of elements. Finally,you make it a point to create json files to show the
        percent composition in the following format: {"element1": "some percentage", "element2": "some percentage", "element3": "some percentage"}.
        You are very careful only to return the alloy's composition in the format described above; you will not return the alloy's name.
        You are very careful not to make mistakes in your work. If you do not know the answer from the alloy name and composition, you will inform me that you are uncertain. You will not try to guess or predict anything outside of the data."""

        alloy_composition_data = []
        original_indices = self.df.index.tolist()
        updated_responses = []

        for index, row in self.df[['alloy_name', 'alloy_composition']].iterrows():
            prompt_data = row.to_dict()
            response_text = self._generate_content(
                system_instruction=system_instruction,
                prompt=f"{prompt_data}\nPlease break down the alloy's percent composition by element based on its name and composition."
            )
            cleaned_data = self._clean_json_response(response_text)
            if cleaned_data:
                alloy_composition_data.append(cleaned_data)
                updated_responses.append(self.extracted_responses[original_indices.index(index)])
            else:
                print(f"Skipping alloy composition for index {index} due to JSON decoding error.")
                # Mark this index for removal from the main DataFrame
                self.df.drop(index, inplace=True)

        self.df = self.df.reset_index(drop=True)
        self.extracted_responses = updated_responses
        alloy_composition_df = pd.DataFrame(alloy_composition_data)
        self.df = pd.concat([self.df, alloy_composition_df], axis=1)

    def _process_property(self, column_name: str, instruction: str):
        """
        Helper method to process and clean numerical properties (YS, UTS, Elongation).

        Args:
            column_name (str): The name of the column to process (e.g., 'ys', 'uts', 'elongation', 'AM').
            instruction (str): The system instruction for the Gemini model.
        """
        if self.df.empty:
            print("No data to process. Run previous steps first.")
            return

        final_values = []
        original_indices = self.df.index.tolist()
        updated_responses = []

        system_instruction = instruction
        for index, value in self.df[column_name].items():
            response_text = self._generate_content(
                system_instruction=system_instruction,
                prompt=f"{value}\nPlease provide the value."
            )
            if 'Not specified' in response_text:
                print(f"Skipping {column_name} for index {index} as 'Not specified'.")
                # Mark this index for removal from the main DataFrame
                self.df.drop(index, inplace=True)
            else:
                final_values.append(response_text.strip())
                updated_responses.append(self.extracted_responses[original_indices.index(index)])

        self.df = self.df.reset_index(drop=True)
        self.extracted_responses = updated_responses
        self.df[column_name] = final_values


    def clean_am_process_data(self):
        """
        Cleans and abbreviates the AM process data.
        """
        instruction = f"""
        You are an expert material scientist, focused on the physical and mechanical properties of alloys. You are given
        an Additive Process (AM) and you are tasked with just providing the abbreviation for the respective AM. For example,
        you know to abbreviate "Laser Powder Bed Fusion" as "LPBF". Here is a dictionary that you have available to reference.
        {self.AM_KEYWORDS}. You use this dictionary as a general guide for your task. You also know that if you are given an
        abbreviation, you will not make any changes, you will simply return the abbreviation as is. Meaning, if you are given
        'SLM', you know to return 'SLM' without making any changes to it. Lastly, if you see a 'NaN' or
        'Not specified', you know not to estimate a value, but rather to say 'Not specified'. You are careful not to make
        any mistakes and you double check your answer to make sure you are correct."""

        self._process_property('AM', instruction)


    def clean_ys_data(self):
        """
        Cleans and extracts numerical yield strength (YS) data.
        """
        instruction = """
        You are an expert material scientist, focused on the physical and mechanical properties of alloys. You are given
        a Yield Strength (YS) that may be accompanied by its units and error. You are tasked with just providing the value.
        For example, if you see '495 MPa', you will return just '495'. If you see '600 MPa +- 10', you know just to return '600'.
        One last example is that if you see multiple YS terms, like if you are given multiple YS values in the context of different
        treatments or element concentrations in different alloys, you know to take the average of the multiple YS values and return the
        average YS. Lastly, if you see a 'NaN' or 'Not specified', or if you simply do not know the value, you know not to estimate
        a value, but rather to say 'Not specified'. You are careful not to make any mistakes and you double check your answer to
        make sure you are correct."""
        self._process_property('ys', instruction)

    def clean_uts_data(self):
        """
        Cleans and extracts numerical ultimate tensile strength (UTS) data.
        """
        instruction = """
        You are an expert material scientist, focused on the physical and mechanical properties of alloys. You are given
        a Ultimate Tensile Strength (UTS) that may be accompanied by its units and error. You are tasked with just providing the value.
        For example, if you see '495 MPa', you will return just '495'. If you see '600 MPa +- 10', you know just to return '600'.
        One last example is that if you see multiple UTS terms, like if you are given multiple UTS values in the context of different
        treatments or element concentrations in different alloys, you know to take the average of the multiple UTS values and return the
        average UTS. Lastly, if you see a 'NaN' or 'Not specified', or if you simply do not know the value, you know not to estimate
        a value, but rather to say 'Not specified'. You are careful not to make any mistakes and you double check your answer to
        make sure you are correct."""
        self._process_property('uts', instruction)

    def clean_elongation_data(self):
        """
        Cleans and extracts numerical elongation data.
        """
        instruction = """
        You are an expert material scientist, focused on the physical and mechanical properties of alloys. You are given
        an Elongation percentage that may be accompanied by its units and error. You are tasked with just providing the value.
        For example, if you see '5.4%', you will return just '5.4'. If you see '6% +- 0.5%', you know just to return '6'.
        One last example is that if you see multiple elongation percentage terms, like if you are given multiple elongation percentage values in the context of different
        treatments or element concentrations in different alloys, you know to take the average of the multiple elongation percentage values and return the
        average elongation percentage. Lastly, if you see a 'NaN' or 'Not specified', or if you simply do not know the value, you know not to estimate
        a value, but rather to say 'Not specified'. You are careful not to make any mistakes and you double check your answer to
        make sure you are correct."""

        self._process_property('elongation', instruction)

    def _weight_to_atom_percent(self, weight_dict: dict) -> dict:
        """
        Convert a dictionary of element weight % values to atom %.

        Args:
            weight_dict (dict): Dictionary with element symbols and their weight percentages.

        Returns:
            dict: Dictionary with element symbols and their atomic percentages.
        """
        mol_dict = {}
        total_mol = 0.0

        for el, wt in weight_dict.items():
            if el in self.ATOMIC_WEIGHTS:
                try:
                    mol = float(wt) / self.ATOMIC_WEIGHTS[el]
                    mol_dict[el] = mol
                    total_mol += mol
                except ValueError:
                    print(f"Could not convert weight '{wt}' for element '{el}' to numeric. Skipping.")
                    continue


        atom_pct = {el: round((mol / total_mol) * 100, 2) for el, mol in mol_dict.items()} if total_mol > 0 else {}
        return atom_pct

    def convert_to_atomic_percentage(self):
        """
        Converts elemental weight percentages to atomic percentages.
        """
        if self.df.empty:
            print("No data to process. Run previous steps first.")
            return

        # Identify columns that are likely element compositions (dynamically)
        # Exclude known non-composition columns and columns that might have been added later
        excluded_cols = ['alloy_name', 'alloy_composition', 'ys', 'uts', 'elongation', 'AM']
        element_cols = [col for col in self.df.columns if col not in excluded_cols]
        elements_df = self.df[element_cols].copy()

        # Clean string artifacts from element columns
        for col in elements_df.columns:
            elements_df[col] = elements_df[col].astype(str).str.replace('%', '').str.strip()

        # Convert to numeric, coercing errors to NaN
        for col in elements_df.columns:
            elements_df[col] = pd.to_numeric(elements_df[col], errors='coerce')

        # Fill NaN with 0 for composition calculation, then convert to dicts
        elements_df = elements_df.replace(np.nan, 0)
        elements_dicts = elements_df.to_dict(orient='records')

        atomic_percentages = []
        for element_dict in elements_dicts:
            atomic_percentages.append(self._weight_to_atom_percent(element_dict))

        atomic_df = pd.DataFrame(atomic_percentages)

        # Drop original weight percentage columns from self.df before concatenating atomic percentages
        self.df = self.df.drop(columns=element_cols, errors='ignore')
        self.df = pd.concat([self.df, atomic_df], axis=1)


    def generate_final_output(self, output_filename: str = "alloy_data_output.xlsx"):
        """
        Generates the final cleaned and processed DataFrame and saves it to an Excel file.

        Args:
            output_filename (str): The name of the Excel file to save.
        """
        if self.df.empty:
            print("No data to export. Ensure all processing steps have been run.")
            return

        # Add paper names back to the DataFrame
        # The self.extracted_responses should contain (response_text, file_name) tuples
        papers = [file_name for _, file_name in self.extracted_responses]
        papers_df = pd.DataFrame(papers, columns=['paper'])

        # Align papers_df with the current self.df after potential row drops
        # This is a crucial step to ensure correct paper-data mapping
        # We assume self.extracted_responses is already filtered to match self.df's rows
        self.df['paper'] = papers_df['paper'].reset_index(drop=True)

        # Reorder columns for final presentation
        # Get all columns from self.df dynamically, except 'alloy_name', 'ys', 'uts', 'elongation', 'AM', 'paper'
        composition_cols = [col for col in self.df.columns if col not in ['alloy_name', 'ys', 'uts', 'elongation', 'AM', 'paper', 'alloy_composition']]

        final_cols_order = ['alloy_name'] + composition_cols + ['ys', 'uts', 'elongation', 'AM', 'paper']
        self.df = self.df[final_cols_order]

        # Drop rows with any NaN values in the final DataFrame for a clean output
        self.df.dropna(inplace=True)
        self.df = self.df[self.df[f'{self.alloy}'] > 80]

        self.df.to_excel(output_filename)#, sep=' ', index=False)
        print(f"Processed data saved to {output_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    api_key_input = input("Please enter your GCP Gemini API key: ")
    drive_root_input = input("Please enter your folder path containing PDF files: ")

    extractor = AlloyDataExtractor(api_key=api_key_input, drive_root=drive_root_input, alloy='Al')

    # Step 1: Extract initial alloy data from PDFs
    print("\n--- Step 1: Extracting alloy data from PDFs ---")
    extractor.extract_alloy_data()
    extractor.process_extracted_data()

    # Step 2: Get and clean alloy composition
    print("\n--- Step 2: Getting and cleaning alloy composition ---")
    extractor.get_alloy_composition()

    # Step 3: Clean AM process data
    print("\n--- Step 3: Cleaning AM process data ---")
    extractor.clean_am_process_data()

    # Step 4: Clean YS data
    print("\n--- Step 4: Cleaning Yield Strength (YS) data ---")
    extractor.clean_ys_data()

    # Step 5: Clean UTS data
    print("\n--- Step 5: Cleaning Ultimate Tensile Strength (UTS) data ---")
    extractor.clean_uts_data()

    # Step 6: Clean Elongation data
    print("\n--- Step 6: Cleaning Elongation data ---")
    extractor.clean_elongation_data()

    # Step 7: Convert weight percentages to atomic percentages
    print("\n--- Step 7: Converting composition to atomic percentages ---")
    extractor.convert_to_atomic_percentage()

    # Step 8: Generate final output
    print("\n--- Step 8: Generating final output Excel file ---")
    extractor.generate_final_output()

    print("\nData extraction and processing complete!")