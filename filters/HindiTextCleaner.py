import re
import os
import json


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


class HindiTextCleaner:
    def __init__(
        self,
        hindi_punctuations=None,
        sundry_stops=None,
        numbers=None,
        stopwords_path="Data/stopwords/",
        transliterate=False,
        remove_non_hindi=True
    ):
        data_dir = os.path.join(os.path.dirname(__file__), "data")

        if hindi_punctuations is None:
            self.hindi_punctuations = load_json(
                os.path.join(data_dir, "hindi_punctuations.json")
            )["punctuations"]
        else:
            self.hindi_punctuations = hindi_punctuations

        if sundry_stops is None:
            self.sundry_stops = load_json(os.path.join(data_dir, "sundry_stops.json"))[
                "stops"
            ]
        else:
            self.sundry_stops = sundry_stops

        if numbers is None:
            self.numbers = load_json(os.path.join(data_dir, "hindi_numbers.json"))[
                "numbers"
            ]
        else:
            self.numbers = numbers

        self.stopwords_path = stopwords_path
        self.remove_non_hindi = remove_non_hindi
        self.transliterate = transliterate
        self.transliterator = None
        if transliterate:
            # Lazy import of XlitEngine to avoid hard dependency at module import time
            try:
                from ai4bharat.transliteration import XlitEngine  # type: ignore

                self.transliterator = XlitEngine("hi", beam_width=4, rescore=True)
            except Exception as e:
                # If transliteration stack (fairseq etc.) is not compatible with this
                # Python version, fall back gracefully without transliteration.
                print(f"[HindiTextCleaner] Transliteration disabled due to error: {e}")
                self.transliterate = False
                self.transliterator = None

    def convert_to_hindi_numbers(self, english_num):
        """Converts English numbers to Hindi.

        Args:
        - english_num (str): The number in English.

        Returns:
        - str: The number in Hindi.
        """
        text = ""
        for char in english_num:
            if char.isdigit():
                text += self.numbers[int(char)]
            else:
                text += char
        return text

    def translit_english(self, token):
        """Transliterate English words to Hindi Devanagari.

        Args:
        - token (str): The word token to be transliterated.

        Returns:
        - str: The transliterated token.
        """
        match = re.search("[a-zA-Z]", token)
        if not match:
            return token

        # If transliterator could not be initialised (e.g. fairseq incompatibility),
        # just return the original token instead of crashing.
        if not self.transliterator:
            return token

        try:
            # Use XlitEngine's translit_word method and get top result
            result = self.transliterator.translit_word(token, topk=1)
            if result and "hi" in result and result["hi"]:
                return result["hi"][0]
        except Exception as e:
            print(f"[HindiTextCleaner] Error during transliteration for '{token}': {e}")

        return token

    def remove_non_printable(self, text):
        """
        Removes non-printable characters from the given text.

        Args:
        - text (str): The text from which non-printable characters need to be removed.

        Returns:
        - str: The text after removing non-printable characters.
        """
        pattern = r'[^\x20-\x7E\t\n\r\u0900-\u097F]'
        return re.sub(pattern, '', text)

    def remove_non_hindi_sentences(self, line):
        """
        Determines if a given line is predominantly Hindi.

        Args:
        - line (str): The line to be checked.

        Returns:
        - bool: True if the line is predominantly Hindi, False otherwise.
        """
        non_hindi_chars = re.findall(r"[^ऀ-ॿ0-9\s]", line)
        hindi_chars = re.findall(r"[ऀ-ॿ0-9\s]", line)

        total_chars = len(hindi_chars) + len(non_hindi_chars)
        if total_chars == 0:
            return False

        return len(hindi_chars) / total_chars >= 0.7
    
    def spell_check(self, text, threshold=0.9):
        """
        Corrects spelling mistakes in the text and removes sentences that don't meet the specified threshold.
        
        Args:
        - text (str): The text to be processed.
        - threshold (float): The threshold for removing sentences based on spelling correctness.
        
        Returns:
        - str: The text after spell check and sentence removal.
        
        Note:
        This feature is currently in progress and the implementation will be available in future releases.
        """
        # Implementation will be added in future releases.
        return text

    def auto_punctuate(self, text):
        """
        Automatically punctuates the given text using a trained mBERT based model.
        
        Args:
        - text (str): The text to be punctuated.

        Returns:
        - str: The punctuated text.
        
        Note:
        This feature is currently in progress and the implementation will be available in future releases.
        """
        # Implementation will be added in future releases.
        return text
    
    def named_entity_recognition(self, text):
        """
        Identifies and classifies named entities (like names of persons, organizations, locations) present in the text.
        
        Args:
        - text (str): The text to be processed.
        
        Returns:
        - list: A list of named entities found in the text.
        
        Note:
        This feature is currently in progress and the implementation will be available in future releases.
        """
        # Implementation will be added in future releases.
        return []

    def pos_tagging(self, text):
        """
        Determines the part-of-speech (noun, verb, adjective, etc.) for each word in the text.
        
        Args:
        - text (str): The text to be processed.
        
        Returns:
        - list: A list of tuples where each tuple contains a word and its corresponding part-of-speech tag.
        
        Note:
        This feature is currently in progress and the implementation will be available in future releases.
        """
        # Implementation will be added in future releases.
        return []

    def __call__(self, text, save=False):
        """Cleans the text by removing English words and converting numbers.

        Args:
        - text (str): The text to be cleaned.
        - save (str, optional): If path or name is passed it will saves the cleaned text to a file.

        Returns:
        - str: The cleaned text.
        """
        lines = text.split('\n')

        cleaned_lines = []
        for line in lines:
            line = self.remove_non_printable(line)
            
            words = line.split()
            if len(words) < 3:
                continue

            if self.remove_non_hindi and not self.remove_non_hindi_sentences(line):
                continue

            english_numbers = re.findall("[0-9]+", line)
            # Extract individual digits and store in a set to get unique digits
            english_numbers = set(digit for number in english_numbers for digit in number)

            # Convert set to a sorted list (optional)
            english_numbers = sorted(english_numbers)
            for number in english_numbers:
                line = line.replace(number, self.convert_to_hindi_numbers(number))

            if self.transliterate:
                tokens = line.split()
                line = " ".join(map(self.translit_english, tokens))

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        #use this configuration for pretaining
        #if len(text)<60:
            #text = ""

        if save:
            with open(save, "w", encoding="utf-8") as f:
                f.write(text)
                print("File saved")

        return text

    def find_stopwords(self, dialect):
        """Finds stopwords for a given dialect.

        Args:
        - dialect (str): The dialect for which to find stopwords.

        Returns:
        - list: A list of stopwords for the given dialect.
        """
        _arrstopwords = []
        file_name = dialect + "_stopwords.txt"
        file_path = os.path.join(self.stopwords_path, file_name)
        try:
            with open(file_path, "r", encoding="UTF-8") as f:
                txt = f.readlines()
            for item in txt:
                _arrstopwords.append(item.replace("\n", ""))
        except FileNotFoundError:
            print(f"Stopwords file for dialect {dialect} not found.")
        return _arrstopwords


if __name__ == "__main__":
    cleaner = HindiTextCleaner(transliterate=True)
    
    sample_text = """४ ३.1४16 + ८१० ३.1४16 संसद \n के विशेष सत्र (Parliament Special Session) के बीच कल यानी, सोमवार, 18 सितंबर को प्रधानमंत्री नरेंद्र मोदी (PM Narendra Modi) की अध्यक्षता में हुई केंद्रीय कैबिनेट बैठक हुई जिसमें  महिला आरक्षण बिल (Women's Reservation Bill) मंजूरी दे दी गई है. 
    सूत्रों के हिसाब से यह खबर सामने आ रही है. 
    मोदी कैबिनेट की बैठक में लोकसभा और विधानसभाओं जैसी निर्वाचित संस्थाओं में 33 फीसदी महिला आरक्षण (Women Quota Bill 2023) पर मुहर लग गई है. 
    मीडिया रिपोर्ट्स के अनुसार, महिला आरक्षण बिल को आज यानी  मंगलवार को लोकसभा में नए संसद भवन (New Parliament Building) में पेश किया जाएगा.
    This line has less than three words.
    This line contains 90% non-Hindi characters and should be removed.
    12345 को १२३४५ में परिवर्तित किया जाना चाहिए"""

    '''sample_text = """मोदी कैबिनेट की बैठक में लोकसभा और विधानसभाओं जैसी निर्वाचित संस्थाओं में"""'''

    cleaned_text = cleaner(sample_text)
    print("Cleaned Text:", cleaned_text)
