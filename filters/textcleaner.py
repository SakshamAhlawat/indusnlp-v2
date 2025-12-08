import codecs
import os
import re
from lxml import html


class TextCleaner:
    def __init__(self, config, clean_html=False):
        """
        Initialize the TextCleaner with a given configuration.

        Parameters:
        ----------
        config : list of tuple
            Each tuple contains a method name followed by its arguments.
            The method name should be a string that matches the name of one of the
            methods in this class (e.g., 'remove_line_with_keyword'). The arguments
            for each method vary based on the method, but they are typically a list
            of strings or patterns. If the argument is a string ending with ".txt",
            it is assumed to be a filepath and the file will be read to get the list of
            keywords or patterns.

        Example Config:
        ---------------
        config = [
            ("remove_line_with_pattern", [r".*\{.*?\}"]),
            ("remove_line_and_before", ["- फोटो :"]),
            ("remove_line_and_below", ["Next Article"]),
            ...
        ]
        """
        self.config = []
        self.clean_html = clean_html

        for method, args in config:
            if isinstance(args, str) and args.endswith(".txt"):
                self.config.append((method, self._read_file(args)))
            else:
                self.config.append((method, args))

    def clean_html_with_lxml(self, raw_html):
        """Clean HTML content using lxml."""
        tree = html.fromstring(raw_html)
        text = tree.text_content()
        return text

    def filter_punctuated_lines(self, text):
        """Filter lines that end with specified Hindi punctuation marks."""
        pattern = r".*[।॥?,.]$"
        lines = text.split("\n")
        return "\n".join([line for line in lines if re.search(pattern, line)])
    

    def decode_unicode_escapes(self,text):
        """Decodes Unicode escape sequences in a string while preserving normal text."""
        
        def is_double_escaped(text):
            double_escaped_patterns = [r'\\\\', r'\\u[0-9a-fA-F]{4}']
            return any(re.search(pattern, text) for pattern in double_escaped_patterns)
        
        def decode_match(match):
            try:
                return bytes(match.group(), "utf-8").decode("unicode_escape")
            except:
                return match.group()  # return the original string if decoding fails

        # Check if the text is double-escaped and decode accordingly
        if is_double_escaped(text):
            try:
                text = codecs.decode(text, 'unicode_escape')
            except UnicodeDecodeError:
                pass

        # Then, replace Unicode escape sequences
        return re.sub(r'\\u[0-9a-fA-F]{4}', decode_match, text)

    def _read_file(self, filepath, decode_escapes=False):
        """
        Read a file and return its content as a multiline string, optionally decoding Unicode escape sequences.

        Parameters:
        ----------
        filepath : str
            Filepath to read from.
        decode_escapes : bool, optional
            Whether to decode unicode escape sequences, by default False.

        Returns:
        -------
        str
            The file content.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            if decode_escapes:
                content = self.decode_unicode_escapes(content)
            return content

    def __call__(self, input, filter_punctuation=False, decode_escapes=False):
        """
        Apply all configured cleaning methods to the input text or file path.
        
        Parameters:
        ----------
        input : str
            A string containing the text to be cleaned or a filepath to a text file.
        filter_punctuation : bool, optional
            Whether to filter lines based on punctuation, by default False.
        decode_escapes : bool, optional
            Whether to decode unicode escape sequences, by default False.

        Returns:
        -------
        str
            The cleaned text.
        """
        text = ""

        # Check if the input is a filepath and read the file
        if isinstance(input, str) and os.path.isfile(input):
            text = self._read_file(input, decode_escapes)
        elif isinstance(input, str):
            # Treat input as the text to be cleaned
            text = input
            if decode_escapes:
                text = self.decode_unicode_escapes(text)
        else:
            raise ValueError("Input must be a filepath or a string of text")

        if self.clean_html:
            text = self.clean_html_with_lxml(text)
            if not text:
                return text
        
        text = self.handle_whitespace(text, None)

        if filter_punctuation:
            text = self.filter_punctuated_lines(text)
            if not text:
                return text

        for method_name, args in self.config:
            method = getattr(self, method_name, None)
            if callable(method):
                text = method(text, args)
                if not text:
                    return text
            else:
                print(f"Method {method_name} not found in the class.")

        return text

    def add_newline_on_pattern(self, text, patterns):
        """Insert a newline for each match of given patterns."""
        for pattern in patterns:
            text = re.sub(pattern, r"\1\n", text, flags=re.DOTALL)
        return text

    def select_on_pattern(self, text, patterns):
        """Select text for given patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                text = match.group(1)
        return text

    def insert_on_pattern(self, text, args):
        """Insert on matrched pattern."""
        pattern, rep = args[0], args[1]
        return re.sub(pattern, rep, text, flags=re.DOTALL)

    def remove_line_with_keyword(self, text, keywords):
        """Remove lines containing any of the specified keywords."""
        lines = text.split("\n")
        return "\n".join(
            [line for line in lines if not any(keyword in line for keyword in keywords)]
        )

    def remove_line_with_pattern(self, text, patterns):
        """Remove lines matching any of the given patterns."""
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        return text

    def remove_line_and_before(self, text, keywords):
        """Remove a line and the line before it if it contains a keyword."""
        lines = text.split("\n")
        to_remove = set()
        for keyword in keywords:
            for i, line in enumerate(lines):
                if keyword in line:
                    to_remove.add(i)
                    if i > 0:
                        to_remove.add(i - 1)
        return "\n".join(
            [line for idx, line in enumerate(lines) if idx not in to_remove]
        )

    def remove_line_and_after(self, text, keywords):
        """Remove a line and the line after it if it contains a keyword."""
        lines = text.split("\n")
        to_remove = set()
        for keyword in keywords:
            for i, line in enumerate(lines):
                if keyword in line:
                    to_remove.add(i)
                    if i < len(lines) - 1:
                        to_remove.add(i + 1)
        return "\n".join(
            [line for idx, line in enumerate(lines) if idx not in to_remove]
        )

    def remove_line_and_above(self, text, keywords):
        """Remove a line and all lines above it if it contains a keyword."""
        lines = text.split("\n")
        for keyword in keywords:
            to_remove = set()
            for i, line in enumerate(lines):
                if keyword in line:
                    if i > 0:
                        to_remove.update(range(0, i + 1))
                    else:
                        to_remove.add(i)
            lines = [line for idx, line in enumerate(lines) if idx not in to_remove]
        return "\n".join(lines)

    def remove_line_and_below(self, text, keywords):
        """Remove a line and all lines below it if it contains a keyword."""
        lines = text.split("\n")
        for keyword in keywords:
            to_remove = set()
            for i, line in enumerate(lines):
                if keyword in line:
                    if i < len(lines) - 1:
                        to_remove.update(range(i, len(lines)))
                    else:
                        to_remove.add(i)
            lines = [line for idx, line in enumerate(lines) if idx not in to_remove]
        return "\n".join(lines)

    def remove_after_keyword(self, text, keywords):
        """Remove all words in a line after a specified keyword."""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword in line:
                    index = line.find(keyword)
                    lines[i] = line[:index].strip()
        return "\n".join(lines)

    def remove_single_word_lines(self, text, _):
        """Remove lines that consist of a single word."""
        lines = text.split("\n")
        return "\n".join([line for line in lines if len(line.split()) != 1])

    def remove_blank_lines(self, text, _):
        """Remove lines that are blank or contain only whitespace."""
        lines = text.split("\n")
        return "\n".join([line for line in lines if line.strip()])

    def remove_lines_starting_with(self, text, keywords):
        """Remove lines starting with any of the given keywords."""
        lines = text.split("\n")
        return "\n".join(
            [
                line
                for line in lines
                if not any(line.startswith(keyword) for keyword in keywords)
            ]
        )

    def handle_whitespace(self, text, _):
        """Trim leading and trailing whitespace from each line."""
        lines = text.split("\n")
        return "\n".join([line.strip() for line in lines])

    def remove_redundant_lines(self, text, _):
        """Remove duplicate lines from the text."""
        lines = text.split("\n")
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return "\n".join(unique_lines)

    def remove_lines_with_repeated_seqs(self, text, min_repeat):
        """Remove lines containing repeated sequences."""
        lines = text.split("\n")
        cleaned_lines = [
            line for line in lines if not self.has_repeated_substring(line, min_repeat)
        ]
        return "\n".join(cleaned_lines)

    def has_repeated_substring(self, line, min_repeat):
        """Check if a line contains repeated substrings."""
        length = len(line)
        checked_substrings = set()
        for i in range(1, length // min_repeat + 1):
            for start in range(0, i):
                substring = line[start : start + i]
                if substring in checked_substrings:
                    continue
                checked_substrings.add(substring)
                if substring * min_repeat in line:
                    return True
        return False

    def remove_patterns(self, text, patterns):
        """Remove all occurrences of the specified patterns."""
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL)
        return text


if __name__ == "__main__":
    config = [
        ("remove_line_with_pattern", [r".*\{.*?\}"]),
        ("remove_line_and_before", ["- फोटो :"]),
        ("remove_line_and_below", ["Next Article"]),
        (
            "remove_line_with_keyword",
            [
                "Updated", "Published by", "Link Copied",
                "Follow Us", "Next Article", "Followed",
                "अमर उजाला,", "News in Hindi", "Hindi news",
                "सब्सक्राइब करें", "डाउनलोड करें", "सब्सक्रिप्शन",
                "Disclaimer", "एड फ्री अनुभव", "Get all Sports",
                "ब्यूरो", "ब्यूरो,", "Get all India News", "Read the latest",
                "हम डाटा संग्रह टूल्स", "लेटेस्ट अपडेट्स",
            ],
        ),
        ("remove_patterns", [r", \{.*\}"]),
        ("handle_whitespace", None),
        ("remove_single_word_lines", None),
        ("remove_redundant_lines", None),
        ("remove_blank_lines", None),
        ("remove_lines_with_repeated_seqs", 4),
    ]

    cleaner = TextCleaner(config,clean_html=True)

    sample_text = """
    Updated Thu, 02 Jun 2016 01:54 PM IST
    India Today
    Link Copied
    गुजरात के गुलबर्ग सोसाइटी नरसंहार मामले में अहमदाबाद की एक विशेष अदालत ने 24 आरोपियों दोषी करार दिया गया है। 14 साल बाद विशेष अदालत ने ये फैसला सुनाया। कोर्ट ने मामले में 36 आरोपियों को बरी कर दिया है।
    एड फ्री अनुभव के लिए अमर उजाला प्रीमियम सब्सक्राइब करें
    अतिरिक्त ₹50 छूट सालाना सब्सक्रिप्शन पर
    सी-III-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई-आई -
    Next Article
    Disclaimer
    ], "{'pyle _ سیٹ _ नेम': 'बुककोर्पस', बुककोर्पस
    """

    cleaned_text = cleaner(sample_text,filter_punctuation=True)
    print(cleaned_text)
