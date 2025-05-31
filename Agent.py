import os
import json
import re
import spacy
from datetime import datetime
from typing import Union
from spacy.matcher import PhraseMatcher
!pip install transformers torch
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from huggingface_hub import login

# Load a spaCy model for NER while formatting
nlp = spacy.load("en_core_web_sm")

# Initialize the NER pipeline for EmailInfoExtractor
# We assume this part remains as the user requested changes to the ClassifierAgent, not the EmailInfoExtractor's reliance on NER
ner_model_name = "dslim/bert-base-NER" # Example NER model
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


# Memory Logging agent
class MemoryLogger:
    def __init__(self, filename="agent_logs.json"):
        self.filename = filename
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.logs = json.load(f)
        else:
            self.logs = []

    def log(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        self.logs.append(entry)
        with open(self.filename, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"[MemoryLogger] Logged: {entry}")

#Classifier Agent using spaCy's PhraseMatcher with more patterns
class ClassifierAgent:
    def __init__(self, logger: MemoryLogger, nlp):
        self.logger = logger
        self.nlp = nlp
        self.matcher = PhraseMatcher(nlp.vocab)
        self._add_patterns()

    def _add_patterns(self):
        # Added more patterns for better accuracy
        rfq_patterns = [self.nlp("quotation"), self.nlp("rfq"), self.nlp("quote"), self.nlp("price estimate"), self.nlp("request for price"), self.nlp("bid request")]
        invoice_patterns = [self.nlp("invoice"), self.nlp("bill"), self.nlp("statement of account")]
        complaint_patterns = [self.nlp("complaint"), self.nlp("issue"), self.nlp("problem"), self.nlp("feedback"), self.nlp("dissatisfied"), self.nlp("not working")]
        regulation_patterns = [self.nlp("regulation"), self.nlp("rule"), self.nlp("policy"), self.nlp("compliance")]
        self.matcher.add("RFQ", rfq_patterns)
        self.matcher.add("Invoice", invoice_patterns)
        self.matcher.add("Complaint", complaint_patterns)
        self.matcher.add("Regulation", regulation_patterns)

    def classify(self, input_text: str, file_name: str = "") -> dict:
        format_ = "Email"
        if file_name.endswith(".pdf"):
            format_ = "PDF"
        elif file_name.endswith(".json") or input_text.strip().startswith("{"):
            format_ = "JSON"

        doc = self.nlp(input_text.lower())  # Convert to lowercase for matching
        matches = self.matcher(doc)
        intent = "Unknown"  # Default intent if no match is found

        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]
            intent = string_id  # Assign the matched pattern label as intent
            break  # Stop after the first match

        result = {"format": format_, "intent": intent}
        self.logger.log(result)
        return result

#JSON Agent - No changes requested
class JSONAgent:
    def __init__(self):
        self.required_fields = ["customer", "items", "date", "total"]

    def process(self, json_data: dict) -> dict:
        reformatted = {
            "customer_name": json_data.get("customer"),
            "item_count": len(json_data.get("items", [])),
            "total_amount": json_data.get("total"),
            "date": json_data.get("date")
        }

        anomalies = [field for field in self.required_fields if field not in json_data]
        return {"reformatted": reformatted, "anomalies": anomalies}

#Email info formatter agent - No changes requested in its logic, relies on EmailInfoExtractor
class EmailAgent:
    def __init__(self):
        self.extractor = EmailInfoExtractor()

    def process(self, email_text: str) -> dict:
        info = self.extractor.extract(email_text)
        # Added more robust urgency check
        urgency = "High" if re.search(r'\b(asap|urgent|immediately)\b', email_text.lower()) else "Normal"
        format = {
            "sender": info["sender_name"],
            "company": info["company"],
            "intent": info["request_type"],
            "urgency": urgency,
            "date": info["date"],
            "products": info["products"]
        }
        return format

#Email extractor agent - Enhanced product and date extraction
class EmailInfoExtractor:
    def extract(self, email_text: str) -> dict:
        # Use the global ner_pipeline
        ner_results = ner_pipeline(email_text)

        info = {
            "sender_name": None,
            "receiver_name": None,
            "date": None,
            "company": None,
            "products": [],
            "quantity": None, # Kept for potential future use, though product extraction is now list-based
            "deadline": None,
            "request_type": None,
        }

        # Extract Persons (Sender and Receiver)
        persons = [entity['word'] for entity in ner_results if entity['entity'].startswith("B-PER")]
        if persons:
            info["sender_name"] = persons[-1]
            if len(persons) > 1:
                info["receiver_name"] = persons[0]

        # Extract Organization (Company)
        for entity in ner_results:
            if entity['entity'].startswith("B-ORG"):
                info["company"] = entity['word']
            # Extract Date using NER
            elif entity['entity'].startswith("B-DATE"):
                info["date"] = entity['word']


        # Determine Request Type based on keywords (Similar to ClassifierAgent but specific to email content)
        lower = email_text.lower()
        if re.search(r'\b(quotation|quote|rfq|price estimate|request for price|bid request)\b', lower):
            info["request_type"] = "RFQ"
        elif re.search(r'\b(inquiry|question|query)\b', lower):
            info["request_type"] = "Inquiry"
        elif re.search(r'\b(complaint|issue|problem|feedback|dissatisfied)\b', lower):
            info["request_type"] = "Complaint"
        # Added Regulation check
        elif re.search(r'\b(regulation|rule|policy|compliance)\b', lower):
             info["request_type"] = "Regulation"


        # Enhanced Product Extraction with Quantity
        # Looks for patterns like "X units of Product Y" or "Product Z (Quantity X)"
        product_patterns = re.findall(r"(\d+)\s*(?:units|pcs|pieces|items)?\s+of\s+([\w\s\-]+?)(?:\.|\n|$)", lower)
        product_patterns += re.findall(r"([\w\s\-]+?)\s+\(?(\d+)\s*(?:units|pcs|pieces|items)?\)?(?:\.|\n|$)", lower)


        for qty, product in product_patterns:
             info["products"].append({"name": product.strip(), "quantity": int(qty)})

        # Extract Deadline - More flexible pattern
        deadline_pattern = re.search(r"by\s+(.*?)(?:\.|\n|$)", email_text, re.IGNORECASE)
        if deadline_pattern:
            info["deadline"] = deadline_pattern.group(1).strip()


        # If date was not found by NER, try to extract it using a pattern
        if not info["date"]:
            date_pattern = re.search(r'\b(\w+\s+\d{1,2},?\s+\d{4})\b', email_text, re.IGNORECASE)
            if date_pattern:
                info["date"] = date_pattern.group(1)


        return info


#Usage
if __name__ == "__main__":
    logger = MemoryLogger()
    classifier = ClassifierAgent(logger, nlp)
    json_agent = JSONAgent()
    email_agent = EmailAgent()

    Example_email_text = """
    Dear Supplier,

    We would like to request a quotation for 100 units of Product A and 50 pieces of Product B.
    Please provide the price estimate by November 15, 2023.
    This is an urgent request.

    Best regards,
    John Doe
    Acme Corp
    """

    # Simulate processing an email string (instead of a file path)
    classification = classifier.classify(Example_email_text)
    print("Classification:", classification)

    if classification["format"] == "Email":
        result = email_agent.process(Example_email_text)
        print("Email Agent Output:", json.dumps(result, indent=2))

    # Its usage with JSON data
    json_data_str = """
    {
      "customer": "Jane Smith",
      "items": [
        {"name": "Widget C", "price": 10.0},
        {"name": "Gadget D", "price": 25.0}
      ],
      "date": "2023-11-10",
      "total": 35.0
    }
    """
    json_data = json.loads(json_data_str)

    classification_json = classifier.classify(json_data_str) # Classify based on the string representation
    print("\nClassification (JSON):", classification_json)

    if classification_json["format"] == "JSON":
        result_json = json_agent.process(json_data)
        print("JSON Agent Output:", json.dumps(result_json, indent=2))
