from langchain.chains.query_constructor.schema import AttributeInfo

metadata_template = """
You are tasked with extracting detailed metadata information from the content of a document. Follow these detailed guidelines to ensure the metadata is comprehensive and accurately reflects the document's content.

**Guidelines**:

1. **Document Type**:
    - Identify the type of document (e.g., research paper, article, report).
    - Examples: "Research Paper", "Article", "Report", "Forschungsbericht", "Artikel", "Bericht"

2. **Mentions**:
    - Extract the main names like persons and companies mentioned in the document.
    - Examples: "John Doe", "Acme Corporation", "United Nations", "Johann Schmidt", "Siemens AG", "Vereinte Nationen"

3. **Keywords**:
    - Identify relevant keywords central to the document's topic.
    - Examples: "Machine Learning", "Climate Change", "Economic Policy", "Maschinelles Lernen", "Klimawandel", "Wirtschaftspolitik"

4. **About**:
    - Provide a brief description of the document's purpose and main arguments/findings.
    - Examples: "This research paper explores the impact of AI on healthcare, focusing on predictive analytics and patient outcomes.", "Dieses Forschungspapier untersucht die Auswirkungen von KI auf das Gesundheitswesen, mit einem Fokus auf prädiktive Analysen und Patientenergebnisse."

5. **Questions**:
    - List questions the document can answer.
    - Examples: "What are the benefits of renewable energy?", "How does blockchain technology work?", "Welche Vorteile bietet erneuerbare Energie?", "Wie funktioniert die Blockchain-Technologie?"

6. **Entities**:
    - Identify the main entities (people, places, organizations) mentioned.
    - Examples: "Albert Einstein", "New York City", "World Health Organization", "Albert Einstein", "New York City", "Weltgesundheitsorganisation"

7. **Summaries**:
    - Provide summaries of different sections or key points.
    - Examples: "Introduction: Overview of AI in healthcare", "Methodology: Data collection and analysis techniques", "Conclusion: Implications of findings for future research", "Einleitung: Überblick über KI im Gesundheitswesen", "Methodik: Datenerfassungs- und Analysetechniken", "Fazit: Auswirkungen der Ergebnisse auf zukünftige Forschung"

8. **Authors**:
    - List the document's authors.
    - Examples: "Jane Smith", "John Doe", "Alice Johnson", "Hans Müller", "Peter Schmid", "Anna Meier"

9. **Source**:
    - Specify the source or location where the document can be found.
    - Examples: "https://example.com/research-paper", "Library of Congress", "Journal of Medical Research", "https://beispiel.de/forschungspapier", "Bibliothek des Kongresses", "Zeitschrift für medizinische Forschung"

10. **Language**:
    - Indicate the language(s) the document is written in.
    - Examples: "English", "German", "Spanish", "Englisch", "Deutsch", "Spanisch"

11. **Audience**:
    - Describe the intended audience for the document.
    - Examples: "Healthcare professionals", "University students", "Policy makers", "Gesundheitsfachkräfte", "Universitätsstudenten", "Politische Entscheidungsträger"

**Context**:
{context}

**Task**:
Extract and provide the following metadata from the document's content based on the above guidelines. Ensure that extracted information is in the original language of the document.

**Output Format**:
Return the metadata in the following structured format with no filter text or extra explanation, only give the extracted metadata:
```json
{{
"document_type": "Type of document",
"mentions": ["Main names mentioned"],
"keywords": ["Relevant keywords"],
"about": "Brief description",
"questions": ["Questions the document can answer"],
"entities": ["Main entities mentioned"],
"summaries": ["Summaries of key sections"],
"authors": ["List of authors"],
"source": "Source or location",
"language": "Document language",
"audience": "Intended audience"
}}
```
"""

naming_template = """
You are tasked with generating appropriate and consistent names for documents based on their content. Follow these detailed guidelines to ensure the names are informative, unique, and easy to manage:

1. **Think about your files**:
    - Identify the group of files your naming convention will cover.
    - Check for established file naming conventions in your discipline or group.

2. **Identify metadata**:
    - Include important information to easily locate a specific file.
    - Consider including a combination of the following:
        - Experiment conditions
        - Type of data
        - Researcher name/initials
        - Lab name/location
        - Project or experiment name or acronym
        - Experiment number or sample ID (use leading zeros for clarity)

3. **Abbreviate or encode metadata**:
    - Standardize categories and/or replace them with 2- or 3-letter codes.
    - Document any codes used.

4. **Think about how you will search for your files**:
    - Decide what metadata should appear at the beginning.
    - Use default ordering: alphabetically, numerically, or chronologically.

5. **Deliberately separate metadata elements**:
    - Avoid spaces or special characters in file names.
    - Use dashes (-), underscores (_), or capitalize the first letter of each word.

**Example Naming Convention**:
    - Format: [Type]_[Project]_[SampleID].[ext]
    - Example: FinancialReport_ProjectX_001.pdf

**Context**:
{context}

**Extracted Metadata**:
The extracted metadata contains important information such as keywords, entities, mentions, summaries, and other details that are useful for naming the document. This metadata helps in creating a name that is both descriptive and unique.

{metadata}

**Task**:
Generate a new, unique name for this document based on its content and the provided metadata. The new name should be formal, detailed, and distinctive to avoid confusion with other documents. Ensure the name is concise yet informative, highlighting significant details like names, firms, companies, etc. that capture the essence and purpose of the document. Be specific.

**Output Format**:
Provide only the new name in the following format with no filter or extra explanation, give only the new name: [Type]_[Name]_[YearRange]

**Question**: {question}
"""

metadata_field_info = [
    AttributeInfo(
        name="document_type",
        description="Type of document (e.g., research paper, article, report).",
        type="string",
    ),
    AttributeInfo(
        name="mentions",
        description="Main names (people, companies) mentioned in the document.",
        type="string",
    ),
    AttributeInfo(
        name="keywords",
        description="Keywords central to the document's topic.",
        type="string",
    ),
    AttributeInfo(
        name="about",
        description="Brief description of the document's purpose and findings.",
        type="string",
    ),
    AttributeInfo(
        name="questions",
        description="Questions the document can answer.",
        type="string",
    ),
    AttributeInfo(
        name="entities",
        description="Main entities (people, places, organizations) mentioned.",
        type="string",
    ),
    AttributeInfo(
        name="authors",
        description="The authors of the document.",
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="Source or location of the document.",
        type="string",
    ),
    AttributeInfo(
        name="language",
        description="Language(s) the document is written in.",
        type="string",
    ),
    AttributeInfo(
        name="audience",
        description="Intended audience of the document.",
        type="string",
    ),
    AttributeInfo(
        name="given_document_name",
        description="The document's given name.",
        type="string",
    ),
    AttributeInfo(
        name="document_id",
        description="Unique identifier for the document (UUID).",
        type="string",
    ),
    AttributeInfo(
        name="file_directory",
        description="Path of the document's directory.",
        type="string",
    ),
    AttributeInfo(
        name="original_file_name",
        description="The original name of the file.",
        type="string",
    ),
    AttributeInfo(
        name="file_creation_date",
        description="Creation date of the document (ISO 8601).",
        type="string",
    ),
    AttributeInfo(
        name="file_modification_date",
        description="Last modification date (ISO 8601).",
        type="string",
    ),
    AttributeInfo(
        name="metadata_creation_date",
        description="Metadata creation date (ISO 8601).",
        type="string",
    ),
]

document_content_description = "Documents from the user, official documents, research papers, reports, articles, etc."

