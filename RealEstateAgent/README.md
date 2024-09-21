# Personalized Real Estate Agent
### Project Introduction
Imagine you're a talented developer at "Future Homes Realty", a forward-thinking real estate company. In an industry where personalization is key to customer satisfaction, your company wants to revolutionize how clients interact with real estate listings. The goal is to create a personalized experience for each buyer, making the property search process more engaging and tailored to individual preferences.

### The Challenge
Your task is to develop an innovative application named "HomeMatch". This application leverages large language models (LLMs) and vector databases to transform standard real estate listings into personalized narratives that resonate with potential buyers' unique preferences and needs.

### Project Instructions
#### Step 1: Setting Up the Python Application
Initialize a Python Project: Create a new Python project, setting up a virtual environment and installing necessary packages like LangChain, a suitable LLM library (e.g., OpenAI's GPT), and a vector database package compatible with Python (e.g., ChromaDB or LanceDB). If you don't wish to create your files from scratch, starter files are available in the workspace on the next page as an application skeleton.

#### Step 2: Generating Real Estate Listings
Generate real estate listings using a Large Language Model. Generate at least 10 listings This can involve creating prompts for the LLM to produce descriptions of various properties. An example of a listing might be:
```
Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
```

#### Step 3: Storing Listings in a Vector Database
Vector Database Setup: Initialize and configure ChromaDB or a similar vector database to store real estate listings.
Generating and Storing Embeddings: Convert the LLM-generated listings into suitable embeddings that capture the semantic content of each listing, and store these embeddings in the vector database.

#### Step 4: Building the User Preference Interface
Collect buyer preferences, such as the number of bedrooms, bathrooms, location, and other specific requirements from a set of questions or telling the buyer to enter their preferences in natural language. 

#### Step 5: Searching Based on Preferences
Semantic Search Implementation: Use the structured buyer preferences to perform a semantic search on the vector database, retrieving listings that most closely match the user's requirements.
Listing Retrieval Logic: Fine-tune the retrieval algorithm to ensure that the most relevant listings are selected based on the semantic closeness to the buyer’s preferences.

#### Step 6: Personalizing Listing Descriptions
LLM Augmentation: For each retrieved listing, use the LLM to augment the description, tailoring it to resonate with the buyer’s specific preferences. This involves subtly emphasizing aspects of the property that align with what the buyer is looking for.
Maintaining Factual Integrity: Ensure that the augmentation process enhances the appeal of the listing without altering factual information.

#### Step 7: Deliverables and Testing

Test your "HomeMatch" application and make sure it meets all of the requirements in the rubric(opens in a new tab). Your project code will be run when it's assessed. Enter different "buyer preferences" and ensure it works.
Jupyter Notebook/Python Program: Compile the application code in a Jupyter notebook or a standalone Python program. Ensure the code is well-commented and logically structured.
Example Outputs: Include example outputs showcasing how user preferences are processed and how the application generates personalized listing descriptions. You can include these in comments in your application or in a Jupyter notebook that's saved with outputs.

#### Step 8: Project Submission

Generated Listings: Include a file that contains your synthetically generated real estate listings. Name this file "listings"
Project Documentation: Include a readme file or an accompanying document explaining the functionality, how to run the code, and any prerequisites or dependencies.
Code Submission: Submit the Jupyter Notebook or Python program on the "Project Submission Page" that follows the workspace page.
