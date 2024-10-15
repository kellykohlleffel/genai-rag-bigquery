# :wine_glass: Create a wine country travel assistant with Fivetran, Google BigQuery, and Google Vertex AI
## Scripts and code for the Fivetran + Google BigQuery RAG-based, Gen AI Hands on Lab (60 minutes)

This repo provides the high-level steps to create a RAG-based, Gen AI travel assistant using Fivetran and Google BigQuery (detailed instructions are in the lab guide provided by your lab instructor). The required transformation scripts and the required Gradio code are both included. This repo is the "easy button" to copy/paste the transformations and the code. If you have any issues with copy/paste, you can download the code [here](https://github.com/kellykohlleffel/genai-rag-bigquery/archive/refs/heads/main.zip).

> ## IMPORTANT - STEP 1: This lab requires you to sign in to Google Cloud CE Qwiklabs Account and a Fivetran Account
* These will both be provided by the lab instructor
* Sign in to both accounts
* The CE Qwiklabs console will provide you with a username, password, and project id
* The Fivetran account is Fivetran_HoL

## STEP 2: Setup your Fivetran Destination: Google BigQuery
* Detailed in the lab guide

## STEP 3: Create a Fivetran connector to BigQuery

* **Source**: Google Cloud PostgreSQL (G1 instance)
* **Fivetran Destination**: Use the new destination you just created
* **Schema name**: yourlastname 
* **Host**: 34.94.122.157 **(see the lab guide for credentials)**
```
34.94.122.157
```
* **Schema**: agriculture
* **Table**: california_wine_country_visits

## STEP 4: View the new dataset in Google BigQuery

* **Google Account**: **use your ce quiwlabs project**
* **Schema**: yourschemaprefix_agriculture 
* **Table**: california_wine_country_visits
* Click on **Preview** to take a look

## STEP 5: Configure Required Database Objects in BigQuery

* Before our Gradio application can interact with the Gemini Foundation Model for responding to our travel inquiries, there are a few configuration steps that we need to take to allow the code to interact with Gemini.
* We’ll use a Python notebook for our configuration. BigQuery Studio now supports Python Notebooks, which will be ideal for our use case because they support the authoring and execution of code.  The code can be linux commands, python code or BigQuery SQL commands, which is perfect for our use case.  
* Open a new Python notebook name it **BigQuery Config** in the BigQuery Studio Console by clicking the down arrow on the Query Tool Bar and selecting “Python Notebook” and "Enable All" APIs
* Copy and paste the following code blocks into the BigQuery Notebook

> ### NOTE - Because this is the first code block that is being executed, it may take up to 60 seconds for the runtime environment to be allocated before the code is executed.  This is a one-time initialization and won’t be required in subsequent steps.

### **BigQuery Config - Code Block 1**
```
#@title BigQuery Config - Code Block 1: Set Project Id Environment Variable
#Replace <Your Project ID> with your own Project Id

%env PROJECT=<Your Project Id>
```

**The output should be similar to this:**
<br/> env: PROJECT=qwiklabs-gcp-02-1c0d9f2f41cb

### **BigQuery Config - Code Block 2**

A Google Cloud Resource Connection is a connection to authorize access to other Google Cloud resources from within BigQuery.  For our lab, we will be using this connection in future steps to access Vertex AI.

```
#@title BigQuery Config - Code Block 2: Create Cloud Resource Connection

!bq mk \
  --connection \
  --location=US \
  --project_id=$PROJECT \
  --connection_type=CLOUD_RESOURCE travel_asst_conn
```

**The output should be similar to this:**
<br/> Connection 203414464784.us.travel_asst_conn successfully created

### **BigQuery Config - Code Block 3**

When the Cloud Resource Connection was created in the previous step, a Service Account associated with the connection was also created.  A Service Account is a special type of account that is used by an application or a workload, rather than an actual human user. When connecting to external services, the Service Account must have been granted sufficient privileges to interact with the external service.  In order to grant the necessary permissions to the Service Account, we must first retrieve the Service Account ID associated with the Service Account, which is a unique email address.

To access the details for the connection that you just created, create a new Code Block and paste in the following code. Execute the code block.

```
#@title BigQuery Config - Code Block 3: Retrieve Details for the new Cloud Resource Connection

!bq show \
  --format=json --connection \
  $PROJECT.us.travel_asst_conn
```

> ### NOTE - For the next step, you will need the value of the serviceAccountId field, which is the unique email address associated with the Service Account.  If you scroll the output from the previous command all the way to the right, you’ll see the email address. You can easily copy the email address by right-clicking on it and selecting “Copy Email Address”.

### **BigQuery Config - Code Block 4**

Using the Service Account Email from the previous step, assign the aiplatform.user role to the Service Account.  This allows the Service Account to access Vertex AI services.

```
#@title BigQuery Config - Code Block 4: Give Service Account Permissions for Vertex AI
# Replace <Service Account Email From The Previous Step> with your Service Account Id


!gcloud projects add-iam-policy-binding $PROJECT \
  --member='serviceAccount:<Service Account Email From The Previous Step>' \
  --role='roles/aiplatform.user' \
  --condition=None
```
**The output should be a list of all bindings that were updated and the first line of output should be similar to this:**
<br/> Updated IAM policy for project [qwiklabs-gcp-02-919b7b91e253]

### **BigQuery Config - Code Block 5**

Create a new dataset in your BigQuery project.  We will use this dataset in future steps for storing new database objects that we will be creating.

```
#@title BigQuery Config - Code Block 5: Create BigQuery Dataset for storing new database objects

!bq --location=US mk --dataset \
--default_table_expiration=0 \
$PROJECT:travel_assistant_ds
```
**The output should be similar to this:**
<br/> Dataset 'qwiklabs-gcp-02-919b7b91e253:travel_assistant_ds' successfully created

### **BigQuery Config - Code Block 6**

Create another new model in the dataset that you created previously. We’ll be using a special type of CREATE MODEL syntax that lets you register a Vertex AI endpoint as a REMOTE MODEL so that you can call it directly from BigQuery. We’ll use the connection that we created earlier, which we already granted permissions to interact with Vertex AI, and we’ll use a Text Embedding Model as our endpoint.

We will be using this model for creating the vector embeddings for our data, which will be used by our Travel Assistant for Semantic Search.

```
#@title Code Block 6: Create Remote Model for interacting with a Text Embedding Model

%%bigquery
CREATE OR REPLACE MODEL travel_assistant_ds.travel_asst_embed_model
REMOTE WITH CONNECTION `us.travel_asst_conn`
OPTIONS (ENDPOINT = 'text-embedding-004');
```
**The output should be similar to this:**
<br/> Job ID 483a4f46-07e7-4ba9-9745-a9fa7a8b7bc4 successfully executed: 100%

### **BigQuery Config - Code Block 7**

Now, we will create a new table in the dataset you created in one of the earlier steps.  This table will contain the vector embeddings of our data.  Before creating the embeddings, the columns in the table are concatenated to create a textual description of the winery, which is a more appropriate format for LLMs to work with.

> ### NOTE - For this step, you will need to replace <your Fivetran dataset name> with the name of the dataset that you specified. Because this query concatenates the data in each row into a single string per row and then converts that data into its vector representation for all 700+ rows in our table, this query can take about a minute to run.

```
#@title BigQuery Config - Code Block 7: Append Winery Details and Create Search Embeddings
#Replace <Your Fivetran Dataset Name> with the name of the dataset that you specified
#when setting up your BigQuery destination in Fivetran
#For Example: lastname_agriculture
%%bigquery

#Change the value of <Your Fivetran Dataset Name> to the name of the dataset that contains the table
#california_wine_country_visits. The dataset name should be <Your Last Name>_agriculture
#For Example:
# SET @@dataset_id = 'kohlleffel_agriculture'

SET @@dataset_id = '<Your Fivetran Dataset Name>';

#Concatenate the columns in the california_wine_country_visits table into a more LLM-friendly format
#and store them in a temporary table
CREATE TEMPORARY TABLE winery_text AS
SELECT CONCAT('This winery name is ', IFNULL(WINERY_OR_VINEYARD, ' Name is not known')
      , '. California wine region: ', IFNULL(CA_WINE_REGION, 'unknown'), ''
      , ' The AVA Appellation is the ', IFNULL(AVA_APPELLATION_SUB_APPELLATION, 'unknown'), '.'
      , ' The website associated with the winery is ', IFNULL(WEBSITE, 'unknown'), '.'
      , ' Price Range: ', IFNULL(PRICE_RANGE, 'unknown'), '.'
      , ' Tasting Room Hours: ', IFNULL(TASTING_ROOM_HOURS, 'unknown'), '.'
      , ' Are Reservations Required or Not: ', IFNULL(RESERVATION_REQUIRED, 'unknown'), '.'
      , ' Winery Description: ', IFNULL(WINERY_DESCRIPTION, 'unknown'), ''
      , ' The Primary Varietals this winery offers: ', IFNULL(PRIMARY_VARIETALS, 'unknown'), '.'
      , ' Thoughts on the Tasting Room Experience: ', IFNULL(TASTING_ROOM_EXPERIENCE, 'unknown'), '.'
      , ' Amenities: ', IFNULL(AMENITIES, 'unknown'), '.'
      , ' Awards and Accolades: ', IFNULL(AWARDS_AND_ACCOLADES, 'unknown'), '.'
      , ' Distance Travel Time considerations: ', IFNULL(DISTANCE_AND_TRAVEL_TIME, 'unknown'), '.'
      , ' User Rating: ', IFNULL(USER_RATING, 'unknown'), '.'
      , ' The Secondary Varietals for this winery: ', IFNULL(SECONDARY_VARIETALS, 'unknown'), '.'
      , ' Wine Styles: ', IFNULL(WINE_STYLES, 'unknown'), '.'
      , ' Events and Activities: ', IFNULL(EVENTS_AND_ACTIVITIES, 'unknown'), '.'
      , ' Sustainability Practices: ', IFNULL(SUSTAINABILITY_PRACTICES, 'unknown'), '.'
      , ' Social Media Channels: ', IFNULL(SOCIAL_MEDIA, 'unknown'), ''
      , ' Address: ', IFNULL(ADDRESS, 'unknown'), ''
      , ' City: ', IFNULL(CITY, 'unknown'), ''
      , ' State: ', IFNULL(STATE, 'unknown'), ''
      , ' ZIP: ', IFNULL(ZIP, 'unknown'), ''
      , ' Phone: ', IFNULL(PHONE, 'unknown'), ''
      , ' Winemaker: ', IFNULL(WINEMAKER, 'unknown'), ''
      , ' Did Kelly Kohlleffel recommend this winery?: ', IFNULL(KELLY_KOHLLEFFEL_RECOMMENDED, 'unknown'), ''

     ) AS winery_information
 FROM
     california_wine_country_visits;

#Create text embeddings from our concatenated wine country data and
#store it in a new BigQuery table
CREATE OR REPLACE TABLE travel_assistant_ds.california_wine_country_embeddings AS
SELECT *
FROM ML.GENERATE_TEXT_EMBEDDING(
     MODEL `travel_assistant_ds.travel_asst_embed_model`,
     ( select winery_information as content
       from winery_text
     ),
     STRUCT(TRUE AS flatten_json_output)
);
```
**The output should be similar to this:**
<br/> Job ID 68295311-d637-47ab-a5af-1e5960b45948 successfully executed: 100%

### **BigQuery Config - Code Block 8**

Now, let's check out the output of those LLM transformations.

```
#@title BigQuery Config - Code Block 8: Run this select statement to generate output of the LLM transformations for a single record
%%bigquery

SELECT content, text_embedding
FROM `travel_assistant_ds.california_wine_country_embeddings`
WHERE content LIKE 'This winery name is Kohlleffel Vineyards%'
LIMIT 1;
```

> ### Click on the graph icon and scroll down.

**The output should be similar to this:**

**Content:**

This winery name is Kohlleffel Vineyards. California wine region: This winery is
located in California's North Coast wine region, spanning Sonoma, Napa,
Mendocino, and Lake counties. The North Coast is a celebrated viticultural
paradise. Characterized by its diverse terroirs, from the coastal breezes to the
inland hills, this region yields a wide range of grape varietals, including its
renowned Cabernet Sauvignon and Sauvignon Blanc and elegant coastal Pinot Noirs
and Chardonnays, embodying the essence of California winemaking. The AVA
Appellation is the Sonoma Coast. The website associated with the winery is
https://kohlleffelvineyards.com. Price Range: medium range. Tasting Room Hours:
24 x 7...

**Text_Embedding:**

0.03821619  0.05215483 -0.03741358  0.00874872  0.03864389  0.0241658
0.03700958 -0.00570941 -0.04640496  0.02133269  0.01968173  0.05466348
-0.00195844  0.01548036  0.07116827 -0.02989442  0.01493446  0.06761404
-0.04398699 -0.02521998 -0.02368315 -0.0490068   0.0152873  -0.06923058
0.01284278 -0.07185011  0.00750731 -0.00983331  0.01127757 -0.05489314
0.07358457  0.00205145 -0.01487004 -0.01827036 -0.01846506  0.01059951
0.06516002 -0.00390975  0.03528809 -0.04468951  0.00236333  0.0712541
-0.06551971  0.05187486 -0.017887   -0.05857743 -0.0337937   0.05416118

## STEP 6: Build the Gradio Application (the travel assistant)

* We’ll be building our application using Gradio, which is an open-source Python package that allows you to quickly build a User Interface that can be embedded in a Python notebook or presented as a web page.
* We’ll use a new Python notebook for our Gradio application.   
* Open a new Python notebook name it "Wine Country App" in the BigQuery Studio Console by clicking the down arrow on the Query Tool Bar and selecting “Python Notebook” and "Enable All" APIs
* Copy and paste the following code blocks into the Gradio Notebook

### **Wine Country App - Code Block 1**

This Code Block will install the Gradio software and the IPython interpreter.  

```
#@title Wine Coutnry App - Code Block 1: Dependency installation

#Install the Open source Python Framework that will be used to create the
#front end for our application
!pip install gradio

import IPython
```

**The output should be similar to this:**
<br/> Successfully installed …

### **Wine Country App - Code Block 2**

This code will restart the environment so that future code blocks will be able to access the software that we installed in the previous code block.  

**When the environment is restarted, you will receive an error message similar to this.:**
<br/> Your session crashed for an unknown reason. View runtime logs.
<br/> Note - you can safely close this error message and continue.

```
#@title Wine Country App - Code Block 2: Restart the application

#Automatically restart kernel after installs so that your environment can access the new packages
#You will get an error message saying that your application has crashed.  It is safe to ignore this.

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### **Wine Country App - Code Block 3**

This code block loads all of the Python Modules that will be required by our application.  

```
#@title Wine Country App - Code Block 3: Import modules

#Import all of the Python modules that will be used in our application
import time
from datetime import date
import json
import re

#BigQuery Module
from google.cloud import bigquery

# Vertex AI
import vertexai

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from vertexai import generative_models

#Iphython for prettier printing
from IPython.display import display, Markdown, Latex

```

### **Wine Country App - Code Block 4**

> ## NOTE - This is a special type of Code Block which presents user customizable parameters that are used by our code.  We need to properly populate one of these parameters before executing the Code Block.

> **Update the PROJECT ID:** Your Project ID

```
#@title Wine Country App - Code Block 4: Parameters
#Configurable parameters that can be set specifically for your application

PROJECT = "<Replace With Your Project ID>"  # @param {type:"string"}
LOCATION = "us-central1" # @param {type:"string"}

MODEL = "gemini-1.5-pro-001" #@param ["gemini-1.5-pro-001","gemini-1.5-flash-001"]
TEMPERATURE = 1 #@param {type:"number"}
TOP_P = 1 #@param {type:"number"}
MAX_OUTPUT_TOKENS = 8192 #@param {type:"number"}

BQ_TABLE = "travel_assistant_ds.california_wine_country_embeddings" #@param {type:"string"}
BQ_MODEL = "travel_assistant_ds.travel_asst_embed_model"#@param {type:"string"}
BQ_NUMBER_OF_RESULTS =  50 #@param {type:"number"}
#Maximium distance between vectors
THRESHOLD = 0.9  #@param {type:"number"}

#Set initialization paramters based on the settings that you specified
aiplatform.init(project=PROJECT, location=LOCATION)
model = generative_models.GenerativeModel(MODEL)

#Authenticate using your Google Cloud User
from google.colab import auth as google_auth
google_auth.authenticate_user()

```

### **Wine Country App - Code Block 5**

This Code Block defines all of the utility functions used for executing queries interactively and via the user interface.  The interactions between the application, BigQuery, and Gemini are defined here.

```
#@title Wine Country App - Code Block 5: Functions

#Creates embeddings via the embedding model, executes the query and returns
#the results
def query_bigquery(query):
 bq_template = f"""
 SELECT query.query,distance, base.content
     FROM VECTOR_SEARCH(
     TABLE `{BQ_TABLE}`, 'text_embedding',
     (SELECT text_embedding, content AS query
     FROM ML.GENERATE_TEXT_EMBEDDING(
       MODEL `{BQ_MODEL}`,
       (SELECT \"{query}\" as content))
     ), top_k => {BQ_NUMBER_OF_RESULTS},distance_type => 'EUCLIDEAN')
     WHERE distance < {THRESHOLD}
     ORDER BY distance ASC
 """
 # Initialization using default credentials
 client = bigquery.Client(project = PROJECT)  # This now uses your environment variable
 try:
     query_job = client.query(bq_template)
     results = query_job.result()
     return [dict(row) for row in results]
 except Exception as e:
     print(e)

#Query BigQuery and populate the context
#This is used when running queries interactively rather
#than through the User Interface
def get_context(query):
 bq_context = query_bigquery(query)
 if len(bq_context) < 1:
   context = ""
 else:
   counter = 1
   context = "Use the following wineries to formulate your answer: \n"
   for result in bq_context:
     context = context+f"Winery #{counter}:\n{result['content']}\n\n"
     counter = counter+1
 return(context)

#Query BigQuery and populate the context.
#This is used when running through the User Interface
#when we need to recognize which flags have been set.
def get_context_with_params(query,use_bq):
 if use_bq == True:
   bq_context = query_bigquery(query)
   if len(bq_context) < 1:
     context = ""
   else:
     counter = 1
     context = "Use the following wineries to formulate your answer: \n"
     for result in bq_context:
       context = context+f"Winery #{counter}:\n{result['content']}\n\n"
       counter = counter+1
 else:
   context = ""
 return(context)

#Take the context and the history and send it to Gemini for inference
def process_llm(model, prompt):
 try:
   responses = model.generate_content(
   [prompt],
   generation_config={
   "max_output_tokens": MAX_OUTPUT_TOKENS,
   "temperature": TEMPERATURE,
   "top_p": TOP_P},
   stream=False,
   )
   if(responses.candidates[0].finish_reason.value == 1):
     #print("\n\n____\nResponses:\n"+str(responses)+"\n\n____\n\n")
     return(responses.candidates[0].content.parts[0].text)
   else:
     return(f"Content has been blocked for {responses.candidates[0].finish_reason.name} reasons.")
 except Exception as e:
   print(e)

#Take the context and the history and send it to Gemini for inference.
#Set the parameter indicating the Google Search should be used for grounding.
def process_llm_grounding(model, prompt):
 try:
   tool = generative_models.Tool.from_google_search_retrieval(
               generative_models.grounding.GoogleSearchRetrieval())

   responses = model.generate_content(
               [prompt], tools=[tool],
               generation_config={
   "max_output_tokens": MAX_OUTPUT_TOKENS,
   "temperature": TEMPERATURE,
   "top_p": TOP_P},
   stream=False,
   )

   if(responses.candidates[0].finish_reason.value == 1):
     #print("\n\n____\nResponses:\n"+str(responses)+"\n\n____\n\n")
     return(responses.candidates[0].content.parts[0].text)
   else:
     return(f"Content has been blocked for {responses.candidates[0].finish_reason.name} reasons.")
 except Exception as e:
   print(e)

#Orchestrate the query based on the flags that have been set.
#This is used when running queries interactively rather
#than through the User Interface

def ask_question(prompt, model, history):
 #Execute a BQ query in case there is no history
 if history == "":
   context = get_context(prompt)
   if context == "":
     formatted_prompt = template.format(history="This is the first turn", context=context, prompt=prompt)
     answer = process_llm(model, formatted_prompt)
     return(answer, "")
   else:
     formatted_prompt = template.format(history="This is the first turn", context=context, prompt=prompt)
     answer = process_llm(model, formatted_prompt)
     history = f"{formatted_prompt}\n\n{answer}\n"
     return(answer, history)
 else:
   formatted_prompt = template.format(history=history, context="", prompt=prompt)
   answer = process_llm(model, formatted_prompt)
   history = f"{formatted_prompt}\n\n{answer}\n"
   return(answer, history)

#Orchestrate the query based on the flags that have been set.
#This is used when running through the User Interface
#when we need to recognize which flags have been set.
def ask_question_with_params(prompt, model, history,use_bq, use_gs):
 #Execute a BQ query in case there is no history
 if history == "":
   context = get_context_with_params(prompt,use_bq)
   if context == "":
     formatted_prompt = template.format(history="This is the first turn", context=context, prompt=prompt)
     if use_gs == True:
       answer = process_llm_grounding(model, formatted_prompt)
     else:
       answer = process_llm(model, formatted_prompt)
     return(answer, "")
   else:
     formatted_prompt = template.format(history="This is the first turn", context=context, prompt=prompt)
     answer = process_llm(model, formatted_prompt)
     history = f"{formatted_prompt}\n\n{answer}\n"
     return(answer, history)
 else:
   formatted_prompt = template.format(history=history, context="", prompt=prompt)
   answer = process_llm(model, formatted_prompt)
   history = f"{formatted_prompt}\n\n{answer}\n"
   return(answer, history)

```

### **Wine Country App - Code Block 6**

This Code Block sets our system prompt, which explains to the LLM what we are trying to accomplish and some guidelines to use when formulating responses.  Send this prompt to the LLM to initialize it, which must be done before sending the first query.

```
#@title Wine Country App - Code Block 6: Model Initialization

#Initialize the model with the system instructions
#The model needs to be initialized prior to being used

system_prompt = """
You are a friendly California wine tourism expert helping visitors in the California wine country who want an incredible visit and tasting experience.

It is very important that you NEVER make up information. Always leverage the content provided by the database search, if the answer is not in the context, don't make it up, just say I don't know.

Always optimize for providing and answer, request more details only if the user requests different information.
"""

template = """
This is what we have been talking about:
<history>
{history}
</history>

You have the following context:

<context>
{context}
<\context>

This is what the user asked:
{prompt}

Answer:
"""

model = vertexai.generative_models.GenerativeModel(MODEL, system_instruction=[system_prompt])
```

### **Wine Country App - Code Block 7**

This basic query is sent to Gemini and is simply used to verify that everything up to this point has been configured correctly.  If everything works as expected, you get a friendly response from Gemini.  It should be similar to this, but may not be exactly the same.

<br/> Hi! 👋  How can I help you today? 😊 

```
#@title Wine Country App - Code Block 7: Test Model Interaction

query = "hi"
history = ""
answer, history = ask_question(query,model, history)
print(answer)
```

### **Wine Country App - Code Block 8**

Unlike the other code blocks that we have run, this block continues to run for as long as you want to interact with the Travel Assistant.  Once the User Interface starts, the Play Button changes to a Stop button, which you can press whenever you are done interacting with the Travel Assistant.

```
#@title Wine Country App - Code Block 8: Gradio App
import gradio as gr

def respond(message, chat_history, history,use_bq, use_gs):
 try:

   answer, history = ask_question_with_params(message, model,history,use_bq, use_gs)
 except Exception as e:
   print(e)
   answer = "I'm sorry, something went wrong."
 bot_message = answer
 chat_history.append((message, bot_message))
 return "", chat_history, history

with gr.Blocks() as demo:
   history = ""
   gr.Markdown(
       """
   # Wine Country Concierge
   Chat with your data with Google Gemini
   """
   )
   chatbot = gr.Chatbot(show_label=False)
   msg = gr.Textbox(scale=2,
           show_label=False,
           placeholder="Enter your question",)
   state = gr.State(value="")
   sub_btn = gr.Button(value="Submit")
   clear = gr.ClearButton([msg, chatbot,state])
   with gr.Row():
      use_bq = gr.Checkbox(label="Use RAG and the Fivetran dataset moved into BigQuery", value=True)
      use_gs = gr.Checkbox(label="Use Google Search for Grounding")


   msg.submit(respond, [msg, chatbot, state, use_bq, use_gs], [msg, chatbot, state])
   sub_btn.click(respond, inputs=[msg, chatbot, state, use_bq, use_gs], outputs=[msg, chatbot, state])

demo.launch(debug=True)

```

> ## Click on the *gradio.live URL* in the output from the code block above to start your Wine Country Visit Assistant in a new tab.

## Step 7: Have some fun checking out the travel assistant features and creating prompts for unique visits using RAG
* Test the Gradio application with your own prompts or check out the sample prompts in the lab guide
* Control records used for testing RAG include the following and these only exist in the dataset moved with Fivetran from PostgreSQL to BigQuery:
  * Kohlleffel Vineyards
  * Millman Estate
  * Hrncir Family Cellars
  * Kai Lee Family Cellars
  * Tony Kelly Pamont Vineyards
  * Sandell Estates

### Sample Prompts

```
Tell me about Kohlleffel Vineyards
```

```
Tell me about Sandell Estates
```

```
Tell me everything you know about Kohlleffel Vineyards and Sandell Estates in bulletized format and highlight anything unusual about each winery or what I can expect when I visit.
```

```
Plan an incredible 2 day trip to california wine country. I want to visit 2 wineries on the Sonoma Coast on day 1 including Kohlleffel Vineyards. Be sure to provide detail about Kohlleffel Vineyards and the other winery. On day 2, I want to visit 3 wineries in Napa Valley and be sure to include Sandell Estates as one of the wineries. Provide all detail on Sandell Estates. Also, give me hotel recommendations, restaurant recommendations, other activity recommendations, and clothing recommendations for this time of year. I want to impress the friend that I am traveling with so provide a catchy name for this trip. I will also need you to estimate the total cost of this trip for me with detail on each activity.
```

### Fivetran + Google BigQuery California Wine Country Visit Assistant

![Travel Assistant Screenshot](images/Fivetran%20+%20BigQuery%20+%20Vertex%20+%20Gradio%20-%20Travel%20Assistant.png)

-----
