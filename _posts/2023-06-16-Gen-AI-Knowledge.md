## General Description of the SS project
SS is a web-based solution, which helps organizations to upskill employees on training materials, simulate role experience. It runs on GCP leveraging LLM capability by prompt engineering,
fine tuning using documents and materials provided by organizations.

### Value

* Automate the training and evaluation process using simulated real-world experience
* Reducing Training costs
* Increase speed to traind and upskill
* Increase customer satisfaction scores by identifying areas of concerns

## Use Case
Trainees read training material and viewing simulation, and then they select the persona/issue they want to simulate. Based on the responses given from trainees, real time scoring function 
will assess trainees' responses based on accuracy, professionalism, compassion, and clarity.

## Technology Stack

### GCP 
The whole application is based on GCP, in which we will deploy everything on the cloud platform. The followings are the services that we use on GCP

* Firestore
* Cloud Run
* App Engine
* Vertex AI
### Firestore
This is the Nosql database on GCP. The google cloud provides an user friendly libray to allow developer to perform CRUD related business.


### GCP Deployment
Docker and Cloud run. The requirement is contained in a docker file. The overall service is hosted first on cloud run as a 
micro-service and then was shifted to app engine. The app engine was hosted as a server providing base url.

### Vertext AI


### Langchain
The whole pipeline was operated based on the langchain libray using the Vertax ChatModel and memory component. The former is used to instantiate the chatmodel
and the later is used to maintain the conversation history of the chatmodel.

### LLM Prompt
Writing prompt to instruct the llm to act as a customer. It should follow the principles such as precise and explicit.

### Prompt Build
We use the prompt template which allows us to dynamic configure the context. This functionality is essential to scale up the solution.
We are not fixing the application domain to one client. By providing the the training context as one variable to the prompt template 
we can easily shift to other domains. 

The idea is that we are aiming to provide fully automated solution which can substitute human labor optimally.

### DataModel Definition

1. Topics: 
2. personas
3. simulation_config
4. user
5. conversation
















