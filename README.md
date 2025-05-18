# End-to-end-Healthcare-Chatbot-Generative-AI


# Steps to run
## STEP 01:

Clone the repository

```bash
Project repo: https://github.com/
```
## STEP 02- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


## STEP 03- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & Google credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

```bash
In web browser open localhost
```


### Techstack Used:

- Python
- LangChain
- Flask
- Gemini
- Pinecone

<br>

# AWS-CICD-Deployment-with-Github-Actions
	Description about the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

## Steps

### 1. Login to AWS console.

### 2. Create IAM user for deployment
	Policy:

	1. AmazonEC2ContainerRegistryFullAccess
	2. AmazonEC2FullAccess

	
### 3. Create ECR repo to store/save docker image
	- ECR: Elastic Container registry to save your docker image in aws
    - Save the URI(example): 970547337635.dkr.ecr.ap-south-1.amazonaws.com/healthcare-chatbot

	
### 4. Create EC2 machine (Ubuntu)
	- EC2: It is virtual machine

### 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    In Github repo goto: setting>actions>runner>new self hosted runner> choose os> then run command one by one in EC2


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - GOOGLE_API_KEY
