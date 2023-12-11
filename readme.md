# AI Junior Developer Test 
Welcome! You’ve stepped into the arena – now show us what you’ve got! 

## Science QA Bot

- This project aims to build a Science Question-Answering (QA) bot capable of providing accurate and relevant answers to user queries in the domain of science. The solution employs a combination of cosine similarity and a specialized pre-trained model for enhanced performance.

### Dataset
- This dataset is used for the cosine similarity to get an answer https://huggingface.co/datasets/rcherukuri14/science-qa-instructions 

### Logic
- Cosine similarity for answer selection: The primary mechanism for choosing an answer is based on cosine similarity. The system calculates the similarity between the user's question and potential answers, selecting the one with the highest similarity score.

- Fallback to "tiiuae/falcon-7b-instruct": In cases where the cosine similarity falls below a certain threshold, indicating low confidence in the answer, the question is passed to the "tiiuae/falcon-7b-instruct" model. This model excels in handling questions in the form of natural language and has been fine-tuned on the Baize, GPT4All, and GPTeacher datasets, making it suitable for the task at hand. 

>  _To be noted, falcon-7b-instruct is almost 15GB so you need big RAM_

### Other Models
- BlenderBot Model: 
Initially, the BlenderBot model was considered for the Science QA task. However, its performance did not meet the desired output

- Fine-Tuning Falcon-7B-Instruct: 
I tried to fine-tune the Falcon-7B-Instruct model to for the specific requirements of the Science QA task. Unfortunately, the results were not good, leading to the decision to use the pre-trained Falcon-7B-Instruct without additional fine-tuning.

<hr>

## Results

- Physics

![Physics](https://github.com/Mahmoud-Hesham99/scienceQA-bot-ofTask/assets/73784370/8a7eaaaa-dc31-43ed-b9cc-01d804362a28)

- Chemistry

![Chemistry](https://github.com/Mahmoud-Hesham99/scienceQA-bot-ofTask/assets/73784370/4f2afd3a-1b3e-4bc2-9a02-f2474b5bc637)


- Biology
  
![Biology](https://github.com/Mahmoud-Hesham99/scienceQA-bot-ofTask/assets/73784370/2fc97218-646d-44b2-8d09-1e84def5b1d8)


- Astronomy

![Astronomy](https://github.com/Mahmoud-Hesham99/scienceQA-bot-ofTask/assets/73784370/bfcb4298-29c7-4a48-824b-6b036069bef9)


- When faced with similar questions, one from the dataset and its paraphrased version (low cosine similarity), the system defers to falcon-7b-instruct for more confident and accurate answers.

![noted](https://github.com/Mahmoud-Hesham99/scienceQA-bot-ofTask/assets/73784370/a9935d25-6a45-492c-8107-24cfcbc04524)


> Disclaimer: This model can make mistakes sometimes.


## Mission
You're not just fiddling with code here; you're architecting the future. Your battleground? An AI app framework crying out for a brain.

Your task: Forge an 💬NLP chatbot that doesn’t just answer, but masters science-related questions.

Immerse yourself in the main.py file. Your battlefield is the execute function. Time to unleash your genius:
```python
############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = 'Hello!' <<-- Here you add the magic 
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))
```
## Ground Rules
Step up with any arsenal (read: libraries or packages) you believe in, but remember:
* 👎 External services like chatGPT are off-limits. Stand on your own.
* 👎 Plagiarism is for the weak. Forge your own path.
* 👎 A broken app equals failure. Non-negotiable.

## Deployment Options
The application can be executed in two different ways:
* locally by running the `start.sh` 
* on in a docker container using `Dockerfile` 

## Proving Your Mettle
* Submit your masterpiece on GitHub. We want the link within **1 week, not a second more**.
* Go the extra mile and include a video where you walk us through your solution, showcasing 
it in live action. 
* We want to see not just what you've created but also how you envisioned and executed it


## This Is It
We're not just evaluating a project; we're judging your potential to revolutionize our 
landscape. A half-baked app won’t cut it.

We're zeroing in on:
* 👍 Exceptional documentation.
* 👍 Code that speaks volumes.
* 👍 Inventiveness that dazzles.
* 👍 A problem-solving beast.
* 👍 Unwavering adherence to the brief
