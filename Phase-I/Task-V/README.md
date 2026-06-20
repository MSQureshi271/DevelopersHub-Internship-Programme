# Empathetic Chatbot using Fine-Tuned DistilGPT2

## Project Overview

This project focuses on building a basic empathetic chatbot that can generate supportive and emotionally aware responses for users dealing with stress, anxiety, sadness, loneliness, and general emotional wellness concerns.

The chatbot was developed by fine-tuning an existing Hugging Face language model, **DistilGPT2**, on the **EmpatheticDialogues dataset by Facebook AI**. The model was trained to respond in a gentle, supportive, and empathetic tone using real human dialogue examples.

## Google Colab Workspace

This task was completed in **Google Colab**.

**Colab Notebook:**  
https://colab.research.google.com/drive/1EVN91MVqMwpMZ3YFfPv9y8jOjvxideEB?usp=sharing

## Objective

The main objective of this project was to:

- Fine-tune a small language model for empathetic response generation.
- Use real human conversation data from the EmpatheticDialogues dataset.
- Train the model using Hugging Face's `Trainer` API.
- Compare the outputs of the base DistilGPT2 model and the fine-tuned model.
- Build a command-line interface to test the chatbot.

## Model Used

The base model used for this project was:

```text
DistilGPT2
```

DistilGPT2 is a smaller and lighter version of GPT-2. It is suitable for experimentation and fine-tuning when hardware resources are limited.

## Dataset Used

The dataset used was:

```text
EmpatheticDialogues by Facebook AI
```

The dataset contains conversations designed around emotional situations, making it suitable for training a chatbot that can generate empathetic and supportive responses.

The dataset columns used included:

```text
Situation
emotion
empathetic_dialogues
labels
```

Where:

- `Situation` describes the emotional context.
- `emotion` identifies the emotional category.
- `empathetic_dialogues` contains the user-side dialogue prompt.
- `labels` contains the expected empathetic response.

## Training Limitation

Due to limited hardware resources, the model was trained on only **30,000 samples** from the dataset instead of the complete dataset.

This limitation was mainly because training larger language models or using the full dataset requires more GPU memory, longer runtime, and stronger computational resources.

## Training Environment

The model was trained in **Google Colab** using GPU acceleration.

### Tools and Libraries

The main tools and libraries used were:

- Python
- Google Colab
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Pandas
- DistilGPT2
- Trainer API

## Training Process

The general training workflow was:

1. Load the EmpatheticDialogues dataset.
2. Clean and preprocess the dataset.
3. Format the data into a prompt-response structure.
4. Load the DistilGPT2 tokenizer and model.
5. Tokenize the dataset.
6. Fine-tune the model using Hugging Face `Trainer`.
7. Save the trained model.
8. Test the model using sample emotional wellness prompts.
9. Compare base model outputs with fine-tuned model outputs.
10. Build a CLI-based chatbot interface.

## Prompt Format

The training examples were formatted in a structure similar to:

```text
Emotion: stressed
Situation: The user feels overwhelmed by workload.
User: I am stressed because I have too much work to do.
Assistant: <empathetic response>
```

This helped the model learn how to generate responses based on both the user's message and the emotional context.

## Sample Results

### Test Case 1

**User Message:**

```text
I am stressed because I have too much work to do.
```

**Emotion:**

```text
stressed
```

**Situation:**

```text
The user feels overwhelmed by workload.
```

**Base DistilGPT2 Output:**

```text
It is very difficult for me to get a job at the end of this week and if you don't feel well, it's only after that time we will start making changes...
```

**Fine-Tuned DistilGPT2 Output:**

```text
Heck, it's hard for me to feel stress or something like that! Maybe he can help you out? How about helping him get some extra time off from his job now?
```

### Test Case 2

**User Message:**

```text
I have exams tomorrow and I feel like I will fail.
```

**Emotion:**

```text
anxious
```

**Situation:**

```text
The user is worried about exams.
```

**Base DistilGPT2 Output:**

```text
It's not my fault that the test was delayed for a while due to an unrelated issue...
```

**Fine-Tuned DistilGPT2 Output:**

```text
ive never studied before but i guess you should do it for now! You can always check the results in advance...
```

## Observations

The base DistilGPT2 model produced mostly random and unrelated text completions because it was not specifically trained for emotional support conversations.

The fine-tuned model showed improvement in task relevance because its responses became more connected to the emotional context. However, the responses were still not perfect and require further improvement in tone, coherence, grammar, and empathy quality.

## Key Learnings

This project helped demonstrate that fine-tuning can adapt a general-purpose language model to a more specific conversational task.

Important learnings from this task include:

- Fine-tuning improves task-specific relevance.
- Dataset quality strongly affects model response quality.
- Prompt formatting plays an important role in model behavior.
- Small models like DistilGPT2 can produce useful experiments but may struggle with coherence.
- Hardware limitations affect the amount of data and model size that can be used.
- Post-processing and safety filters are important for emotional wellness chatbots.

## CLI Application

A command-line interface was also created to test the chatbot locally. The CLI app allows the user to enter a message, after which the fine-tuned model generates an empathetic response.

Basic CLI workflow:

```text
User enters emotional wellness message
        ↓
Prompt is formatted with emotion and situation
        ↓
Fine-tuned DistilGPT2 generates response
        ↓
Response is cleaned and displayed in terminal
```

## How to Run the CLI App

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the chatbot:

```bash
python app.py
```

Example:

```text
You: I am stressed because I have too much work to do.
Bot: That sounds really overwhelming. Try taking one small step at a time.
```

## Future Improvements

The model can be improved further by:

- Training on the complete dataset.
- Cleaning noisy responses from the dataset.
- Using prompt masking during fine-tuning.
- Improving generation settings.
- Adding stronger response filtering.
- Using a larger instruction-tuned model.
- Building a Streamlit web interface.
- Adding more safety handling for crisis-related messages.

## Conclusion

This project successfully demonstrates the process of fine-tuning an existing Hugging Face model, DistilGPT2, for an empathetic chatbot task using Facebook AI's EmpatheticDialogues dataset.

Although the model was trained on only 30,000 samples due to limited hardware resources, the experiment showed that fine-tuning can make a base language model more relevant to emotional support conversations. Further improvements in dataset cleaning, training strategy, and model selection can make the chatbot more coherent, gentle, and reliable.
