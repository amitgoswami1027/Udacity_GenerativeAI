# Udacity GenerativeAI - Nanodegree
Gen AI brings the creative dimension to AI. It helps to create novel content which include auto generative text, audio, code and Audio and much more.

## Nanodegree from Udacity in Generative AI
The exciting world of training generative AI models is about teaching computers to create new content, like text or images, by learning from huge datasets. This training helps AI to understand and recreate the complex patterns found in human language and visual arts. The process is intricate but immensely rewarding, leading to AI that can generate amazingly realistic outputs. Generation algorithms are incredible tools that allow AI to create text and images that seem amazingly human-like. By understanding and applying these smart algorithms, AI can generate new content by building upon what it knows, just like filling in missing puzzle pieces.

** Autoregressive text generation: Autoregressive text generation is like a game where the computer guesses the next word in a sentence based on the words that came before it. It keeps doing this to make full sentences.

** Latent space decoding: Imagine if you had a map of all the possible images you could create, with each point on the map being a different image. Latent space decoding is like picking a point on that map and bringing the image at that point to life.

** Diffusion models: Diffusion models start with a picture that's full of random dots like TV static, and then they slowly clean it up, adding bits of the actual picture until it looks just like a real photo or painting.
## Application of Generative AI
The applications of Generative AI span a gamut of exciting fields, broadening creativity and innovation in content creation, product design, scientific inquiry, data enhancement, and personalized experiences. The power of Generative AI lies in its ability to imagine and refine with speed, offering solutions and opening doors to future inventions.
![image](https://github.com/user-attachments/assets/34eed4d5-f5ec-4b4e-a96b-4b481c8fb4fe)

![image](https://github.com/user-attachments/assets/e5b2cb50-6f74-4847-9855-b51f3499f990)

![image](https://github.com/user-attachments/assets/90c94382-9bde-4f5d-94e0-80c3f7a7b688)

![image](https://github.com/user-attachments/assets/3ea8771b-5f07-45d3-b736-9d5972a41625)

![image](https://github.com/user-attachments/assets/69a17a2c-b31c-4eeb-8d3f-31028ebc32fd)

## Now it's your turn
Can you find a question that the LLM cannot answer? Try to stump the LLM by asking it a question that you think it should be able to answer, but it cannot. Sometimes this can be hard, because the commercial LLM offerings are continually improving their models to handle different types of questions more gracefully, and often, if an LLM cannot answer a question, it will simply say something like "I don't know" or "I don't understand".

Spend a few minutes finding a question that the LLM cannot answer. Then, write down the question and the LLM's response in the cell below.


 # What Is a Perceptron
 A perceptron is an essential component in the world of AI, acting as a binary classifier capable of deciding whether data, like an image, belongs to one class or another. It works by adjusting its weighted inputs—think of these like dials fine-tuning a radio signal—until it becomes better at predicting the right class for the data. This process is known as learning, and it shows us that even complex tasks start with small, simple steps.

 ![image](https://github.com/user-attachments/assets/d5344e0c-69b8-44b5-97f8-7e1506d80d07)

 * Perceptron: A basic computational model in machine learning that makes decisions by weighing input data. It's like a mini-decision maker that labels data as one thing or another.
 * Binary Classifier: A type of system that categorizes data into one of two groups. Picture a light switch that can be flipped to either on or off.
 * Activation Function: A mathematical equation that decides whether the perceptron's calculated sum from the inputs is enough to trigger a positive or negative output

We learned that training deep neural networks involves guided adjustments to improve their performance on tasks like image recognition. By gradually refining the network's parameters and learning from mistakes, these networks become smarter and more skilled at predicting outcomes. The marvel of this technology is its ability to turn raw data into meaningful insights.

* Gradient Descent: This method helps find the best settings for a neural network by slowly tweaking them to reduce errors, similar to finding the lowest point in a valley.
* Cost Function: Imagine it as a score that tells you how wrong your network's predictions are. The goal is to make this score as low as possible.
* Learning Rate: This hyperparameter specifies how big the steps are when adjusting the neural network's settings during training. Too big, and you might skip over the best setting; too small, and it'll take a very long time to get there.
* Backpropagation: Short for backward propagation of errors. This is like a feedback system that tells each part of the neural network how much it contributed to any mistakes, so it can learn and do better next time.

## Generation Algorithms
Generation algorithms are incredible tools that allow AI to create text and images that seem amazingly human-like. By understanding and applying these smart algorithms, AI can generate new content by building upon what it knows, just like filling in missing puzzle pieces.

Technical Terms Explained:
Autoregressive text generation: Autoregressive text generation is like a game where the computer guesses the next word in a sentence based on the words that came before it. It keeps doing this to make full sentences.

Latent space decoding: Imagine if you had a map of all the possible images you could create, with each point on the map being a different image. Latent space decoding is like picking a point on that map and bringing the image at that point to life.

Diffusion models: Diffusion models start with a picture that's full of random dots like TV static, and then they slowly clean it up, adding bits of the actual picture until it looks just like a real photo or painting

## What Is a Foundation Model
A foundation model is a powerful AI tool that can do many different things after being trained on lots of diverse data. These models are incredibly versatile and provide a solid base for creating various AI applications, like a strong foundation holds up different kind of buildings. By using a foundation model, we have a strong starting point for building specialized AI tasks.

## Research Pre-Training Datasets

### Step 1: Evaluate the available pre-training datasets
When it comes to training language models, selecting the right pre-training dataset is important. In this exercise, we will explore the options available for choosing a pre-training dataset, focusing on four key sources:
*CommonCrawl,
*Github,
*Wikipedia, and
*the Gutenberg project.
These sources provide a wide range of data, making them valuable resources for training language models. If you were tasked with pre-training an LLM, how would you use these datasets and how would you pre-process them? Are there other sources you would use?

### Step 2. Select the appropriate datasets
Based on the evaluation, choose the datasets that best suit the requirements of pre-training a Language Model (LLM). Consider factors such as the diversity of data, domain-specific relevance, and the specific language model objectives.
For your use case, rank the datasets in order of preference. For example, if you were training a language model to generate code, you might rank the datasets as follows:

*Github
*Wikipedia
*CommonCrawl
*Gutenberg project
Explain your reasoning for the ranking. For example, you might say that GitHub is the best dataset because it contains a large amount of code, and the code is structured and clean. You might say that Wikipedia is the second-best dataset because it contains a large amount of text, including some code. You might say that CommonCrawl is the third-best dataset because it contains a large amount of text, but the text is unstructured and noisy. You might say that the Gutenberg project is the worst dataset because it contains text that is not relevant to the task.

### Step 3. Pre-process the selected datasets
Depending on the nature of the chosen datasets, pre-processing may be required. This step involves cleaning the data, removing irrelevant or noisy content, standardizing formats, and ensuring consistency across the dataset. Discuss how you would pre-process the datasets based on what you have observed.

### Step 4. Augment with additional sources
Consider whether there are other relevant sources that can be used to augment the selected datasets. These sources could include domain-specific corpora, specialized text collections, or other publicly available text data that aligns with your language model's objectives, such as better representation and diversity.

## The GLUE Benchmarks
The GLUE benchmarks serve as an essential tool to assess an AI's grasp of human language, covering diverse tasks, from grammar checking to complex sentence relationship analysis. By putting AI models through these varied linguistic challenges, we can gauge their readiness for real-world tasks and uncover any potential weaknesses.

Semantic Equivalence: When different phrases or sentences convey the same meaning or idea.

Textual Entailment: The relationship between text fragments where one fragment follows logically from the other.

![image](https://github.com/user-attachments/assets/a7bded29-519a-4158-85fc-113e85cdcedb)

SuperGlue is designed as a successor to the original GLUE benchmark. It's a more advanced benchmark aimed at presenting even more challenging language understanding tasks for AI models. Created to push the boundaries of what AI can understand and process in natural language, SuperGlue emerged as models began to achieve human parity on the GLUE benchmark. It also features a public leaderboard, facilitating the direct comparison of models and enabling the tracking of progress over time.

## What Is Adaptation
Adaptation in AI is a crucial step to enhance the capabilities of foundation models, allowing them to cater to specific tasks and domains. This process is about tailoring pre-trained AI systems with new data, ensuring they perform optimally in specialized applications and respect privacy constraints. Reaping the benefits of adaptation leads to AI models that are not only versatile but also more aligned with the unique needs of organizations and industries.

Fine-tuning: This is a technique in machine learning where an already trained model is further trained (or tuned) on a new, typically smaller, dataset for better performance on a specific task.




## Training Generative AI Models

https://github.com/DrRuin/Lightweight-Fine-Tuning/blob/main/Lightweight-Fine-Tuning.ipynb

## Important Links
* Read about CommonCrawl on its website: https://commoncrawl.org/
* Read about the Github dataset on its website: https://www.githubarchive.org/
* Read about the Wikipedia dataset on its website: Wikimedia Downloads(opens in a new tab).
* Read about the Gutenberg Project on its website: https://www.gutenberg.org/
* Hugging face reference Tutorial: https://huggingface.co/docs/transformers/en/index
* https://medium.com/@MUmarAmanat/fine-tune-llm-with-peft-60b2798f1e5f
