# Udacity GenerativeAI - Nanodegree
Gen AI brings the creative dimension to AI. It helps to create novel content which include auto generative text, audio, code and Audio and much more.

## Nanodegree from Udacity in Generative AI
The exciting world of training generative AI models is about teaching computers to create new content, like text or images, by learning from huge datasets. This training helps AI to understand and recreate the complex patterns found in human language and visual arts. The process is intricate but immensely rewarding, leading to AI that can generate amazingly realistic outputs. Generation algorithms are incredible tools that allow AI to create text and images that seem amazingly human-like. By understanding and applying these smart algorithms, AI can generate new content by building upon what it knows, just like filling in missing puzzle pieces.

** Autoregressive text generation: Autoregressive text generation is like a game where the computer guesses the next word in a sentence based on the words that came before it. It keeps doing this to make full sentences.

** Latent space decoding: Imagine if you had a map of all the possible images you could create, with each point on the map being a different image. Latent space decoding is like picking a point on that map and bringing the image at that point to life.

** Diffusion models: Diffusion models start with a picture that's full of random dots like TV static, and then they slowly clean it up, adding bits of the actual picture until it looks just like a real photo or painting.

## Application of Generative AI
The applications of Generative AI span a gamut of exciting fields, broadening creativity and innovation in content creation, product design, scientific inquiry, data enhancement, and personalized experiences. The power of Generative AI lies in its ability to imagine and refine with speed, offering solutions and opening doors to future inventions.

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

## Large Language Model Introduction

![image](https://github.com/user-attachments/assets/ff1f01c4-0f0a-4c24-877e-fcf4d02d4b64)

![image](https://github.com/user-attachments/assets/a64a5206-8222-4ac1-87bf-bfd2edad1db6)

Sequence-to-sequence models can be used in NLP since we can think of text as a sequence.
Recurrent neural networks, or RNNs, are a deep learning model that can work well for sequences by taking in one token at a time to produce a hidden state that condenses past information so it can be used further down the line.

Disadvantages of RNNs include:
The vanishing gradient problem where tokens earlier in the sequence are "forgotten"
Long training and slow usage because of RNNs' sequential nature
RNNs can be used as autoregressive models, which can generate new text indefinitely by feeding in input text and the newly generated tokens.

Sampling techniques can adjust how tokens are chosen by the autoregressive model during text generation.

### Temperature
* Adjusts the randomness in choosing the next token
### Top-k sampling
* Samples from only the k most likely tokens
### Nucleus or top-p sampling
* Uses a dynamic cutoff for sampling the most likely tokens (cumulative probability is under p)
### Beam search
Considers the likelihood of strings of multiple tokens instead of just a single next token

### Limitations of RNNs
Recurrent neural networks (RNNs) were once the architecture of choice for many NLP tasks. However, they are limited because they can only keep track of a single hidden state (context vector), which creates an information bottleneck and causes gradual "forgetting" over long sequences.

Long short-term memory (LSTM) models were developed to address the problem of gradual forgetting using a forget gate and an input gate. This manages the "memory" and "forgetting" more intentionally.

LSTMs are not a complete solution for these challenges because they are still built on the underlying RNN architecture. This means that they still have information bottlenecks and can't use the built-in parallelism of GPUs.

## ELMo
Embeddings from Language Models (ELMo) built on LSTMs to create a new state-of-the-art in 2018. In particular, it was able to create contextual embeddings that could differentiate between the same word being used in different contexts.

ELMo also introduced the large-scale pre-training that we see with modern LLMs.

## Attention
Attention was first applied to language models in 2014. Instead of relying on a single context vector, attention allowed the model to combine all of the encoder's hidden states to become its context vector.

This raised the question: instead of trying to apply attention to RNNs, can we do better with just attention?

### Attention Scores
![image](https://github.com/user-attachments/assets/26af7e3f-ec61-4126-a435-00b96673b09c)

## "Attention Is All You Need"?
If self-attention is so powerful, can it replace RNNs altogether? Vaswani et al. (2017) argues that it can, with the right solutions in place to address its limitations.

### Problem: lack of input order
As previously described with ELMo, context is important for understanding the meaning of words.
Self-attention doesn't understand this by default, so we add positional encodings as part of the input embeddings.
### Problem: no nonlinearity between repeated self-attention layers
The reason that we typically use an activation function like ReLU in a neural network layer, rather than just a linear output, is to enable to model to capture more complexity. Linear outputs can be reduced to a simple y=mx+b style formula.
Self-attention layers don't have this nonlinearity by default, so we add a feed-forward network for each processed token afterward.
### Problem: "cheating" when predicting a sequence
The goal of a deep learning model is to be able to predict some unknown information given some known information. If all of the information is known, the model can't learn the relationships properly.
By default, self-attention can look at all of the data, including the "future" that it is trying to predict. To prevent this, we mask attention on future words during decoding.


## "Attention Is All You Need"?
If self-attention is so powerful, can it replace RNNs altogether? Vaswani et al. (2017) argues that it can, with the right solutions in place to address its limitations.

### Problem: lack of input order
As previously described with ELMo, context is important for understanding the meaning of words.
Self-attention doesn't understand this by default, so we add positional encodings as part of the input embeddings.
### Problem: no nonlinearity between repeated self-attention layers
The reason that we typically use an activation function like ReLU in a neural network layer, rather than just a linear output, is to enable to model to capture more complexity. Linear outputs can be reduced to a simple y=mx+b
y=mx+b style formula.
Self-attention layers don't have this nonlinearity by default, so we add a feed-forward network for each processed token afterward.
### Problem: "cheating" when predicting a sequence
The goal of a deep learning model is to be able to predict some unknown information given some known information. If all of the information is known, the model can't learn the relationships properly.
By default, self-attention can look at all of the data, including the "future" that it is trying to predict. To prevent this, we mask attention on future words during decoding.

## Building Generative AI Solutions
Traditional machine learning models typically served one specific task and developers integrate the single-task AI capability into their applications, like identifying an anomaly in an x-ray with a computer vision model. Transformer architecture enables the development of generative AI models called foundation models that train on huge datasets. Unlike earlier deterministic models that just identified patterns or followed rules, these AI models can generate data.

![image](https://github.com/user-attachments/assets/d11e8c3e-f010-45d6-ac0c-1fc94c6d5404)

These models handle multiple tasks and can be fine-tuned for specific ones. They generate new data like text, images, and music. Analysts estimate the market for generative AI to reach 200 billion by 2032. This means that the market is likely to double every two years for the upcoming decade.

The technical foundation that enables generative AI is a layered stack made of three layers:

### Infrastructure and Hardware Layer

The infrastructure and hardware layer consists of the resources that are needed to train, fine-tune, and run generative models. This layer includes silicon customized for AI computation like GPUs, Google TPUs, as well as hyperscale compute. The massive computing power, inclusive of network and storage, is required for training and serving models. Cloud providers like Amazon, Microsoft, Google, and Baidu typically possess these infrastructure resources and send them to developers via cloud services. Generative AI service providers like Amazon, Cohere, and Google offer specialized model deployment services in this layer for developers to accelerate model deployment, security, monitoring, and testing.

### Model and Platform Layer

This layer encompasses the algorithms, architectures, neural networks, and machine learning models that generate content or make predictions. If you're a developer who wants to work with generative AI, it's essential to have an understanding of the underlying models you're working with, the data they're trained on, and the potential biases and limitations they have. We can interact with the model layer in many ways, using APIs to access pre-trained models in this layer like OpenAI GPT models, or we can take pre-trained foundation models and fine-tune them with additional training data. This is useful for adapting a generic model to perform specific tasks or work with niche datasets. There are also development kits and libraries that you can take advantage of in this layer, like TensorFlow, PyTorch, and Hugging Face Transformers, which provide tools for building and training generative models. Finally, this layer encompasses deploying models into production environments. This could involve serving the model via a web service or endpoint, embedding it in an application, or deploying it on edge devices.

### Application layer

User-facing B2B and B2C applications that use generative models reside in the topmost layer, the application layer. This layer enables developers and users to interact with the complex underlying models in a user-friendly way.

The application layer is what end users see and interact with.
It translates user inputs into model queries and converts model outputs into interpretable results for users. At this layer, we can build user-facing applications using APIs that communicate with models, query them, and fetch results.
The application layer is dependent on the model layer for output to use in applications. Models often expect inputs in a specific format or structure, so application layer code is also responsible for formatting and transforming user inputs before they're fed into models.
Generative models have the potential to produce output that can be harmful, misleading, or biased. The application layer needs mechanisms to detect and mitigate these issues, ensuring the ethical use of your applications and solutions.

## The Problem with Generative Models
As a developer, there are two challenges you'll commonly face when working with Generative models in your applications and solutions:
** Unpredictable responses
** Statelessness

The only way we can interact with large language models is through natural language. That's it! You're probably used to writing programming functions that return reliable, reproducible results. These models work differently.
Generative AI models are unpredictable, and you send them natural language input, called a prompt, and receive output called a response that might be what you expected, or not.
* The same prompt might return different responses every time.
* The prompt you send the model will affect the quality of the response.
* The other problem you'll face using these models in development is their stateless nature. If you use the ChatGPT web or mobile application, it might appear like the model is keeping a history of your conversation and every time you send a prompt, the model is somehow adding it to a memory of the conversation you are having. This is an illusion.

Every time you call the API to send a prompt to the model, it responds in isolation, only using the messages you've sent in that API call to generate a response.
* These messages are called the context window of the model, and it's only so large.
* Suppose you want to create a chatbot that has a conversation with the user. In that case, you'll need to send all the previous messages in the entire conversation in your API call, which gives the model context of the entire conversation in order to generate a response that appears to the user like the model is remembering the previous conversation messages.
* At some point, you'll run out of context length in the unit of tokens to send the entire conversation, and you'll only be able to send the most recent messages
Tools and techniques exist to deal with this issue from a development standpoint

## Prompts and Prompt Engineering
Prompting is a crucial aspect of interacting with models, especially with language models like OpenAI GPT models. Prompts are input statements or questions that are provided to a model to generate a desired output or response.

The relationship between prompts and prompt engineering bridges both the model layer and the application layer.
* Without well-engineered prompts, generative models can provide poor output, resulting in your application not working the way you envisioned it.
* Code in the application layer often pre-processes or structures user input into effective prompts for the model.

![image](https://github.com/user-attachments/assets/79819192-db62-484b-bc21-4951be6e3278)

### Example: Let's consider an application that helps users identify potential illnesses based on symptoms they input. The user might input "I have a fever, cough, and tiredness."

The application interface could also provide checkboxes or buttons for common symptoms to aid users.
The raw input from the checkboxes the user checked will be too ambiguous for the model.
Application logic must engineer a prompt based on the selected checkbox values ({fever}, {cough}, and {tiredness}).
Code pre-processes the checkbox values into this prompt: "What are the potential illnesses associated with symptoms like {fever}, {cough}, and {tiredness}?" This is the prompt we send to the model.
The model receives the engineered prompt and outputs a response based on its training. It might respond with "The symptoms of fever, cough, and tiredness can be associated with conditions like the flu or mononucleosis. It's essential to consult a healthcare professional for a definitive diagnosis."

The application post-processes the model's response and displays it to the user in the application, perhaps providing links to more information about each one. It might add disclaimers or advise the user to consult a physician. In this example, prompt engineering in the application layer ensures that the model receives the most effective query to provide a relevant and valuable response.

### Generative AI Solution Components
Let's explore the components you'll be working with when you're designing and building applications that leverage generative AI models.
* At the heart of your solutions are large language models for text generation or image generation models. Your application will be sending them prompts and receiving and processing their output.
* The user interface is where users interact with the application. It's the bridge between users and AI models.
* Application logic is the controller of the application. It processes user inputs, interacts with AI models, as well as parsing and managing the flow of data.
* Databases can store user information and AI responses or be used alongside OpenAI functions or LangChain tools, or you can use them to fetch data to augment in RAG solutions. You might also have vector databases in your app for semantic search solutions.
* APIs connect our application with external services and platforms. They allow our solution to communicate with other AI models or fetch data from other applications.

  ![image](https://github.com/user-attachments/assets/88582578-bf01-4d49-b150-7cf07c934926)
  

### Evaluating, Versioning, and Auditing Output
How do we know that the response from the model is not biased, toxic, leaking proprietary data, or will damage organizational reputation? There are several software libraries and products available to integrate into our applications that perform this important task.

The initial input, the resulting prompt, and the corresponding response from the large language model can all be stored in a database. This historical record can be used for tracking user interactions, building history features, refining the AI model's performance, or providing audit trails.

## ReAct Prompting for Agents
A popular prompting method called ReAct can be used to create LLM-based agents. REACT prompt techniques use LLMs to generate both reasoning traces and task-specific actions.

* The reasoning aspect involves the AI agent thinking through a problem, breaking it down into smaller, manageable parts, and forming a plan or series of steps to solve it.
* The action component allows the agent to make decisions about gathering additional information relevant to the task.
* The observation step often involves delivering a final response about a task or recording an event. These observations can be used for further reasoning. REACT-enabled agents actively seek out new information, update their understanding, and refine their reasoning based on observations.
* A REACT prompt for a wellness agent might be:

"Your goal is to improve the wellness of the user by interleaving thought, action, and observation."







## Important Resources
* Read about CommonCrawl on its website: https://commoncrawl.org/
* Read about the Github dataset on its website: https://www.githubarchive.org/
* Read about the Wikipedia dataset on its website: Wikimedia Downloads(opens in a new tab).
* Read about the Gutenberg Project on its website: https://www.gutenberg.org/
* Hugging face reference Tutorial: https://huggingface.co/docs/transformers/en/index
* https://medium.com/@MUmarAmanat/fine-tune-llm-with-peft-60b2798f1e5f
* https://medium.com/@dassum/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07
* Audi GEN AI : https://www.audi-mediacenter.com/en/press-releases/reinventing-the-wheel-felgan-inspires-new-rim-designs-with-ai-15097
* Synthetic dataset course: https://www.udacity.com/course/small-data--cd12528
* Attention to want you need : https://arxiv.org/abs/1706.03762
* Transformer Illustration : https://jalammar.github.io/illustrated-transformer/
* 
