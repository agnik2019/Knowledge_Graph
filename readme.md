### Introduction to Knowledge Graphs

Knowledge Graphs are networks that represent data in a graphical format. The beauty of Knowledge Graphs lies in their representation of concepts, events and entities as nodes, and the relationships between them as edges. These relationships determine the context of the node, and consequently, allow for better understanding of a word’s semantics and distinguishing between its multiple possible meanings. For example, Google’s Knowledge Graph supports Google Search, which can distinguish between the brand “Apple” and the fruit “apple”. Knowledge Graphs are applicable across a range of domains and applications, including product recommendations in retail, search engine optimisation, anti-money laundering initiatives, and healthcare.  
However, the utilisation of Knowledge Graphs is hindered by their difficult, costly and time-consuming construction process. This challenge has spurred a new wave of research exploring automatic Knowledge Graph construction. Particularly, there is growing interest in integrating large language models (LLMs) like GPT-4 into the construction process due to their remarkable language processing abilities. In this article, we will begin by briefly examining the difficulties associated with Knowledge Graph construction. Then, we will compare Knowledge Graphs and LLMs as knowledge bases. Finally, we will review existing methods for automatic Knowledge Graph construction that utilise LLMs.

### Difficulties in Constructing Knowledge Graphs

Previous Knowledge Graph construction methods are based on crowdsourcing or text mining. Popular crowdsourcing-based Knowledge Graphs like [WordNet](https://wordnet.princeton.edu/) and [ConceptNet](https://conceptnet.io/) were constructed with significant human labour but are limited to predefined sets of relations. Meanwhile, text mining-based approaches extract knowledge from documents, but are limited to extracted relations stated explicitly within the text. This approach also involves many steps like coreference resolution, named entity recognition, and more. You are welcome to read more about the Knowledge Graph construction process in [this article](https://medium.com/@researchgraph/unlocking-intelligence-the-journey-from-data-to-knowledge-graph-4d7a08e5f4e0).  
The difficulties are compounded by the fact that different Knowledge Graphs are constructed for each field or application. Given the various concepts and terminologies used in each field, there is no universal approach to create Knowledge Graphs. Specific domains also present their own challenges. For example, Knowledge Graphs are immensely useful in the service computing community as they assist in resource management, personalised recommendations and customer understanding. However, the Knowledge Graph in this context requires knowledge and concepts from diverse fields, and the data required to build the Knowledge Graph is both highly scattered and largely unannotated. These factors significantly increase the time, effort and costs involved in producing a Knowledge Graph.

### Knowledge Graphs versus Large Language Models

Both Knowledge Graphs and LLMs can be queried to retrieve knowledge. In the figure below, Knowledge Graphs locate answers by finding relevant connected nodes, while LLMs are prompted to fill in the \[MASK\] token to complete the sentence. LLMs like GPT-4 and BERT have recently gained lots of attention due to their impressive abilities to understand language. It is well-known that LLMs continue to grow in size every year and are trained on vast amounts of data, enabling them to possess immense knowledge. Many people might turn to ChatGPT to ask a question instead of searching for it on Google. Naturally, the next question for the research community was to explore if LLMs (like GPT) could replace Knowledge Graphs (like Google Knowledge Graph) as primary sources of knowledge.  
![][image1]  
Fig: Retrieving knowledge from knowledge graphs (left) and large language models (right). Source: Yang et al., 2023 [https://doi.org/10.48550/ARXIV.2306.11489](https://doi.org/10.48550/ARXIV.2306.11489)

Further research revealed that despite possessing more fundamental world knowledge, LLMs struggled to recall relational facts and deduce relationships between actions and events. Despite possessing numerous advantages, LLMs also suffer from challenges such as:

* **Hallucinations:** LLMs occasionally produce convincing but incorrect information. Conversely, Knowledge Graphs provide structured and explicit knowledge grounded in its factual data sources.  
* **Limited reasoning abilities:** LLMs struggle to comprehend and use supporting evidence to draw conclusions, especially in numerical computation or symbolic reasoning. The relationships captured in Knowledge Graphs allow for better reasoning capabilities.  
* **Lack of domain knowledge:** While LLMs are trained on vast amounts of general data, they lack knowledge from domain-specific data like medical or scientific reports with specific technical terms. Meanwhile, Knowledge Graphs can be constructed for specific domains.  
* **Knowledge obsolescence:** LLMs are expensive to train and are not regularly updated, causing their knowledge to become outdated over time. Knowledge Graphs, on the other hand, have a more straightforward update process that does not require retraining.  
* **Bias, privacy and toxicity:** LLMs may give biased or offensive responses, whereas Knowledge Graphs are typically built from reliable data sources devoid of these biases.

Knowledge Graphs do not encounter these same issues and exhibit better consistency, reasoning ability and interpretability, though they do have their own set of limitations. Aside from those discussed previously, Knowledge Graphs also lack the flexibility that LLMs enjoy from their unsupervised training process.

### Merging Knowledge Graphs and Large Language Models

Consequently, there have been numerous research efforts aimed at merging LLMs and Knowledge Graphs. While Knowledge Graphs possess the ability to guide LLMs toward better accuracy, LLMs can assist Knowledge Graphs in knowledge extraction during construction and improve the Knowledge Graph’s quality. There are a few approaches to merging these two concepts:

* **Employing LLMs to aid in automatic Knowledge Graph construction:** LLMs can extract knowledge from data to populate a Knowledge Graph. Further details on this method will be discussed below.  
* **Teaching LLMs to search for knowledge from Knowledge Graphs:** As shown in the image below, Knowledge Graphs can enhance the reasoning processes of LLMs so that LLMs can arrive at more accurate answers.  
* **Combining them into knowledge graph enhanced pre-trained language models (KGPLMs):** These methods aim to incorporate Knowledge Graphs into the LLM training process.

![][image2]  
Fig: (a) LLMs alone might not be sufficient. (b) With Knowledge Graphs, the reasoning processes of LLMs can be enhanced. Source: Feng et al., 2023 [https://doi.org/10.48550/arXiv.2309.03118](https://doi.org/10.48550/arXiv.2309.03118)

### Automatic Knowledge Graph Construction with Large Language Models

#### Earlier methods

One of the earlier methods proposed in 2019 was **COMET (or COMmonsEnse Transformers)**, which used a fine-tuned generative LLM, GPT in this case, to construct Knowledge Graphs by generating tail entities given head entities and relations. Given the “seed” and “relation” in the image below, COMET generated the “completion” response, which was evaluated by humans to assess the plausibility of the responses. These seed-relation-completion triples could then be used to form Knowledge Graphs. For example, “piece” and “machine” could form two nodes connected by a “PartOf” relationship.  
![][image3]  
Fig: Responses in the “completion” column are generated by COMET, given the “seed” and “relation”. Source: Bosselut et al., 2019 [https://doi.org/10.48550/ARXIV.1906.05317](https://doi.org/10.48550/ARXIV.1906.05317)

#### Using ChatGPT as an information extractor

A Knowledge Graph constructed specifically for the service domain, named **BEAR**, was developed using ChatGPT to avoid the effort and costs associated with manual data annotation. To do this, an ontology specific to the domain was created, which served as a base for the Knowledge Graph and identified the concepts and characteristics that the Knowledge Graph should be populated with later on. ChatGPT would then be prompted to extract the relevant content and relationships from unstructured text data, like in the image below. The automatically extracted information was subsequently incorporated into the Knowledge Graph to build it.  
![][image4]  
Fig: ChatGPT was used to extract information from text data in the BEAR model. Source: Yu et al., 2023 [https://doi.org/10.1007/978-3-031-48421-6\_23](https://doi.org/10.1007/978-3-031-48421-6_23)

#### Semi-automatic Knowledge Graph construction using LLMs

Once again using ChatGPT as an information extractor, Kommineni et al. recently proposed using ChatGPT-3.5 in their Knowledge Graph construction method with human domain experts verifying results in two stages, as illustrated below. The difference between this method and the previous one is that LLMs play a more active role here. Beginning with specific datasets, ChatGPT was prompted to generate competency questions (CQs), which were abstract-level questions about the data. An ontology was created by extracting concepts and relationships from the CQs, again by prompting ChatGPT. Answers to the CQs were retrieved from the data and given to ChatGPT, which was instructed to extract key entities, relationships and concepts and map them onto the ontology to construct the Knowledge Graph.  
![][image5]  
Fig: A semi-automatic method for Knowledge Graph construction using LLMs. Source: Kommineni et al., 2024 [https://doi.org/10.48550/ARXIV.2403.08345](https://doi.org/10.48550/ARXIV.2403.08345)

#### **Harvesting Knowledge Graphs from LLMs**

The final method discussed in this article **sought to extract information directly from LLMs** themselves. Hao et al. recognised that there was vast amounts of knowledge stored within LLMs from their initial training that could be put to use. The image below shows the steps to harvesting the LLM’s knowledge. The process started with an initial prompt and as few as two example entity pairs. A text paraphrase model was employed to paraphrase the prompt and derive modified prompts from the original one. Subsequently, the LLM was searched for entity pairs corresponding to this set of prompts. Using a search and re-scoring method, the most relevant pairs were extracted to form the Knowledge Graph, with the entities in the pairs as nodes and the prompts as relationships.  
This approach allowed for better relation qualities in the resulting Knowledge Graphs as the derived relations possessed several characteristics unseen in traditionally constructed Knowledge Graphs:  
Relations could be complex, for example, “A is capable of, but not good at, B”.  
Relations could involve more than two entities, like “A can do B at C”.  
Interestingly, forming Knowledge Graphs using LLMs also presented a new way to visualise and quantify the knowledge captured within a LLM.  
![][image6]  
Fig: The process of automatic Knowledge Graph construction in BertNet. Source: Hao et al., 2022 [https://doi.org/10.48550/ARXIV.2206.14268](https://doi.org/10.48550/ARXIV.2206.14268)

### Knowledge Graph in Medical Domains \[Breast Cancer\]

* Construction and application of Chinese breast cancer knowledge graph based on multi-source heterogeneous data [https://www.aimspress.com/aimspress-data/mbe/2023/4/PDF/mbe-20-04-292.pdf](https://www.aimspress.com/aimspress-data/mbe/2023/4/PDF/mbe-20-04-292.pdf)   
* Knowledge Graph for Breast Cancer Prevention and Treatment: Literature-Based Data Analysis Study [https://medinform.jmir.org/2024/1/e52210](https://medinform.jmir.org/2024/1/e52210)   
* Building a Knowledge Graph Representing Causal Associations Between Risk Factors and Incidence of Breast Cancer [https://pubmed.ncbi.nlm.nih.gov/34042671/](https://pubmed.ncbi.nlm.nih.gov/34042671/)   
* Healthcare knowledge graph construction [https://arxiv.org/pdf/2207.03771](https://arxiv.org/pdf/2207.03771)   
* Accelerating Medical Knowledge Discovery through Automated Knowledge Graph Generation and Enrichment [https://arxiv.org/pdf/2405.02321](https://arxiv.org/pdf/2405.02321)   
* Towards electronic health record-based medical knowledge graph construction, completion, and applications: A literature study [https://pubmed.ncbi.nlm.nih.gov/37230406/](https://pubmed.ncbi.nlm.nih.gov/37230406/) 

### Self-Refinement Enhanced Knowledge Graph Retrieval (Re-KGR) \[Development and Evaluation\]

This section discusses the development and evaluation of the **Self-Refinement Enhanced Knowledge Graph Retrieval** (Re-KGR) technique, designed to mitigate hallucinations in large language models (LLMs) by integrating knowledge graphs (KGs) to ensure truthful and factual outputs.

Hallucinations in large language models (LLMs) pose a significant threat to their reliability, particularly in critical domains such as healthcare, finance, and legal services. To address this, researchers have proposed a novel technique called Self-Refinement Enhanced Knowledge Graph Retrieval (Re-KGR). This method strategically retrieves relevant knowledge from **curated knowledge graphs** (KGs) to reduce the likelihood of hallucinations, focusing on critical components of the generated output that are most likely to be prone to errors. The Re-KGR approach employs a **"refine-then-retrieve"** paradigm, leveraging the LLM's own **predictive probability distributions to identify tokens or entities that may lead to hallucinations**. It then extracts and refines relevant knowledge triples, verifies them against a domain-specific KG, and rectifies the output accordingly. Experiments on a medical question-answering dataset have shown that Re-KGR, when integrated with the LLaMA-7b model and the contrastive decoding technique DoLa, significantly improves truthfulness scores and reduces retrieval time, demonstrating its potential to enhance the factual accuracy of LLMs while maintaining computational efficiency.

Opinions

* The authors believe that the integration of knowledge graphs is a promising approach to mitigate hallucinations in LLMs, enhancing the truthfulness and factual accuracy of their outputs.  
* The Re-KGR method is seen as an improvement over existing KG-augmented approaches due to its selective retrieval process, which reduces computational costs and improves efficiency.  
* The effectiveness of Re-KGR is highly dependent on the quality and comprehensiveness of the domain-specific knowledge graphs used for retrieval and verification.  
* The authors acknowledge that while Re-KGR shows promise, it is not a complete solution to all hallucination challenges and should be used in conjunction with other techniques, including data and prompt construction, model re-training, and human oversight.  
* The study suggests that Re-KGR can be extended to other knowledge-intensive domains beyond healthcare, indicating the method's versatility and potential for broader applications.  
* The authors advocate for the importance of ongoing curation and expansion of knowledge graphs to ensure their accuracy and relevance for use with Re-KGR and similar techniques.  
* The research highlights the need for future exploration into multi-sentence reasoning and coherent knowledge infusion to handle more complex language generation tasks.  
* The authors propose that human-computer interaction could further refine and validate the knowledge injection process, enhancing the reliability of AI-generated content in high-stakes applications.


**Harnessing Knowledge Graphs to Mitigate Hallucinations in Large Language Models**  
Large language models (LLMs) have emerged as powerful tools capable of generating human-like text across a wide range of domains. From creative writing and content generation to question answering and task completion, LLMs are demonstrating remarkable capabilities that were once thought to be exclusive to humans. However, as these models continue to grow in size and complexity, a significant challenge has arisen: the tendency for LLMs to generate outputs that deviate from factual knowledge, a phenomenon known as “hallucination.”

Hallucinations in LLMs can manifest in various forms, such as generating content that contradicts the given input, violates contextual coherence, or simply contradicts established facts \[38\]. This issue poses a serious threat to the reliability and trustworthiness of LLMs, particularly in critical domains like healthcare, finance, and legal services, where inaccurate information can have severe consequences \[17, 25\]. As a result, there is an urgent need to develop effective strategies for mitigating hallucinations and ensuring that LLMs produce outputs that are truthful, factual, and grounded in reliable knowledge sources.

One promising approach to addressing this challenge is the integration of knowledge graphs (KGs) into the generation process of LLMs. KGs are structured databases that represent knowledge in the form of entities and their relationships, providing a rich and organized source of factual information \[26\]. By leveraging KGs, LLMs can potentially ground their outputs in established knowledge, reducing the likelihood of hallucinations and enhancing the truthfulness of their generated content.

In this article, we will explore a novel technique called “Self-Refinement Enhanced Knowledge Graph Retrieval” (Re-KGR), proposed by researchers in a recent [study](https://arxiv.org/pdf/2405.06545) \[1\]. The Re-KGR method aims to mitigate hallucinations in LLMs by strategically retrieving and integrating relevant knowledge from curated KGs, while minimizing the computational costs associated with traditional retrieval approaches.

We will examine the underlying principles of Re-KGR, examine its potential advantages, and discuss the experimental results and implications for the future of truth-grounded language generation.

#### **I. Understanding Hallucinations in LLMs**

Before exploring the Re-KGR approach, it is essential to understand the nature and causes of hallucinations in LLMs. Hallucinations can be broadly categorized into three types: **input-conflicting, context-conflicting, and fact-conflicting** \[38\]. Input-conflicting hallucinations occur when the generated output contradicts the given input or prompt, while context-conflicting hallucinations involve deviations from the previously generated context. Fact-conflicting hallucinations, which are the focus of this article, refer to situations where the LLM’s output contradicts established facts or knowledge.  
Hallucinations in LLMs can stem from various factors. One potential cause is the lack of domain-specific information or up-to-date knowledge in the training datasets used to develop the models \[22, 40\]. Even when relevant knowledge is present, LLMs may struggle to effectively leverage and apply it, particularly in specialized domains or for edge cases that were underrepresented in the training data \[34\]. Additionally, some researchers suggest that the intentional incorporation of “randomness” into the generation process to enhance output diversity can inadvertently increase the risk of hallucinations, as it introduces unexpected and potentially erroneous content \[19\].  
In the context of medical question-answering tasks, a domain that is heavily knowledge-intensive and where factual accuracy is critical, fact-conflicting hallucinations pose significant challenges \[18\]. Inaccurate or nonfactual information generated by LLMs in this domain could lead to serious consequences for patients, potentially resulting in misdiagnoses, incorrect treatment recommendations, or even medical incidents.

#### **II. Existing Approaches to Mitigate Hallucinations**

Recognizing the importance of mitigating hallucinations in LLMs, researchers have explored various strategies to address this issue. One prominent approach is **Retrieval Augmented Generation** (RAG), which involves integrating external knowledge sources into the generation process \[17, 33\]. These knowledge sources can take the form of unstructured data, such as web pages or documents \[14\], or structured databases like knowledge graphs (KGs) \[17\].  
The integration of KGs has shown particular promise in mitigating hallucinations, as these structured knowledge repositories can faithfully represent domain-specific knowledge in the form of triples comprised of entities and their relationships \[26\]. For example, in the medical domain, a KG could contain triples representing diseases, their associated treatments, side effects, diagnostic tests, and other complex medical mechanisms.  
Existing KG-augmented approaches **have explored different stages** at which the knowledge can be integrated into the generation process. One approach is to **condition the LLM on the retrieved knowledge prior to generation**, prompting it to generate outputs that align with the provided information \[27, 31\]. However, this method can still yield inaccurate responses due to the LLM’s limited reasoning capabilities and its inability to effectively leverage the provided knowledge.  
To address this limitation, some researchers have proposed **post-generation techniques that leverage the retrieved knowledge to re-rank or re-generate candidate responses** \[17, 28\]. These methods aim to improve the truthfulness and coherence of the generated outputs by incorporating external knowledge after the initial generation phase.  
While these KG-augmented approaches have shown promising results in mitigating hallucinations, they often come with significant computational costs. Many existing methods require multiple rounds of retrieval and verification for each factual statement present in the generated responses, even those that originate from the input question or are generated with high confidence \[11, 15, 17\]. This redundancy can impede the practical application of these techniques in real-world scenarios, where computational efficiency is crucial.  
![][image7]

#### **III. Self-Refinement Enhanced Knowledge Graph Retrieval**

To address the limitations of existing KG-augmented approaches and facilitate the efficient integration of external knowledge for hallucination mitigation, researchers have proposed a novel technique called **“Self-Refinement Enhanced Knowledge Graph Retrieval” (Re-KGR)** \[1\]. The key innovation of Re-KGR is its ability to strategically identify and retrieve only the critical knowledge components that are most likely to be prone to hallucinations, thereby reducing the overall retrieval needs and computational costs.  
The **Re-KGR** approach follows a “**refine-then-retrieve**” paradigm, leveraging the LLM’s own **predictive probability distributions to identify tokens or entities that have a high potential for hallucination**. This identification process is based on the attribution of next-token predictive probability distributions across different tokens and various model layers, as well as the divergence between the output distributions of the final layer and intermediate layers.

##### **A**. **Identifying Critical Entities Likely to Cause Hallucinations**

The entity detection component of Re-KGR analyzes several criteria derived from the LLM’s internal probability distributions to identify tokens or entities that are likely to be associated with hallucinations. These criteria include:

* **Max Value of Next-Token Predictive Probability:** For a given token position s, the LLM outputs a probability distribution p(x\_s) over the vocabulary, representing the predicted probabilities for the next token x\_s. The max value c\_m is calculated as c\_m \= max(p(x\_s)). A lower max value indicates that the model is less confident about the next token prediction, suggesting potential hallucination \[1\].  
* **Entropy of Next-Token Predictive Probability:** The entropy c\_e measures the uncertainty or randomness of the predictive probability distribution p(x\_s). It is calculated as c\_e \= E(p(x\_s)) \= \-Σ\_{d=1}^D p(x\_s^d) log p(x\_s^d), where D is the vocabulary size, and x\_s^d is the d-th element of the probability vector p(x\_s). A higher entropy means the distribution is more spread out and uncertain, which could indicate hallucination \[35\].  
* **Output Divergence Between Layers:** The idea behind this criterion is that for tokens requiring factual knowledge, there will be a higher divergence between the output distributions at the final layer q\_N(x\_s) and intermediate layers q\_j(x\_s), where j ∈ J\* and J\* is a set of candidate intermediate layers. The uncertainty score c\_js is calculated as c\_js \= max(D(q\_N(x\_s), q\_j(x\_s))), j ∈ J\*, where D(·) is a distribution distance measure like Jensen-Shannon divergence. A higher divergence suggests the token is more important and may require external knowledge grounding \[10\].  
* For each of these criteria (c\_m, c\_e, c\_js), Re-KGR performs quartile-based anomaly detection to flag tokens with unusually low max values, high entropy, or high divergence as potentially hallucinated tokens requiring knowledge retrieval.

##### **B. Extracting and Refining Relevant Knowledge Triples**

After identifying the critical entities or tokens that are likely to be associated with hallucinations, Re-KGR extracts all factual statements from the LLM’s generated output in the form of knowledge triples (subject-predicate-object). However, to reduce the retrieval needs, the approach retains only those triples that contain at least one of the identified critical entities.  
For example, if the LLM generates the response “Headaches are a symptom of brain tumors,” it would extract the triple . If the entity “headaches” is flagged as a critical entity based on the uncertainty criteria, this triple would be included in the refined set of triples for subsequent knowledge retrieval. However, if none of the entities in the triple are identified as critical, it would be discarded, reducing the number of triples that need to be verified against external knowledge sources.

##### **C. Efficient Knowledge Graph Retrieval**

Once the refined set of triples has been assembled, Re-KGR employs it to retrieve corresponding knowledge from a domain-specific KG constructed from sources like the Clinical Knowledge Graph (CKG) \[29\] and PrimeKG \[6\] for the medical domain. The retrieval process involves the following steps:  
Triple Expansion: The refined triple set T\_f is expanded into T\_e by finding synonyms for each entity and predicate using a thesaurus or entity linking system. For example, the triple could be expanded to include synonyms like , , and so on.  
Knowledge Graph Querying: The expanded set T\_e is then used to query the domain-specific KG to retrieve all matching knowledge triples T\_g. This step leverages the structured nature of KGs to efficiently retrieve relevant factual information based on the refined set of triples.

##### **D. Verifying and Rectifying Outputs**

Against Retrieved Knowledge In the final stage of the Re-KGR process, the retrieved knowledge triples from the KG are compared against the original set of triples extracted from the LLM’s output. This verification and rectification step aims to ensure that the generated content aligns with the established knowledge in the KG.

* **Semantic Similarity Scoring:** A semantic similarity model S(·) is used to compare each original triple t\_i in T\_f against its retrieved counterparts {t\_{g\_i}} in T\_g. This produces a vector of similarity scores s\_i \= S(t\_i, {t\_{g\_i}}).  
* **Triple Verification:** If the maximum similarity score max(s\_i) exceeds a predefined threshold τ, the original triple t\_i is considered verified and retained in the final output.  
* **Triple Rectification:** If the maximum similarity score is below the threshold τ, the original triple t\_i is replaced with the maximally similar retrieved triple t’i \= t{g\_i}, where g \= argmax(s\_i). This step ensures that inaccurate or hallucinated triples are rectified by incorporating the corresponding factual knowledge from the KG.  
* **Response Update and Grammar Correction:** After verifying or replacing all triples, the rectified set of triples is used to update the LLM’s original response text. Finally, a grammar correction model is applied to ensure coherence and fluency in the final output.

By following this process, Re-KGR effectively mitigates hallucinations in the LLM’s responses by selectively retrieving and integrating relevant knowledge from the KG, while minimizing the overall retrieval efforts and computational costs compared to naive retrieval approaches that verify every factual statement.

#### **IV. Advantages of Re-KGR**

The Re-KGR approach offers several advantages over existing KG-augmented methods for mitigating hallucinations in LLMs:

##### **A. Reduced Retrieval Needs**

One of the key advantages of Re-KGR is its ability to significantly reduce the retrieval needs by focusing on only the critical components of the generated output that are most likely to be prone to hallucinations. By identifying and refining the set of triples to be verified against the KG, Re-KGR avoids the redundancy of retrieving and verifying every factual statement present in the response, even those that are unlikely to contain hallucinations. This selective retrieval process not only reduces computational costs but also improves the overall efficiency of the hallucination mitigation process.

##### **B. Maintaining Truthfulness**

Benefits of KG Integration Despite the reduced retrieval efforts, Re-KGR maintains the benefits of integrating external knowledge from KGs to enhance the truthfulness and factual accuracy of LLM outputs. By verifying and rectifying the identified critical triples against the structured knowledge in the KG, Re-KGR ensures that the final generated content aligns with established facts and domain-specific knowledge, effectively mitigating hallucinations.

##### **C. No External Training Requirements**

Unlike some existing KG-augmented approaches that require supervised training of specialized models or fine-tuning on specific datasets, Re-KGR does not necessitate any external training processes. The approach leverages the LLM’s internal probability distributions and semantic similarity models to identify critical entities and verify knowledge triples. This characteristic makes Re-KGR relatively easy to implement and adapt to different LLMs or knowledge domains, as long as a well-constructed domain-specific KG is available.

#### **V. Experiments and Results**

To evaluate the effectiveness of the Re-KGR approach, the researchers conducted experiments on a medical question-answering dataset called MedQuAD \[4, 18\]. They employed the LLaMA-7b \[32\] model, a powerful LLM pre-trained on trillions of tokens, as the foundational model for their experiments. Additionally, they compared the performance of Re-KGR with a baseline approach using **contrastive decoding**, a technique designed to reduce hallucinations by contrasting output distributions across different layers within the LLM, known as DoLa \[10\].

##### **A. Datasets and Evaluation Setup**

The MedQuAD dataset consists of real-world medical question-answer pairs compiled from various National Institutes of Health websites. It covers a wide range of medical topics, including treatments, diagnoses, side effects, and various medical entities such as diseases, medications, and diagnostic tests. For the experiments, the researchers filtered and preprocessed the dataset to focus on question-answer pairs relevant to the constructed medical knowledge graph.  
To assess the truthfulness and factual accuracy of the generated responses, the researchers employed GPT-4 \[1\], a state-of-the-art language model, as an automated evaluator. Leveraging GPT-4’s extensive knowledge base and language understanding capabilities, they designed prompts to instruct the model to evaluate each response against the given standard answer and established medical knowledge. The evaluation scores ranged from 0 to 1, with 0 indicating an irrelevant or factually incorrect response, and 1 representing perfect alignment with the ground-truth information.

##### **B. Performance Compared to Baselines**

The experimental results, as presented in Table 1 of the study \[1\], demonstrate that the Re-KGR approach outperformed the baselines in terms of truthfulness scores assessed by GPT-4. Specifically, when integrated with the DoLa model, Re-KGR achieved the highest truthfulness performance with a score of 0.610, indicating a 15.75% improvement over the vanilla LLaMA-7b model and a 3.21% improvement compared to the contrastive decoding baseline (DoLa) without KG integration.  
However, when incorporated into the vanilla LLaMA-7b model, the improvement achieved by Re-KGR was more modest, with a 1.90% increase in truthfulness score compared to the baseline. The authors attribute this underperformance to the intrinsic limitations of the vanilla LLaMA-7b model, which may generate brief, repetitive, or irrelevant responses, reducing the need for verification and limiting the potential benefits of KG integration.

##### **C. Ablation Studies on Different Criteria for Entity Detection**

The researchers conducted additional experiments to evaluate the performance of various criteria used for identifying critical entities and tokens prone to hallucinations. As discussed in Section IV.A, Re-KGR employs three main criteria: the max value of the next-token predictive probability (c\_m), the entropy of the next-token predictive probability (c\_e), and the output divergence between the final layer and intermediate layers (c\_js).  
The results, presented in Table 2 of the study \[1\], reveal that the c\_js-based criterion, which relies on the output divergence across layers, was the most efficient for critical entity detection while requiring the least retrieval efforts. However, the c\_e-based criterion, which utilizes the entropy of the next-token probability distribution, demonstrated superior performance in the overall hallucination mitigation task, achieving the highest truthfulness score as assessed by GPT-4.

##### **D. Analysis of Retrieval Time Savings with Triple Refinement**

One of the key advantages of the Re-KGR approach is its ability to reduce the computational costs associated with knowledge retrieval by refining the set of triples that need to be verified against the KG. To quantify this benefit, the researchers analyzed the average retrieval time per question and compared it across different configurations, as shown in Table 3 of the study \[1\].  
The results indicate that the Re-KGR method, with its triple refinement process, achieved a reduction in retrieval time of 63% compared to retrieving all triples without refinement when integrated with the vanilla LLaMA-7b model. When combined with the DoLa model and the c\_js-based criterion for entity detection, the retrieval time savings were even more substantial, with a 75% reduction compared to the baseline without refinement.  
Notably, despite the significant reduction in retrieval time, the Re-KGR approach maintained or even improved the truthfulness scores evaluated by GPT-4, as evidenced by the results in Table 1 \[1\]. This finding demonstrates that the strategic retrieval of critical knowledge components through triple refinement can effectively mitigate hallucinations while simultaneously enhancing computational efficiency.

#### **VI. Potential Limitations and Future Directions**

While the Re-KGR approach presents a promising solution for mitigating hallucinations in LLMs through the integration of knowledge graphs, there are several potential limitations and avenues for future exploration:

##### **A. Dependence on Domain-Specific Knowledge Graph Quality and Coverage**

The effectiveness of Re-KGR heavily relies on the quality and comprehensiveness of the domain-specific knowledge graph used for retrieval and verification. If the KG A. Dependence on Domain-Specific Knowledge Graph Quality and Coverage (continued) The effectiveness of Re-KGR heavily relies on the quality and comprehensiveness of the domain-specific knowledge graph used for retrieval and verification. If the KG itself contains incomplete, outdated, or inaccurate information, the approach may inadvertently propagate these errors into the generated outputs. Additionally, the coverage of the KG in terms of the breadth of topics and entities represented can limit the applicability of Re-KGR to certain domains or scenarios. Addressing these limitations may involve ongoing curation and expansion of the knowledge graphs, as well as exploring methods to automatically assess and enhance the quality and coverage of these structured knowledge bases.

##### **B. Challenges in Multi-Sentence Reasoning and Coherent Knowledge Infusion**

The Re-KGR approach, as presented in the study, primarily focuses on verifying and rectifying individual knowledge triples extracted from the LLM’s output. However, real-world language generation tasks often involve multi-sentence responses and multi-hop reasoning, where the truthfulness and coherence of the generated content depend on the relationships and dependencies between multiple factual statements. Extending Re-KGR to handle such complex scenarios may require additional mechanisms for contextual understanding, logical reasoning, and coherent integration of retrieved knowledge into the generated text.

##### **C. Exploring Extensions to Other Knowledge-Intensive Domains**

While the experiments in the study focused on the medical domain, the principles of the Re-KGR approach are potentially applicable to other knowledge-intensive domains where factual accuracy is crucial, such as scientific literature, news reporting, product information, and legal documents. However, the availability and quality of domain-specific knowledge graphs may vary across different fields, posing challenges for the direct application of Re-KGR. Future research could explore the construction and curation of high-quality knowledge graphs for diverse domains, as well as any necessary adaptations or extensions to the Re-KGR methodology to accommodate domain-specific characteristics and requirements.

##### **D. Improving Human-AI Interaction for Knowledge Validation**

While Re-KGR primarily relies on automated methods for entity detection, knowledge retrieval, and verification, there is potential for incorporating human-computer interaction to further refine and validate the knowledge injection process. For instance, interactive interfaces could allow human experts to review and provide feedback on the identified critical entities, the retrieved knowledge triples, or the final rectified outputs. Such human-in-the-loop approaches could enhance the reliability and trustworthiness of the generated content, particularly in high-stakes domains where factual accuracy is paramount.

#### **VII. Conclusion**

In the rapidly evolving field of large language models, the issue of hallucinations, where generated outputs deviate from factual knowledge, poses a significant challenge to the widespread adoption and trust in these powerful AI systems. The Self-Refinement Enhanced Knowledge Graph Retrieval (Re-KGR) approach presents a novel solution to mitigate hallucinations by strategically retrieving and integrating relevant knowledge from curated knowledge graphs.

The key innovation of Re-KGR lies in its ability to i**dentify critical entities or tokens within the LLM’s output** that are most likely to be prone to hallucinations. By leveraging the attribution of predictive probability distributions and divergence between output layers, Re-KGR can efficiently extract and refine a set of knowledge triples associated with these critical components. This refined set is then used to retrieve corresponding factual information from domain-specific knowledge graphs, which is subsequently verified and used to rectify any inaccuracies in the original output.

Experimental results on a medical question-answering dataset demonstrate the effectiveness of Re-KGR in enhancing the truthfulness and factual accuracy of LLM responses. When integrated with the **contrastive decoding technique DoLa** \[10\], Re-KGR achieved a 15.75% improvement in truthfulness scores compared to the baseline LLaMA-7b model without KG integration. Moreover, the strategic retrieval approach employed by Re-KGR led to substantial reductions in retrieval time, up to 75% compared to naive retrieval methods, while maintaining or improving truthfulness performance.

While the Re-KGR approach shows promising results, there are potential limitations and avenues for future exploration. These include addressing the dependence on the quality and coverage of domain-specific knowledge graphs, extending the approach to handle multi-sentence reasoning and coherent knowledge infusion, and exploring adaptations to accommodate diverse knowledge-intensive domains beyond healthcare. Additionally, incorporating human-computer interaction and feedback loops could further enhance the reliability and trustworthiness of the generated content, particularly in high-stakes applications.  
As the adoption of LLMs continues to grow across various industries and domains, the development of effective techniques for mitigating hallucinations and ensuring factual accuracy becomes increasingly crucial. The Re-KGR approach represents a significant step forward in this direction, harnessing the power of structured knowledge graphs to ground LLM outputs in established facts and domain-specific knowledge.

However, it is important to recognize that Re-KGR is not a panacea for all hallucination challenges. Rather, it should be viewed as a valuable component within a broader ecosystem of complementary techniques, including data and prompt construction, model re-training, and human oversight. By combining Re-KGR with these other approaches, researchers and practitioners can work towards building more reliable and trustworthy AI systems that can generate truthful and factual outputs while minimizing the risks associated with hallucinations.

As the field of natural language processing continues to evolve, the integration of structured knowledge sources and the development of truth-grounding techniques will play a pivotal role in unlocking the full potential of LLMs across diverse applications, from creative writing and content generation to question answering and decision support systems. The Re-KGR approach represents a promising framework for achieving this goal, paving the way for a future where AI-generated language is not only human-like but also firmly rooted in factual knowledge and trusted information sources.

### **GraphRAG**

[https://neo4j.com/blog/graphrag-manifesto/?utm\_source=LinkedIn\&utm\_medium=SocialInfluencer\&utm\_campaign=---\&utm\_ID=\&utm\_term=\&utm\_content=-DevBlog--\&utm\_creative\_format=\&utm\_marketing\_tactic=\&utm\_parent\_camp=\&utm\_partner=\&utm\_persona=](https://neo4j.com/blog/graphrag-manifesto/?utm_source=LinkedIn&utm_medium=SocialInfluencer&utm_campaign=---&utm_ID=&utm_term=&utm_content=-DevBlog--&utm_creative_format=&utm_marketing_tactic=&utm_parent_camp=&utm_partner=&utm_persona=) 

**Linkedin** : [https://www.linkedin.com/in/ii-tae-jeong/](https://www.linkedin.com/in/ii-tae-jeong/) 

**G-Retriever:** Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering  
**PaperLink:** [https://arxiv.org/pdf/2402.07630.pdf](https://arxiv.org/pdf/2402.07630.pdf)   
**CodeLink:** [https://github.com/XiaoxinHe/G-Retriever](https://github.com/XiaoxinHe/G-Retriever) 

**Keywords**: Graph RAG , Retrieval-augmented generation, Large Language Model, hallucination, Graph Neural Network

Before we discuss GraphRAG and review this paper, we would like to know the GraphRAG concept.  
Utilizing **graph embeddings** from **GNN** (graph neural networks) for user query response inference, this approach adds graph embeddings to text embeddings. Termed soft-prompting, it is a form of prompt engineering.  
Prompt engineering can be broadly categorized into Hard and Soft. **Hard prompts** are **explicit**, where context is manually added to the given user query. For example, if a user query is “I want to eat bread today,” a hard prompt might explicitly outline the task, context, persona, example, format, and tone, requiring input on six dimensions. This method is subjective, with the prompt creator’s bias heavily influencing its optimization. However, its simplicity is advantageous.  
Conversely, Soft prompts are implicit, enhancing existing text embeddings with additional embedding information as the model infers answers similar to the query. This method ensures objectivity and optimizes weight values but requires more complex model design and implementation.

#### **When to Use GraphRAG**

GraphRAG isn’t a one-size-fits-all solution. If the existing RAG works well, switching to the more advanced GraphRAG without a compelling reason may not be well-received. Any system improvement requires justification to answer why it’s necessary.  
GraphRAG is worth considering when there’s a mismatch between retrieved information and user intent, a fundamental limitation similar to vector search. Since retrieval is based on similarity rather than exact matches, it may yield inaccurate information.  
Improvements might involve introducing BM25 for an exact search in a hybrid search approach, enhancing the ranking process with re-ranker functions, or fine-tuning to improve embedding quality. If these efforts result in minimal RAG performance improvements, considering GraphRAG is advisable.

#### **How does the G-Retriever work?**

1. ##### **Indexing**

This passage describes the process of refining and storing data in a format that is readily accessible for use beforehand. In GraphRAG, the information to be utilized in advance refers to the textual information contained within the properties of nodes and edges in the graph. To convert this information into quantifiable values, a language model is employed.

2. **Retrieval**

This passage discusses the process of measuring and retrieving data based on its relevance to a user’s query. To assess the relevance, the language model evaluates the similarity between the ‘query’ and the values of ‘nodes’ and ‘edges’ within the graph, utilizing the K-nearest neighbors algorithm for this purpose.

3. ##### **Subgraph Construction**

Unlike other RAG (Retrieval-Augmented Generation) models that retrieve documents, GraphRAG needs to fetch graphs relevant to the user’s query. In the initial retrieval process, simply comparing the user’s text with the graph’s text to fetch information does not strictly utilize the semantics of the graph’s connections.  
To leverage this, it’s necessary to assess **how much semantic similarity each node and edge has** with the user’s query. For this assessment, the **PCST (Prize-Collecting Steiner Tree)** approach is utilized.  
**Briefly explaining the PCST approach:** both nodes and edges are assigned prizes. The value of these prizes is determined by using the ranking similarities of nodes and edges to the user’s query, identified in the earlier retrieval process. Higher prizes are awarded to nodes similar to the query, while dissimilar nodes may receive lower values or even zero.  
The prizes are summed across connected nodes and edges, extracting those with high total values. This total represents the nodes and edges with the highest sums. To manage the size of the subgraph, a parameter called ‘cost’ is used to pre-determine how much penalty to assign to each edge, effectively controlling the subgraph size.  
Ultimately, this process extracts subgraphs containing information similar to the user’s query, while also managing the subgraph size through the cost parameter.

4. **Answer generation**

This passage describes the process of generating answers to queries by combining text embedding values with graph embedding values. Here, text embeddings refer to the values from the self-attention layers of a pre-trained Language Learning Model (LLM) that are kept frozen, meaning their weights are not updated during training.  
By utilizing graph embedding values for training, it takes advantage of the soft-prompt technique mentioned earlier, which involves extracting and updating optimized weight values to incorporate semantics into answer generation.  
The method for deriving graph embedding values and combining them with text embedding values is straightforward:  
1\.  Generate node embeddings using a Graph Neural Network (**GNN**).  
2\.  Aggregate these values using a pooling layer.  
3\.  To align the dimensions of the pooled graph embedding values with the text embedding values, project them through a Multi-Layer Perceptron (MLP) layer.

This process underscores the synergy between text and graph embeddings in enhancing the semantic richness of the generated answers, leveraging the strengths of both pre-trained models and graph-based information.

#### **G-Retrieval Insight**

1. ##### **Efficiency Retrieval**

I think that the criteria may vary. In this paper, we discuss efficiency in terms of the comparison before and after retrieval, based on how much the usage of tokens is saved.

One of the key aspects of RAG (Retrieval-Augmented Generation) is the emphasis on including the most optimal information within the given token capacity. When utilizing G-Retrieval, there is a remarkable effect observed, where the amount of tokens decreases significantly, ranging from 83% to 99%.

2. ##### **Architecture**

To demonstrate the effectiveness of G-Retriever, we conduct comparative experiments across three different architectures: 1\. Architecture utilizing only pretrained weights, 2\. Architecture that employs both pretrained weights and prompt engineering, 3\. Architecture leveraging finetuned weights along with prompt engineering. Each of these architectures has its own implications.  
The first architecture aims to determine the significance of the textual graph. The second architecture is designed to explore the meaningfulness of soft-prompts through the use of a Graph Encoder and projection. Finally, the third architecture seeks to understand the importance of optimizing LLM (Language Learning Model) weights independently of the Graph.

3. **Performance**

The results of the ablation study are also interesting. In particular, it can be observed that performance decreases by nearly 13% in parts related to the graph, specifically in scenarios without Edge Retrieval. This suggests that Edge, or in other words, Semantic Retrieval, plays a critical role in the RAG (Retrieval-Augmented Generation) framework.  
In the end, what we must keep in mind in GraphRAG  
Retrieving a graph is important, but how the entire graph is designed is equally crucial. In this idea, we only showcased retrieval using a benchmark dataset for knowledge graphs, omitting the story behind the graph’s construction.  
With this in mind, we recommend proceeding with the task while maintaining fundamental questions about how nodes were created, how edges were formed, and why semantics were set in a particular way.

#### **Core concept code with explanation**

`def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):`  
    `c = 0.01`  
    `if len(textual_nodes) == 0 or len(textual_edges) == 0:`  
        `desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])`  
        `graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)`  
        `return graph, desc`

    `root = -1  # unrooted`  
    `num_clusters = 1`  
    `pruning = 'gw'`  
    `verbosity_level = 0`  
    `if topk > 0:`  
        `n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)`  
        `topk = min(topk, graph.num_nodes)`  
        `_, topk_n_indices = torch.topk(n_prizes, topk, largest=True)`

        `n_prizes = torch.zeros_like(n_prizes)`  
        `n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()`  
    `else:`  
        `n_prizes = torch.zeros(graph.num_nodes)`

    `if topk_e > 0:`  
        `e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)`  
        `topk_e = min(topk_e, e_prizes.unique().size(0))`

        `topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)`  
        `e_prizes[e_prizes < topk_e_values[-1]] = 0.0`  
        `last_topk_e_value = topk_e`  
        `for k in range(topk_e):`  
            `indices = e_prizes == topk_e_values[k]`  
            `value = min((topk_e-k)/sum(indices), last_topk_e_value-c)`  
            `e_prizes[indices] = value`  
            `last_topk_e_value = value`  
        `# cost_e = max(min(cost_e, e_prizes.max().item()-c), 0)`  
    `else:`  
        `e_prizes = torch.zeros(graph.num_edges)`

    `costs = []`  
    `edges = []`  
    `vritual_n_prizes = []`  
    `virtual_edges = []`  
    `virtual_costs = []`  
    `mapping_n = {}`  
    `mapping_e = {}`  
    `for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):`  
        `prize_e = e_prizes[i]`  
        `if prize_e <= cost_e:`  
            `mapping_e[len(edges)] = i`  
            `edges.append((src, dst))`  
            `costs.append(cost_e - prize_e)`  
        `else:`  
            `virtual_node_id = graph.num_nodes + len(vritual_n_prizes)`  
            `mapping_n[virtual_node_id] = i`  
            `virtual_edges.append((src, virtual_node_id))`  
            `virtual_edges.append((virtual_node_id, dst))`  
            `virtual_costs.append(0)`  
            `virtual_costs.append(0)`  
            `vritual_n_prizes.append(prize_e - cost_e)`

    `prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])`  
    `num_edges = len(edges)`  
    `if len(virtual_costs) > 0:`  
        `costs = np.array(costs+virtual_costs)`  
        `edges = np.array(edges+virtual_edges)`

    `vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)`

    `selected_nodes = vertices[vertices < graph.num_nodes]`  
    `selected_edges = [mapping_e[e] for e in edges if e < num_edges]`  
    `virtual_vertices = vertices[vertices >= graph.num_nodes]`  
    `if len(virtual_vertices) > 0:`  
        `virtual_vertices = vertices[vertices >= graph.num_nodes]`  
        `virtual_edges = [mapping_n[i] for i in virtual_vertices]`  
        `selected_edges = np.array(selected_edges+virtual_edges)`

    `edge_index = graph.edge_index[:, selected_edges]`  
    `selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))`

    `n = textual_nodes.iloc[selected_nodes]`  
    `e = textual_edges.iloc[selected_edges]`  
    `desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])`

    `mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}`

    `x = graph.x[selected_nodes]`  
    `edge_attr = graph.edge_attr[selected_edges]`  
    `src = [mapping[i] for i in edge_index[0].tolist()]`  
    `dst = [mapping[i] for i in edge_index[1].tolist()]`  
    `edge_index = torch.LongTensor([src, dst])`  
    `data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))`

    return data, desc  
 **Original Code Resource** : [https://github.com/XiaoxinHe/G-Retriever](https://github.com/XiaoxinHe/G-Retriever) 

The provided code outlines a function designed to perform subgraph extraction based on the Prize-Collecting Steiner Tree (PCST) approach. The idea is to select a subset of nodes and edges from a given graph that are most relevant to a specific query embedding (\`q\_emb\`). This method is particularly useful in scenarios where the graph represents textual data, and you’re interested in extracting a coherent and relevant subgraph based on semantic similarity to a query.  
Let’s break down the key parts of the function for better understanding:

Function Parameters:  
\- \`graph\`: The original graph from which the subgraph is to be extracted. Expected to be a PyTorch Geometric \`Data\` object.  
\- \`q\_emb\`: The query embedding vector representing the query’s semantic content.  
\- \`textual\_nodes\`, \`textual\_edges\`: Pandas DataFrames containing information about the nodes and edges of the \`graph\`.  
\- \`topk\`, \`topk\_e\`: Parameters specifying the number of top nodes and edges to consider based on similarity to \`q\_emb\`.  
\- \`cost\_e\`: A threshold cost for including edges in the solution.

Key Steps Explained:  
1\. \*\*Early Return for Empty Graph Components\*\*: If there are no textual nodes or edges, it immediately returns the original graph and a description derived from \`textual\_nodes\` and \`textual\_edges\`.

2\. \*\*Initialization\*\*: Sets up variables for PCST including the root (unrooted in this case), number of clusters, and pruning method.

3\. \*\*Node and Edge Prize Calculation\*\*:  
— Calculates similarity scores (\`n\_prizes\` for nodes, \`e\_prizes\` for edges) between the query embedding and graph components using cosine similarity.  
— Adjusts these scores to determine the “prizes” for including each node or edge in the subgraph. For edges, it further filters them based on the \`cost\_e\` threshold.

4\. \*\*Graph Transformation for PCST\*\*:  
— Transforms the original graph into a format suitable for PCST by potentially introducing “virtual” nodes and adjusting edges and their costs based on the computed prizes and costs.

5\. \*\*PCST Algorithm Execution\*\*:  
— Runs the PCST algorithm (\`pcst\_fast\`) on the transformed graph to select a subset of nodes and edges that form the optimal subgraph based on the given prizes and costs.

6\. \*\*Subgraph Reconstruction\*\*:  
— Extracts the selected nodes and edges based on the output of the PCST algorithm.  
— Reconstructs the subgraph using the selected components, ensuring that the resulting subgraph is connected and relevant to the query.

7\. \*\*Subgraph Description Generation\*\*:  
— Generates a textual description of the selected subgraph by converting the relevant parts of \`textual\_nodes\` and \`textual\_edges\` DataFrames to CSV format.

8\. \*\*Return\*\*: The function returns the reconstructed subgraph as a PyTorch Geometric \`Data\` object along with its textual description.

\#\#\# Annotations for Clarity:  
\- \*\*Prize Calculation\*\*: The prizes for nodes and edges are derived from their semantic relevance to the query. Higher similarity scores lead to higher prizes, indicating a stronger preference for including these components in the subgraph.

\- \*\*Virtual Nodes and Edges\*\*: Introduced to facilitate the PCST algorithm. They represent potential modifications to the original graph’s structure to accommodate the prize and cost model. Virtual nodes act as intermediaries, adjusting the connectivity based on the optimization process.

\- \*\*PCST Algorithm\*\*: The core of the function, \`pcst\_fast\`, is an external algorithm that takes the transformed graph (with prizes and costs) and identifies the optimal subgraph. This step is where the actual optimization occurs.

\- \*\*Mapping and Reconstruction\*\*: After identifying the optimal components, the function maps them back to the original graph’s context, ensuring that the resulting subgraph is accurately represented and relevant to the query.

This function encapsulates a complex process of graph optimization tailored to extracting semantically relevant subgraphs based on the PCST model, making it a powerful tool for tasks like document summarization, information retrieval, and knowledge graph exploration.

### Summary

In summary, we have discussed the potential of Knowledge Graphs and large language models (LLMs) as knowledge bases. Knowledge Graphs excel in capturing relationships and have greater reasoning abilities, but are difficult and costly to construct. On the other hand, LLMs contain extensive knowledge but are prone to bias, hallucinations and other issues. They are also computationally expensive to fine-tune or to adapt for specific domains. To harness the benefits of both methods, knowledge graphs and LLMs can be integrated together in several ways.  
In this article, we focused on using LLMs to assist in automatic knowledge graph construction. In particular, we reviewed four examples, including the earlier COMET model, using ChatGPT as an information extractor in BEAR, and directly harvesting knowledge from LLMs. These methods represent a promising path forward in combining the strengths of knowledge graphs and LLMs to enhance knowledge representation.

Knowledge Graph Presentations

* [https://hpi.de/fileadmin/user\_upload/fachgebiete/naumann/lehre/SS2022/KGs\_and\_LMs\_01\_Introduction.pdf](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/lehre/SS2022/KGs_and_LMs_01_Introduction.pdf)  
* [https://www.slideshare.net/slideshow/graphrag-is-all-you-need-llm-knowledge-graph/269450550](https://www.slideshare.net/slideshow/graphrag-is-all-you-need-llm-knowledge-graph/269450550)   
* [https://www.linkedin.com/pulse/what-knowledge-graphs-your-gateway-understanding-linked-hammad-munir-8levf/?trackingId=KgOeB8%2FOTEmOI5YylfkY6g%3D%3D](https://www.linkedin.com/pulse/what-knowledge-graphs-your-gateway-understanding-linked-hammad-munir-8levf/?trackingId=KgOeB8%2FOTEmOI5YylfkY6g%3D%3D) 

### References

* What is a Knowledge Graph? | IBM. (n.d.). [Www.ibm.com](http://www.ibm.com/). [https://www.ibm.com/topics/knowledge-graph](https://www.ibm.com/topics/knowledge-graph)  
* ‌Yang, L., Chen, H., Li, Z., Ding, X., & Wu, X. (2023). Give Us the Facts: Enhancing Large Language Models with Knowledge Graphs for Fact-aware Language Modeling (Version 2). arXiv. [https://doi.org/10.48550/ARXIV.2306.11489](https://doi.org/10.48550/ARXIV.2306.11489)  
* Feng, C., Zhang, X., & Fei, Z. (2023). Knowledge Solver: Teaching LLMs to Search for Domain Knowledge from Knowledge Graphs (Version 1). arXiv. [https://doi.org/10.48550/ARXIV.2309.03118](https://doi.org/10.48550/ARXIV.2309.03118)  
* Bosselut, A., Rashkin, H., Sap, M., Malaviya, C., Celikyilmaz, A., & Choi, Y. (2019). COMET: Commonsense Transformers for Automatic Knowledge Graph Construction (Version 2). arXiv. [https://doi.org/10.48550/ARXIV.1906.05317](https://doi.org/10.48550/ARXIV.1906.05317)  
* Yu, S., Huang, T., Liu, M., & Wang, Z. (2023). BEAR: Revolutionizing Service Domain Knowledge Graph Construction with LLM. In Service-Oriented Computing (pp. 339–346). Springer Nature Switzerland. [https://doi.org/10.1007/978-3-031-48421-6\_23](https://doi.org/10.1007/978-3-031-48421-6_23)  
* Kommineni, V. K., König-Ries, B., & Samuel, S. (2024). From human experts to machines: An LLM supported approach to ontology and knowledge graph construction (Version 1). arXiv. [https://doi.org/10.48550/ARXIV.2403.08345](https://doi.org/10.48550/ARXIV.2403.08345)  
* Hao, S., Tan, B., Tang, K., Ni, B., Shao, X., Zhang, H., Xing, E. P., & Hu, Z. (2022). BertNet: Harvesting Knowledge Graphs with Arbitrary Relations from Pretrained Language Models (Version 3). arXiv. [https://doi.org/10.48550/ARXIV.2206.14268](https://doi.org/10.48550/ARXIV.2206.14268)  
* [https://neo4j.com/developer-blog/construct-knowledge-graphs-unstructured-text/](https://neo4j.com/developer-blog/construct-knowledge-graphs-unstructured-text/) 

### Interesting Readings

* [https://aaai.org/wp-content/uploads/2024/02/AAAI-24\_Main\_2024-02-01.pdf](https://aaai.org/wp-content/uploads/2024/02/AAAI-24_Main_2024-02-01.pdf)   
* MKG-FENN: A Multimodal Knowledge Graph Fused End-to-End Neural Network for Accurate Drug–Drug Interaction Prediction [https://ojs.aaai.org/index.php/AAAI/article/view/28887](https://ojs.aaai.org/index.php/AAAI/article/view/28887)   
* Multimodal knowledge graph [https://github.com/pengfei-luo/multimodal-knowledge-graph](https://github.com/pengfei-luo/multimodal-knowledge-graph)   
* KG-MM-Survey [https://github.com/zjukg/KG-MM-Survey?tab=readme-ov-file\#mmkg-construction-methods](https://github.com/zjukg/KG-MM-Survey?tab=readme-ov-file#mmkg-construction-methods)   
* VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks for Visual Question Answering [https://arxiv.org/abs/2205.11501](https://arxiv.org/abs/2205.11501) (published in iccv)  
* [https://www.youtube.com/watch?v=fKOPBVTCn-Y](https://www.youtube.com/watch?v=fKOPBVTCn-Y)   
* [https://www.youtube.com/@johntanchongmin/videos](https://www.youtube.com/@johntanchongmin/videos)   
* Lab: [https://sites.google.com/view/chaoh](https://sites.google.com/view/chaoh) 

