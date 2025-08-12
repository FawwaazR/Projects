# Import relevant libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import nltk
from nltk.corpus import stopwords
nltk.download ("stopwords")
stop_words = set (stopwords.words("english"))
from collections import defaultdict
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.patches as mpatches
from sklearn.decomposition import TruncatedSVD
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from bertopic import BERTopic
import plotly.io as pio
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoConfig
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score, recall_score, precision_score
from tqdm import tqdm

class HateSpeechClassifier:
    """
    This class provides a unified pipeline for:
    - Cleaning and preprocessing raw text data.
    - Performing topic modeling using one of three methods:
        - Latent Semantic Analysis (LSA),
        - Latent Dirichlet Allocation (LDA),
        - BERTopic.
    - Classifying hate speech using a fine-tuned BERTweet model.

    Attributes:
    - df (Pandas DataFrame): Should contain social media posts and binary labels with 1: Hate Speech, 0: Non-Hate Speech.
    - text (str): Name of column where social media posts are stored.
    - label (str): Name of column where labels are stored.

    Methods:
    - normalise: Performs case folding and punctuation removal and updates self.df inplace.
    - remove_stopwords: Removes all stopwords from social media posts inplace.
    - lemmatise: Performs lemmatisation for social media posts inplace to reduce words to their root form.
    - tfidf: Conducts vectorisation via term frequency inverse document frequency (TF-IDF).
    - LSA: Conducts Latent Semantic Analysis using truncated Singular Value Decomposition with K=2 and results are plotted in a 2-dimensional graph.
    - LDA: Conducts Latent Dirichlet Allocation using coherence score to determine optimal number of topics to be generated. Results are visualised using PyLDAvis thorugh an Intertopic Distance Map and Top 30 most relevant words per topic (Interactive interface).
    - BERTopic: Uses BERT to perform topic modelling. Outputs the BERTopic model. Results can be accessed by (Assume class instance is HateClassifier):
        >>> HateClassifier.BERTopic().visualize_topics() (Intertopic Distance Map)
        >>> HateClassifier.BERTopic().visualize_barchart(top_n_topics = n) (Visualise word scores for top n topics)
        >>> HateClassifier.BERTopic().visualize_heatmap() (Topic similarity matrix)
        >>> HateClassifier.BERTopic().visualise_hierarchy() (Topic dendogram)
    - train_BERTweet: trains the hate speech classifier, a BERTweet model with a learning rate scheduler, 10% warmup, 50% dropout, weight decay and trained over 2 epochs. Returns the model.
    - BERTweet_classify: used BERTweet model to predict labels of social media posts and evaluates performance through a classification report.
    """
    def __init__(self, data, text_col, label_col):
        self.df = data.copy()
        self.text = text_col
        self.label = label_col

    def normalise(self):
        # Case Folding
        self.df[self.text] = self.df[self.text].str.lower()

        # Punctuation removal
        minus_puntuation = []
        for post in self.df[self.text]:
            minus_puntuation.append(re.sub(r"[^\w\s]", "", post))  
        self.df[self.text] = minus_puntuation
    
    def remove_stopwords(self):
        stop_words = set(stopwords.words("english"))
        minus_stopwords = []
        for post in self.df[self.text]:
            new_post = " ".join([word for word in post.split() if word not in stop_words])
            minus_stopwords.append(new_post)
        self.df[self.text] = minus_stopwords
    

    def lemmatise(self):
        lemmatiser = WordNetLemmatizer()
        tweet_lemma = []
        for post in self.df[self.text]:
            tweet_lemma.append(" ".join(lemmatiser.lemmatize(word) for word in post.split()))
        self.df[self.text] = tweet_lemma
    
    def tfidf(self, corpus):
        tfidf_vectorizer = TfidfVectorizer()
        output = tfidf_vectorizer.fit_transform(corpus)
        return output, tfidf_vectorizer

    def LSA(self):
        # Convert text and labels to list
        text_corpus = self.df[self.text].tolist()
        list_labels = self.df[self.label].tolist()
        
        # Fit LSA
        LSA = TruncatedSVD(n_components=2)
        LSA_scores = LSA.fit_transform(text_corpus)
        
        # Plot results
        color_mapper = {label:idx for idx, label in enumerate(set(list_labels))}
        color_column = [color_mapper[label] for label in list_labels]
        colors = ["orange", "blue"]

        plt.scatter(LSA_scores[:,0], LSA_scores[:,1], s=8, alpha=0.8, c=list_labels, cmap=matplotlib.colors.ListedColorMap(colors))
        orange_patch = mpatches.Patch(color="orange", label="Non-Hate Speech")
        blue_patch = mpatches.Patch(color="blue", label="Hate Speech")
        plt.legend(handles=[orange_patch, blue_patch], prop={"size": 30})
    
    def LDA(self, min_T, max_T):
        # Vectorising posts and extracting corpus for hate and non-hate posts
        h_posts = self.df[self.label == 1].text.tolist()
        h_posts = [post.split(",")[0].split() for post in h_posts]
        h_id2word = Dictionary(h_posts)
        h_corpus = [h_id2word.doc2bow(post) for post in h_posts]

        n_h_posts = self.df[self.label == 0].text.tolist()
        n_h_posts = [post.split(",")[0].split() for post in n_h_posts]
        n_h_id2word = Dictionary(n_h_posts)
        n_h_corpus = [n_h_id2word.doc2bow(post) for post in n_h_posts]

        # Calculating coherence score to determine optimal number of topics
        score = float("-inf")
        optimal_T = 0
        for i in range(min_T, max_T+1):
            n_h_lda_model = LdaModel(corpus=n_h_corpus,
                        id2word=n_h_id2word,
                        num_topics=i,
                        random_state=0,
                        chunksize=100,
                        alpha='auto',
                        per_word_topics=True)
            coherence_model_cv = CoherenceModel(model=n_h_lda_model, texts=n_h_posts, dictionary=n_h_id2word, coherence='c_v')
            if coherence_model_cv.get_coherence() > score:
                score = coherence_model_cv.get_coherence()
                optimal_T = i
        
        # Fit and visualise LDA model with optimal number of topics
        h_lda_model = LdaModel(corpus=h_corpus,
                    id2word=h_id2word,
                    num_topics=optimal_T,
                    random_state=0,
                    chunksize=100,
                    alpha='auto',
                    per_word_topics=True)
        vis = gensimvis.prepare(h_lda_model, h_corpus, h_id2word)
        pyLDAvis.display(vis)

    def BERTopic(self):
        hate_df = self.df[self.label == 1]
        hate_corpus_2 = hate_df.text.tolist()

        hate_BERT = BERTopic()
        hate_topic, hate_probs = hate_BERT.fit_transform(hate_corpus_2)

        return  hate_BERT

    def _calculate_metrics(self, preds, labels):
        results = dict()
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        results['precision_score'] = precision_score(labels_flat, preds_flat, average='binary')
        results['recall_score'] = recall_score(labels_flat, preds_flat, average='binary')
        results['f1_score'] = f1_score(labels_flat, preds_flat, average='binary')
        return results
    
    def _encode_data(self, df, tokenizer):
        input_ids = []
        attention_masks = []
        for tweet in df[["test_case"]].values:
            tweet = tweet.item()
            encoded_data = tokenizer.encode_plus(
                                tweet,                      
                                add_special_tokens = True,  
                                max_length = 128,
                                padding = 'max_length',
                                truncation = True,
                                return_attention_mask = True,   
                                return_tensors = 'pt',    
                        )
            input_ids.append(encoded_data['input_ids'])
            attention_masks.append(encoded_data['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        inputs = {
        'input_ids': input_ids,
        'input_mask': attention_masks}
        return inputs
    
    def _prepare_dataloaders(self, train_df, dev_df, test_df, model_name, batch_size):
        tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False, normalization=True)

        data_train = self.encode_data(train_df, tokenizer)
        labels_train = train_df.Labels.astype(int)

        data_valid = self.encode_data(dev_df, tokenizer)
        labels_valid = dev_df.Labels.astype(int)

        data_test = self.encode_data(test_df, tokenizer)
        labels_test = test_df.Labels.astype(int)

        input_ids, attention_masks = data_train.values()
        train_labels = torch.tensor(labels_train.values)
        train_dataset = TensorDataset(input_ids, attention_masks, train_labels)

        input_ids, attention_masks = data_valid.values()
        valid_labels = torch.tensor(labels_valid.values)
        val_dataset = TensorDataset(input_ids, attention_masks, valid_labels)

        input_ids, attention_masks = data_test.values()
        test_labels = torch.tensor(labels_test.values)
        test_dataset = TensorDataset(input_ids, attention_masks, test_labels)

        train_dataloader = DataLoader(
                    train_dataset,
                    sampler = RandomSampler(train_dataset), 
                    batch_size = batch_size 
                )

        validation_dataloader = DataLoader(
                    val_dataset, 
                    sampler = SequentialSampler(val_dataset),
                    batch_size = batch_size 
                )

        test_dataloader = DataLoader(
                    test_dataset, 
                    sampler = SequentialSampler(test_dataset), 
                    batch_size = batch_size
                )

        return train_dataloader, validation_dataloader, test_dataloader
    
    def _prepare_model(self, total_labels, model_name, model_to_load=None):
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=total_labels,
            hidden_dropout_prob=0.5,               # 50% dropout
            attention_probs_dropout_prob=0.5,      # 50% dropout in attention
            output_attentions=False,
            output_hidden_states=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        if model_to_load is not None:
            try:
                model.roberta.load_state_dict(torch.load(model_to_load))
                print("Loaded pre-trained model")
            except:
                pass
        return model
    
    def _prepare_optimizer_scheduler(self, model, total_steps, learning_rate=1e-5):
        optimizer = AdamW(model.parameters(),
                        lr=learning_rate,
                        eps=1e-8,
                        weight_decay=1e-2  # Apply weight decay
                        )
        num_warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)
        return optimizer, scheduler
    
    def _evaluate(self, model, validation_dataloader):
        manual_seed = 2022
        torch.manual_seed(manual_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        preds = []
        true_labels = []
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in tqdm(validation_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].type(torch.LongTensor).to(device)

            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
                
            preds.append(logits)
            true_labels.append(label_ids)
            
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        tqdm.write(f"Avg validation loss: {avg_val_loss}")

        return preds, true_labels
    
    def _train(self, model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs):
        manual_seed = 2022
        torch.manual_seed(manual_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        training_stats = []
        model.train()
        total_train_loss = 0

        for epoch in tqdm(range(1, epochs+1)):
            progress_bar = tqdm(train_dataloader, 
                            desc=" Epoch {:1d}".format(epoch),
                            leave=False, # to overwrite each epoch
                            disable=False)

            for batch in progress_bar:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].type(torch.LongTensor).to(device)

                model.zero_grad()
                outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)

                loss = outputs.loss
                logits = outputs.logits

    

                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            tqdm.write(f"\nEpoch: {epoch}")
            tqdm.write(f"Training loss: {avg_train_loss}")

            preds, val_labels = self.evaluate(model, validation_dataloader)
            predictions = np.argmax(np.concatenate(preds, axis=0), axis=1).flatten()
            labels = (np.concatenate(val_labels, axis=0)).flatten()
            
            print(classification_report(labels, predictions))

        print("Training complete!")

    def train_BERTweet(self):
        manual_seed = 2022
        torch.manual_seed(manual_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = pd.read_csv("Pre-processed data.csv")[["text", "Labels"]]
        dataset = dataset[dataset.Labels != 2]

        train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
        dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        train_dataloader, validation_dataloader, test_dataloader = self.prepare_dataloaders(train_data, dev_data, test_data, "vinai/bertweet-large", 16)

        EPOCHS = 2
        NUM_LABELS = 2
        TOTAL_STEPS = len(train_dataloader) * EPOCHS
        model = self.prepare_model(total_labels=NUM_LABELS, model_name="vinai/bertweet-large", model_to_load=None)
        model.to(device)
        optimizer, scheduler = self.prepare_optimizer_scheduler(model, total_steps=TOTAL_STEPS, learning_rate=1e-5)

        self.train(model, optimizer, scheduler, train_dataloader, validation_dataloader, EPOCHS)

        return model
    
    def BERTweet_classify(self):
        # Pre-process unseen data
        self.normalise()
        self.remove_stopwords()
        self.lemmatise()

        # Prepare dataloader for unseen data
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large",use_fast=False, normalization=True)
    
        data = self.encode_data(self.df, tokenizer)
        labels = self.df[self.label].astype(int)
        
        input_ids, attention_masks = data.values()
        test_labels = torch.tensor(labels.values)
        test_dataset = TensorDataset(input_ids, attention_masks, test_labels)
        
        test_dataloader = DataLoader(
                    test_dataset,
                    sampler = RandomSampler(test_dataset), 
                    batch_size = 16
                )

        # Train hate speech classifier
        model = self.train_BERTweet()

        # Evaluate model on unseen data and display results
        preds, true = self.evaluate(model, test_dataloader)

        preds = np.argmax(np.concatenate(preds, axis=0), axis=1).flatten()
        true = (np.concatenate(true, axis=0)).flatten()
            
        print(classification_report(true, preds))
